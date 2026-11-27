#!/usr/bin/env python3
# mt5_fetch_mcp.py
# Copyright Su Nie | BSD-3C License | https://github.com/can87683

"""customtkinter==5.2.2 requests==2.31.0 setproctitle==1.3.3"""

import os
import sys
import setproctitle
setproctitle.setproctitle(os.path.basename(sys.argv[0]))

import base64
import configparser
import csv
import hashlib
import json
import re
import shutil
import socket
import tempfile
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
import customtkinter as ctk
import requests
import tkinter as tk


TOOL_CANDIDATES = {
    "account":           ["get_trading_account_info", "get_account_info"],
    "add_symbol":        ["add_marketwatch_symbol"],
    "charts":            ["list_open_charts"],
    "close_position":    ["trade_close_single_position", "trade_close_by_position"],
    "delete_order":      ["trade_delete_order"],
    "history_orders":    ["get_trading_history_orders"],
    "history_positions": ["get_trading_history_positions"],
    "modify_sl_tp":      ["trade_modify_sl_tp"],
    "order_market":      ["trade_send_market_order"],
    "order_pending":     ["trade_send_pending_order"],
    "positions":         ["get_trading_open_positions", "get_open_positions"],
    "rates":             ["get_chart_history"],
    "remove_symbol":     ["remove_marketwatch_symbol"],
    "symbols":           ["get_marketwatch_symbols"],
    "ticks":             ["get_chart_ticks_history"],
    "time":              ["get_time_information"],
    "workspace":         ["get_workspace_info"],
}

TOOLS_FILE = Path(sys.argv[0]).resolve().parent / "mt5_mcp_tools.json"


class Config:
    """Static configuration and INI merge helper."""

    TF_MAX_BARS = {
        "M5": 200_000,
        "M15": 100_000,
        "H1": 50_000,
        "H4": 30_000,
        "D1": 10_000,
        "W1": 2_000,
    }

    WINDOW_WIDTH = 640
    WINDOW_HEIGHT = 800
    INI_FILE = str(Path(sys.argv[0]).resolve().with_suffix(".ini"))
    LOG_MAX_LINES = 1000
    LOOPBACK_HOSTS = ("127.0.0.1", "localhost", "[::1]")
    DISCONNECT_GRACE = 2

    INI_SECTION_ORDER = ("Window", "Mcp", "Fetch")
    INI_SECTION_WINDOW = "Window"

    TIMEFRAME_ORDER = ["M5", "M15", "H1", "H4", "D1", "W1"]
    TF_MINUTES = {
        "M5": 5, "M15": 15, "H1": 60, "H4": 240,
        "D1": 1440, "W1": 10080,
    }
    DEFAULT_MAX_BARS = 10000
    DATA_DIR = os.path.join(str(Path.home()), "mt5_data")

    WINDOW_X = None
    WINDOW_Y = None

    COLOR_BORDER = "white"
    BORDER_WIDTH = 2

    @staticmethod
    def atomic_merge_ini(path, section, updates):
        disk = configparser.ConfigParser(interpolation=None)
        if os.path.exists(path):
            disk.read(path, encoding="utf-8")
        if not disk.has_section(section):
            disk.add_section(section)
        for key, value in updates.items():
            disk.set(section, key, str(value))

        preferred = [s.lower() for s in Config.INI_SECTION_ORDER]
        ordered = configparser.ConfigParser(interpolation=None)
        ordered.optionxform = str
        for sect in preferred:
            if disk.has_section(sect):
                ordered.add_section(sect)
                for key, value in disk.items(sect):
                    ordered.set(sect, key, value)
        for sect in disk.sections():
            if sect in preferred:
                continue
            ordered.add_section(sect)
            for key, value in disk.items(sect):
                ordered.set(sect, key, value)

        directory = os.path.dirname(os.path.abspath(path)) or "."
        fd, tmp = tempfile.mkstemp(
            prefix=".mcp_", suffix=".ini", dir=directory,
        )
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            ordered.write(handle)
        if os.name == "posix":
            os.chmod(tmp, 0o600)
        os.replace(tmp, path)

    @staticmethod
    def load_window_geometry(path):
        parser = configparser.ConfigParser(interpolation=None)
        if os.path.exists(path):
            parser.read(path, encoding="utf-8")
        if not parser.has_section(Config.INI_SECTION_WINDOW):
            return (None, None)

        geometry = parser.get(
            Config.INI_SECTION_WINDOW, "geometry", fallback="",
        )
        if geometry:
            _size, rest = geometry.split("+", 1)
            x_str, y_str = rest.split("+", 1)
            return (int(x_str), int(y_str))

        x = parser.getint(Config.INI_SECTION_WINDOW, "x", fallback=None)
        y = parser.getint(Config.INI_SECTION_WINDOW, "y", fallback=None)
        return (x, y)

    @staticmethod
    def save_window_geometry(path, geometry):
        parser = configparser.ConfigParser(interpolation=None)
        if os.path.exists(path):
            parser.read(path, encoding="utf-8")
        if parser.has_section(Config.INI_SECTION_WINDOW):
            for key in ("x", "y"):
                if parser.has_option(Config.INI_SECTION_WINDOW, key):
                    parser.remove_option(Config.INI_SECTION_WINDOW, key)

        directory = os.path.dirname(os.path.abspath(path)) or "."
        if os.path.exists(path):
            fd, tmp = tempfile.mkstemp(
                prefix=".mcp_", suffix=".ini", dir=directory,
            )
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                parser.write(handle)
            if os.name == "posix":
                os.chmod(tmp, 0o600)
            os.replace(tmp, path)

        Config.atomic_merge_ini(
            path, Config.INI_SECTION_WINDOW, {"geometry": geometry},
        )


class Mt5McpClient:
    """Pure-python MCP client. No GUI widgets. Thread-safe via lock."""

    AUTO_CONNECT_DEFAULT = True
    AUTO_CONNECT_DELAY_MS = 500
    AUTO_CONNECT_INTERVAL_MS = 5000
    DEFAULT_SERVER_IP = "127.0.0.1"
    DEFAULT_PORT = "22346"
    DEFAULT_ENDPOINT = "/mcp"
    DEFAULT_TOKEN = ""
    SCHEME = "http"
    PROTOCOL_VERSION = "2024-11-05"
    CLIENT_NAME = ""
    CLIENT_VERSION = ""
    HTTP_TIMEOUT = 30.0

    _TOKEN_SALT = b"mcp_client_v1_token_obfuscation"

    BAR_SECONDS = {
        "M1": 60, "M2": 120, "M3": 180, "M4": 240, "M5": 300,
        "M6": 360, "M10": 600, "M12": 720, "M15": 900, "M20": 1200,
        "M30": 1800, "H1": 3600, "H2": 7200, "H3": 10800,
        "H4": 14400, "H6": 21600, "H8": 28800, "H12": 43200,
        "D1": 86400, "W1": 604800, "MN1": 2592000,
    }
    RANGE_BUFFER = 2.0

    MCP_TOOL_LIMIT = 100000
    MCP_PAGE_SIZE = 10000

    DEFAULT_BLACKLIST = ""

    def __init__(self, config_path=None):
        if config_path is None:
            script_name = os.path.splitext(os.path.basename(__file__))[0]
            config_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                f"{script_name}.ini",
            )
        self.ini_path = config_path
        self.ini = configparser.ConfigParser()
        if os.path.exists(self.ini_path):
            self.ini.read(self.ini_path, encoding="utf-8")

        self.scheme = self.SCHEME
        self.server_ip = self.DEFAULT_SERVER_IP
        self.port = self.DEFAULT_PORT
        self.endpoint = self.DEFAULT_ENDPOINT
        self.token = self.DEFAULT_TOKEN
        self.connected = False
        self.protocol_version = None
        self.server_info = None
        self.session_id = None
        self.on_connect_success = None
        self.on_disconnect = None
        self.log_callback = None

        self.market_watch_only = True
        self.blacklist_enabled = True
        self.blacklist_raw = self.DEFAULT_BLACKLIST

        self._closing = False
        self._lock = threading.Lock()

        self._load_state()

    # ------------------------------------------------------------------
    # Token obfuscation (local-only, NOT cryptographic)
    # ------------------------------------------------------------------

    @classmethod
    def _derive_key(cls) -> bytes:
        material = cls._TOKEN_SALT + socket.gethostname().encode("utf-8", "ignore")
        return hashlib.sha256(material).digest()

    @classmethod
    def encrypt_token(cls, plaintext: str) -> str:
        if not plaintext:
            return ""
        key = cls._derive_key()
        data = plaintext.encode("utf-8")
        xored = bytes(b ^ key[i % len(key)] for i, b in enumerate(data))
        return "enc:" + base64.urlsafe_b64encode(xored).decode("ascii")

    @classmethod
    def decrypt_token(cls, stored: str) -> str:
        if not stored:
            return ""
        if not stored.startswith("enc:"):
            return stored
        raw = base64.urlsafe_b64decode(stored[4:].encode("ascii"))
        key = cls._derive_key()
        data = bytes(b ^ key[i % len(key)] for i, b in enumerate(raw))
        return data.decode("utf-8")

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load_state(self):
        if not self.ini.has_section("mcp"):
            self.ini.add_section("mcp")
        self.server_ip = self.ini.get("mcp", "server_ip", fallback=self.DEFAULT_SERVER_IP)
        self.port = self.ini.get("mcp", "port", fallback=self.DEFAULT_PORT)
        self.endpoint = self.ini.get("mcp", "endpoint", fallback=self.DEFAULT_ENDPOINT)
        self.token = self.decrypt_token(
            self.ini.get("mcp", "token", fallback=self.DEFAULT_TOKEN))
        self.auto_connect = self.ini.getboolean("mcp", "auto_connect",
                                                fallback=self.AUTO_CONNECT_DEFAULT)
        self.market_watch_only = self.ini.getboolean("mcp", "market_watch_only", fallback=True)
        self.blacklist_enabled = self.ini.getboolean("mcp", "blacklist_enabled", fallback=True)
        self.blacklist_raw = self.ini.get("mcp", "blacklist", fallback=self.DEFAULT_BLACKLIST)

    def _atomic_write_ini(self):
        disk = configparser.ConfigParser()
        if os.path.exists(self.ini_path):
            disk.read(self.ini_path, encoding="utf-8")
        if not disk.has_section("mcp"):
            disk.add_section("mcp")
        for k, v in self.ini.items("mcp"):
            disk.set("mcp", k, v)

        preferred = ["mcp", "Log", "Window", "Fetch"]
        ordered = configparser.ConfigParser()
        ordered.optionxform = str
        for sect in preferred:
            if disk.has_section(sect):
                ordered.add_section(sect)
                for k, v in disk.items(sect):
                    ordered.set(sect, k, v)
        for sect in disk.sections():
            if sect in preferred:
                continue
            ordered.add_section(sect)
            for k, v in disk.items(sect):
                ordered.set(sect, k, v)

        dirn = os.path.dirname(os.path.abspath(self.ini_path)) or "."
        fd, tmp = tempfile.mkstemp(prefix=".mcp_", suffix=".ini", dir=dirn)
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            ordered.write(f)
        if os.name == "posix":
            os.chmod(tmp, 0o600)
        os.replace(tmp, self.ini_path)

    def save_state(self):
        allowed = {
            "server_ip", "port", "endpoint", "token",
            "auto_connect", "market_watch_only",
            "blacklist", "blacklist_enabled",
        }
        for key in list(self.ini["mcp"].keys()):
            if key not in allowed:
                self.ini.remove_option("mcp", key)
        self.ini.set("mcp", "server_ip", self.server_ip)
        self.ini.set("mcp", "port", self.port)
        self.ini.set("mcp", "endpoint", self.endpoint)
        self.ini.set("mcp", "token", self.encrypt_token(self.token))
        self.ini.set("mcp", "auto_connect", str(self.auto_connect))
        self.ini.set("mcp", "market_watch_only", str(self.market_watch_only))
        self.ini.set("mcp", "blacklist_enabled", str(self.blacklist_enabled))
        self.ini.set("mcp", "blacklist", self.blacklist_raw)
        self._atomic_write_ini()

    def _log(self, message):
        ts = datetime.now().strftime("%H:%M:%S")
        formatted = f"[{ts}] {message}"
        if self.log_callback:
            self.log_callback(formatted)

    def configure_connection(self, server_ip, port, endpoint, token):
        self.server_ip = server_ip
        self.port = port
        self.endpoint = endpoint if endpoint.startswith("/") else f"/{endpoint}"
        self.token = token

    def parse_blacklist(self):
        if not self.blacklist_enabled:
            return set()
        raw = self.blacklist_raw or ""
        return {s.strip() for s in raw.split(",") if s.strip()}

    def filter_symbols(self, symbols):
        blacklist = self.parse_blacklist()
        filtered = [s for s in symbols if s not in blacklist]
        return sorted(set(filtered))

    # ------------------------------------------------------------------
    # MCP transport
    # ------------------------------------------------------------------

    def _mcp_headers(self) -> dict:
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
        }
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        if self.session_id:
            headers["Mcp-Session-Id"] = self.session_id
        return headers

    def _mcp_raw_post(self, payload: dict, timeout: float = None) -> requests.Response:
        url = f"{self.scheme}://{self.server_ip}:{self.port}{self.endpoint}"
        return requests.post(
            url, json=payload, headers=self._mcp_headers(),
            timeout=timeout if timeout is not None else self.HTTP_TIMEOUT)

    def _mcp_parse_response(self, response, request_id=None) -> dict:
        ct = response.headers.get("Content-Type", "")
        if "text/event-stream" not in ct:
            return response.json()
        result = None
        for line in response.text.splitlines():
            line = line.strip()
            if not line or not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if not data:
                continue
            parsed = json.loads(data)
            if not isinstance(parsed, dict):
                continue
            if request_id is not None and parsed.get("id") == request_id:
                return parsed
            result = parsed
        return result

    def mcp_connect(self) -> bool:
        with self._lock:
            if self._closing:
                return False
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            probe.settimeout(2.0)
            refused = probe.connect_ex((self.server_ip, int(self.port)))
        if refused:
            self._log(f"[MCP] {self.server_ip}:{self.port} unreachable (errno {refused}), retrying...")
            return False

        init_payload = {
            "jsonrpc": "2.0", "id": 1, "method": "initialize",
            "params": {
                "protocolVersion": self.PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": {"name": self.CLIENT_NAME, "version": self.CLIENT_VERSION},
            },
        }
        r = self._mcp_raw_post(init_payload, timeout=self.HTTP_TIMEOUT)
        if r.status_code != 200:
            self._log(f"[MCP] initialize failed: HTTP {r.status_code}")
            return False
        sid = r.headers.get("Mcp-Session-Id")
        if sid:
            self.session_id = sid
        body = self._mcp_parse_response(r, request_id=1)
        if not body or "result" not in body:
            self._log("[MCP] initialize returned no result")
            return False
        result = body["result"]
        self.protocol_version = result.get("protocolVersion")
        self.server_info = result.get("serverInfo", {})
        self._mcp_raw_post(
            {"jsonrpc": "2.0", "method": "notifications/initialized"},
            timeout=self.HTTP_TIMEOUT)
        with self._lock:
            self.connected = True
        self._log(f"[MCP] Connected to {self.scheme}://{self.server_ip}:{self.port}{self.endpoint}")
        if callable(self.on_connect_success):
            self.on_connect_success()
        return True

    def mcp_disconnect(self):
        with self._lock:
            self.connected = False
            self.session_id = None

    def mcp_rpc(self, method: str, params: dict = None) -> dict:
        req_id = int(datetime.now().timestamp() * 1_000_000)
        payload = {"jsonrpc": "2.0", "id": req_id, "method": method, "params": params or {}}
        try:
            r = self._mcp_raw_post(payload)
        except Exception as e:
            with self._lock:
                self.connected = False
                self.session_id = None
            self._log(f"[MCP] {method} -> {e}")
            raise RuntimeError(f"{method}: transport error: {e}") from e
        if r.status_code != 200:
            with self._lock:
                self.connected = False
                self.session_id = None
            self._log(f"[MCP] {method} -> HTTP {r.status_code}")
            raise RuntimeError(f"{method}: HTTP {r.status_code}")
        body = self._mcp_parse_response(r, request_id=req_id)
        if not body:
            raise RuntimeError(f"{method}: empty/unmatched response body")
        if "error" in body:
            self._log(f"[MCP] {method} error: {body['error']}")
            raise RuntimeError(f"{method}: RPC error: {body['error']}")
        return body.get("result")

    def mcp_list_tools(self) -> list:
        result = self.mcp_rpc("tools/list", {})
        return result["tools"] if result and "tools" in result else []

    def mcp_call_tool(self, name: str, arguments: dict = None) -> dict:
        return self.mcp_rpc("tools/call", {"name": name, "arguments": arguments or {}})

    def mcp_list_resources(self) -> list:
        result = self.mcp_rpc("resources/list", {})
        return result["resources"] if result and "resources" in result else []

    def mcp_list_prompts(self) -> list:
        result = self.mcp_rpc("prompts/list", {})
        return result["prompts"] if result and "prompts" in result else []

    # ==================================================================
    # Public MCP tool API
    # ==================================================================

    def _unwrap_tool_text(self, envelope, label):
        if envelope.get("isError"):
            content = envelope.get("content") or []
            text = content[0].get("text", "") if content else "(no detail)"
            raise RuntimeError(f"{label}: MCP tool error: {text}")
        content = envelope.get("content") or []
        if not content:
            raise RuntimeError(f"{label}: MCP tool returned empty content")
        text = content[0].get("text", "")
        if not text:
            raise RuntimeError(f"{label}: empty tool payload")
        return json.loads(text)

    def _fetch_tool(self, name, arguments):
        return self._unwrap_tool_text(self.mcp_call_tool(name, arguments), name)

    def get_chart_history(self, symbol, timeframe, datetime_from, datetime_to, limit=None):
        if limit is None:
            limit = self.MCP_TOOL_LIMIT
        payload = self._fetch_tool("get_chart_history", {
            "symbol": symbol, "period": timeframe,
            "datetime_from": datetime_from, "datetime_to": datetime_to,
            "limit": limit,
        })
        if not payload.get("ok", False):
            return []
        return payload.get("history", []) or []

    def get_rates(self, symbol, timeframe, count):
        seconds = self.BAR_SECONDS.get(timeframe)
        if seconds is None:
            raise ValueError(f"Unknown timeframe: {timeframe}")
        want = int(count)
        span = int(seconds * want * self.RANGE_BUFFER)
        dt_to = datetime.now(timezone.utc).replace(microsecond=0) + timedelta(days=1)
        dt_from = dt_to - timedelta(seconds=span)
        history = self.get_chart_history(symbol, timeframe, dt_from.isoformat(), dt_to.isoformat())
        return history[-want:] if len(history) > want else history

    def get_market_watch_symbols(self):
        args = {} if self.market_watch_only else {"include_hidden": True}
        payload = self._fetch_tool("get_marketwatch_symbols", args)
        records = payload.get("symbols", []) or []
        out = {str(r.get("symbol")) for r in records if isinstance(r, dict) and r.get("symbol")}
        blacklist = self.parse_blacklist()
        if blacklist:
            out -= blacklist
        return sorted(out)

    def get_all_symbols(self):
        payload = self._fetch_tool("get_marketwatch_symbols", {"include_hidden": True, "limit": 100000})
        records = payload.get("symbols", []) or []
        out = {str(r.get("symbol")) for r in records if isinstance(r, dict) and r.get("symbol")}
        return sorted(out)

    def get_symbol_info(self, symbol):
        payload = self._fetch_tool("get_marketwatch_symbols", {"symbol": symbol})
        records = payload.get("symbols", []) or []
        return records[0] if records else {}

    def get_current_price(self, symbol):
        rec = self.get_symbol_info(symbol)
        if not rec:
            return {}
        return {
            "symbol": rec.get("symbol", symbol),
            "bid": rec.get("bid"),
            "ask": rec.get("ask"),
            "last": rec.get("price_close"),
            "time": rec.get("update_time"),
        }

    def add_marketwatch_symbol(self, symbol, show=True):
        return self._fetch_tool("add_marketwatch_symbol", {"symbol": symbol, "show": show})

    def remove_marketwatch_symbol(self, symbol):
        return self._fetch_tool("remove_marketwatch_symbol", {"symbol": symbol})

    def get_account_info(self):
        return self._fetch_tool("get_trading_account_info", {})

    def get_account_balance(self):
        return float(self.get_account_info()["balance"])

    def get_account_equity(self):
        return float(self.get_account_info()["equity"])

    def get_account_margin(self):
        return float(self.get_account_info()["margin"])

    def get_account_free_margin(self):
        return float(self.get_account_info()["free_margin"])

    def get_open_positions_and_orders(self, symbol=""):
        args = {"include_orders": True}
        if symbol:
            args["symbol"] = symbol
        return self._fetch_tool("get_trading_open_positions", args)

    def get_positions(self, symbol=""):
        return self.get_open_positions_and_orders(symbol).get("positions", []) or []

    def get_orders(self, symbol=""):
        return self.get_open_positions_and_orders(symbol).get("orders", []) or []

    def get_trading_history_positions(self, datetime_from, datetime_to, symbol=None, limit=100000):
        args = {"datetime_from": datetime_from, "datetime_to": datetime_to, "limit": limit}
        if symbol:
            args["symbol"] = symbol
        return self._fetch_tool("get_trading_history_positions", args)

    def get_trading_history_orders(self, datetime_from, datetime_to, symbol=None,
                                   include_orders=True, include_orders_canceled=False,
                                   include_deals=True, limit=100000):
        args = {"datetime_from": datetime_from, "datetime_to": datetime_to,
                "include_orders": include_orders,
                "include_orders_canceled": include_orders_canceled,
                "include_deals": include_deals, "limit": limit}
        if symbol:
            args["symbol"] = symbol
        return self._fetch_tool("get_trading_history_orders", args)

    def trade_send_market_order(self, symbol, side, volume, sl=None, tp=None, comment=""):
        if side.upper() not in ("BUY", "SELL"):
            raise ValueError(f"Invalid market side: {side}")
        args = {"symbol": symbol, "type": side.lower(), "volume": float(volume)}
        if sl is not None:
            args["sl"] = float(sl)
        if tp is not None:
            args["tp"] = float(tp)
        if comment:
            args["comment"] = str(comment)[:31]
        return self._fetch_tool("trade_send_market_order", args)

    def trade_send_pending_order(self, symbol, order_type, volume, price, sl=None, tp=None,
                                 stoplimit=None, filling_type=None, expiration_type="gtc",
                                 expiration_time=None, comment=""):
        valid = {"buy_limit", "sell_limit", "buy_stop", "sell_stop", "buy_stop_limit", "sell_stop_limit"}
        if order_type not in valid:
            raise ValueError(f"Invalid pending type: {order_type}")
        args = {"symbol": symbol, "type": order_type, "volume": float(volume), "price": float(price)}
        if stoplimit is not None:
            args["stoplimit"] = float(stoplimit)
        if sl is not None:
            args["sl"] = float(sl)
        if tp is not None:
            args["tp"] = float(tp)
        if filling_type:
            args["filling_type"] = filling_type
        if expiration_type:
            args["expiration_type"] = expiration_type
        if expiration_time:
            args["expiration_time"] = expiration_time
        if comment:
            args["comment"] = str(comment)[:31]
        return self._fetch_tool("trade_send_pending_order", args)

    def trade_modify_sl_tp(self, symbol, sl=None, tp=None, position_ticket=None, order_ticket=None):
        if (position_ticket is None) == (order_ticket is None):
            raise ValueError("trade_modify_sl_tp: provide exactly one of position_ticket or order_ticket")
        args = {"symbol": symbol}
        if position_ticket is not None:
            args["position_ticket"] = float(position_ticket)
        if order_ticket is not None:
            args["order_ticket"] = float(order_ticket)
        if sl is not None:
            args["sl"] = float(sl)
        if tp is not None:
            args["tp"] = float(tp)
        return self._fetch_tool("trade_modify_sl_tp", args)

    def trade_delete_order(self, symbol, order_ticket):
        return self._fetch_tool("trade_delete_order", {"symbol": symbol, "order_ticket": float(order_ticket)})

    def trade_close_single_position(self, symbol, position_ticket):
        return self._fetch_tool("trade_close_single_position",
                                {"symbol": symbol, "position_ticket": float(position_ticket)})

    def trade_close_by_position(self, symbol, position_ticket, position_ticket_by):
        return self._fetch_tool("trade_close_by_position", {
            "symbol": symbol,
            "position_ticket": float(position_ticket),
            "position_ticket_by": float(position_ticket_by),
        })

    def get_time_information(self):
        return self._fetch_tool("get_time_information", {})

    def economic_calendar_list_values(self, datetime_from, datetime_to, event_id=None, limit=1000):
        args = {"datetime_from": datetime_from, "datetime_to": datetime_to, "limit": limit}
        if event_id:
            args["event_id"] = str(event_id)
        return self._fetch_tool("economic_calendar_list_values", args)

    def economic_calendar_list_values_last(self, change_id, event_id=None):
        args = {"change_id": str(change_id)}
        if event_id:
            args["event_id"] = str(event_id)
        return self._fetch_tool("economic_calendar_list_values_last", args)

    def get_terminal_journal(self, datetime_from, datetime_to, filter_text=None, limit=None):
        args = {"datetime_from": datetime_from, "datetime_to": datetime_to}
        if filter_text:
            args["filter"] = filter_text
        if limit is not None:
            args["limit"] = int(limit)
        return self._fetch_tool("get_terminal_journal", args)

    def get_expert_journal(self, datetime_from, datetime_to, mql5_program_name=None,
                           filter_text=None, limit=None):
        args = {"datetime_from": datetime_from, "datetime_to": datetime_to}
        if mql5_program_name:
            args["mql5_program_name"] = mql5_program_name
        if filter_text:
            args["filter"] = filter_text
        if limit is not None:
            args["limit"] = int(limit)
        return self._fetch_tool("get_expert_journal", args)

    def get_tester_journal(self, datetime_from, datetime_to, filter_text=None, limit=None):
        args = {"datetime_from": datetime_from, "datetime_to": datetime_to}
        if filter_text:
            args["filter"] = filter_text
        if limit is not None:
            args["limit"] = int(limit)
        return self._fetch_tool("get_tester_journal", args)

    def stop(self):
        self._closing = True
        self.mcp_disconnect()


class Mt5FetchData:
    """Pure-python historical bar fetcher. No GUI widgets."""

    DEFAULT_DATA_FOLDER = os.path.join(str(Path.home()), "mt5_data")
    DEFAULT_FROM_OFFSET_DAYS = 30

    TIMEFRAME_ORDER = ["M5", "M15", "H1", "H4", "D1", "W1"]
    TF_MINUTES = {
        "M1": 1, "M2": 2, "M3": 3, "M4": 4, "M5": 5,
        "M6": 6, "M10": 10, "M12": 12, "M15": 15,
        "M20": 20, "M30": 30,
        "H1": 60, "H2": 120, "H3": 180, "H4": 240,
        "H6": 360, "H8": 480, "H12": 720,
        "D1": 1440, "W1": 10080, "MN1": 43200,
    }
    TF_MAX_BARS = {
        "M1": 500_000, "M5": 200_000, "M15": 100_000, "H1": 50_000,
        "H4": 30_000, "D1": 5_000, "W1": 1_000, "MN1": 240,
    }
    RANGE_RETRY_STEPS = 4
    TABLE_COL_WIDTH = 8
    CSV_COLUMNS = ("time", "open", "high", "low", "close", "tick_volume")

    TOOL_CANDIDATES = {
        "symbols": ["get_marketwatch_symbols", "market_watch_get_symbols", "get_symbols"],
        "rates": ["get_chart_history", "market_data_get_candles", "get_candles", "get_bars", "get_rates"],
    }

    def __init__(self, mcp_client, config_path=None):
        self.mcp_client = mcp_client
        if config_path is None:
            script_name = os.path.splitext(os.path.basename(__file__))[0]
            config_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), f"{script_name}.ini")
        self.ini_path = config_path
        self.ini = configparser.ConfigParser(interpolation=None)
        if os.path.exists(self.ini_path):
            self.ini.read(self.ini_path, encoding="utf-8")

        self._tool_map = {}
        self._tool_registry = {}
        self.data_folder_path = self.DEFAULT_DATA_FOLDER
        self.date_from = None
        self.date_to = None
        self.selected_timeframes = list(self.TIMEFRAME_ORDER)

        self._load_fetch_state()

    def _load_fetch_state(self):
        today = datetime.now(timezone.utc).date()
        from_date = today - timedelta(days=self.DEFAULT_FROM_OFFSET_DAYS)
        saved_from = f"{from_date.year:04d}-{from_date.month:02d}-{from_date.day:02d}"
        saved_to = f"{today.year:04d}-{today.month:02d}-{today.day:02d}"

        if self.ini.has_section("Fetch"):
            self.data_folder_path = self.ini.get("Fetch", "data_folder", fallback=self.DEFAULT_DATA_FOLDER)
            saved_from = self.ini.get("Fetch", "date_from", fallback=saved_from)
            saved_to = self.ini.get("Fetch", "date_to", fallback=saved_to)
            saved_tfs = [tf.strip() for tf in self.ini.get(
                "Fetch", "fetch_timeframes", fallback=",".join(self.TIMEFRAME_ORDER)).split(",") if tf.strip()]
            self.selected_timeframes = [tf for tf in self.TIMEFRAME_ORDER if tf in saved_tfs] or list(self.TIMEFRAME_ORDER)

        fy, fm, fd = saved_from.split("-")
        ty, tm, td = saved_to.split("-")
        self.date_from = datetime(int(fy), int(fm), int(fd), tzinfo=timezone.utc)
        self.date_to = datetime(int(ty), int(tm), int(td), tzinfo=timezone.utc) + timedelta(days=1) - timedelta(seconds=1)

    def save_config(self):
        date_from_str = self.date_from.strftime("%Y-%m-%d")
        date_to_str = (self.date_to - timedelta(days=1)).strftime("%Y-%m-%d")
        Config.atomic_merge_ini(self.ini_path, "Fetch", {
            "date_from": date_from_str,
            "date_to": date_to_str,
            "fetch_timeframes": ",".join(self.selected_timeframes),
            "data_folder": self.data_folder_path,
        })

    def _resolve_tools(self):
        if self._tool_map:
            return
        tools = self.mcp_client.mcp_list_tools() or []
        registry = {}
        for tool in tools:
            if isinstance(tool, dict) and tool.get("name"):
                registry[tool["name"]] = tool
        self._tool_registry = registry
        resolved = {}
        missing = []
        for key, candidates in self.TOOL_CANDIDATES.items():
            hit = next((c for c in candidates if c in registry), None)
            if hit:
                resolved[key] = hit
            else:
                missing.append(key)
        self._tool_map = resolved
        if missing:
            self.mcp_client._log(f"[tools] unresolved in Fetch: {', '.join(missing)}")

    def _call_tool(self, semantic_key, args):
        self._resolve_tools()
        name = self._tool_map.get(semantic_key)
        if not name:
            raise RuntimeError(f"[fetch] no server tool resolved for '{semantic_key}'")
        return self.mcp_client.mcp_call_tool(name, args or {})

    @staticmethod
    def _unwrap_tool_text(envelope, label):
        if envelope is None:
            raise RuntimeError(f"{label}: empty response")
        if envelope.get("isError"):
            content = envelope.get("content") or []
            text = ""
            if content and isinstance(content[0], dict):
                text = content[0].get("text", "")
            raise RuntimeError(f"{label}: MCP tool error: {text}")
        structured = envelope.get("structuredContent")
        if isinstance(structured, (dict, list)):
            return structured
        content = envelope.get("content") or []
        if not content or not isinstance(content[0], dict):
            raise RuntimeError(f"{label}: unexpected MCP envelope")
        text = content[0].get("text", "")
        if not text:
            raise RuntimeError(f"{label}: empty tool payload")
        return json.loads(text)

    @staticmethod
    def _extract_bars(payload):
        if isinstance(payload, list):
            return [b for b in payload if isinstance(b, dict)]
        if not isinstance(payload, dict):
            return []
        for key in ("data", "bars", "items", "candles", "history", "rates", "ohlc", "result"):
            value = payload.get(key)
            if isinstance(value, list):
                return [b for b in value if isinstance(b, dict)]
        return []

    def _estimate_bars(self, timeframe, dt_from, dt_to):
        tf_min = self.TF_MINUTES.get(timeframe, 5)
        total_minutes = max(1, int((dt_to - dt_from).total_seconds() // 60))
        cap = self.TF_MAX_BARS.get(timeframe)
        count = total_minutes // tf_min + 5
        return min(count, cap) if cap else count

    def _build_rates_request(self, symbol, timeframe, dt_from, dt_to):
        return {
            "symbol": symbol,
            "period": timeframe,
            "datetime_from": dt_from.strftime("%Y-%m-%dT%H:%M:%S"),
            "datetime_to": dt_to.strftime("%Y-%m-%dT%H:%M:%S"),
            "limit": self._estimate_bars(timeframe, dt_from, dt_to),
        }

    def _is_range_error(self, envelope):
        content = envelope.get("content") or []
        text = content[0].get("text", "") if content else ""
        return "invalid datetime range" in text.lower()

    def fetch_rates(self, symbol, timeframe, dt_from=None, dt_to=None):
        if dt_from is None:
            dt_from = self.date_from
        if dt_to is None:
            dt_to = self.date_to
        dt_to_attempt = dt_to
        for step in range(self.RANGE_RETRY_STEPS + 1):
            request = self._build_rates_request(symbol, timeframe, dt_from, dt_to_attempt)
            envelope = self._call_tool("rates", request)
            if not envelope.get("isError"):
                payload = self._unwrap_tool_text(envelope, self._tool_map.get("rates", "rates"))
                return self._extract_bars(payload)
            if not self._is_range_error(envelope):
                content = envelope.get("content") or []
                err = content[0].get("text", "") if content else ""
                raise RuntimeError(f"[rates] {symbol} {timeframe}: {err}")
            if step + 1 > self.RANGE_RETRY_STEPS:
                raise RuntimeError(f"[rates] {symbol} {timeframe}: range errors after retries")
            dt_to_attempt = dt_from + (dt_to_attempt - dt_from) * 0.5
        raise RuntimeError(f"[rates] {symbol} {timeframe}: range errors after retries")

    def fetch_symbols(self):
        mw_only = self.mcp_client.market_watch_only
        args = {} if mw_only else {"include_hidden": True}
        envelope = self._call_tool("symbols", args)
        payload = self._unwrap_tool_text(envelope, self._tool_map.get("symbols", "symbols"))
        if isinstance(payload, dict):
            records = payload.get("symbols", []) or []
        elif isinstance(payload, list):
            records = payload
        else:
            records = []
        symbols = {record.get("symbol") for record in records if isinstance(record, dict) and record.get("symbol")}
        return sorted(symbols)

    def _save_to_csv(self, symbol, timeframe, data):
        safe_symbol = re.sub(r"[^A-Za-z0-9_+#!$-]", "_", symbol)
        symbol_dir = os.path.join(self.data_folder_path, safe_symbol)
        os.makedirs(symbol_dir, exist_ok=True)
        rows = []
        for bar in data:
            row = {}
            for col in self.CSV_COLUMNS:
                if col in bar:
                    row[col] = bar[col]
                elif col == "tick_volume" and "volume" in bar:
                    row["tick_volume"] = bar["volume"]
            if row:
                rows.append(row)
        if not rows:
            return
        filepath = os.path.join(symbol_dir, f"{safe_symbol}_{timeframe}.csv")
        with open(filepath, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(self.CSV_COLUMNS), lineterminator=os.linesep)
            writer.writeheader()
            writer.writerows(rows)
        self.mcp_client._log(f"💾 Saved to {filepath}")

    def wipe_data_folder(self):
        if os.path.isdir(self.data_folder_path):
            shutil.rmtree(self.data_folder_path)
            self.mcp_client._log(f"🧹 Removed {self.data_folder_path}")
        else:
            self.mcp_client._log(f"🧹 No existing data folder at {self.data_folder_path}")

    def fetch_single(self, symbol):
        tfs = self.selected_timeframes
        if not tfs:
            raise RuntimeError("Fetch: no timeframe selected")
        self.wipe_data_folder()
        for tf in tfs:
            data = self.fetch_rates(symbol, tf)
            if data:
                self._save_to_csv(symbol, tf, data)
            time.sleep(0.1)

    def fetch_all(self):
        symbols = self.fetch_symbols()
        symbols = self.mcp_client.filter_symbols(symbols)
        if not symbols:
            raise RuntimeError("Fetch: no Market Watch symbols available")
        tfs = self.selected_timeframes
        if not tfs:
            raise RuntimeError("Fetch: no timeframe selected")
        self.wipe_data_folder()
        for idx, symbol in enumerate(symbols, 1):
            self.mcp_client._log(f"📈 {idx}/{len(symbols)}: {symbol}")
            for tf in tfs:
                data = self.fetch_rates(symbol, tf)
                if data:
                    self._save_to_csv(symbol, tf, data)
                time.sleep(0.1)
            time.sleep(0.2)
        self.mcp_client._log(f"✅ Fetch complete: {len(symbols)} symbols")

    def scan_available_bars(self):
        symbols = self.fetch_symbols()
        symbols = self.mcp_client.filter_symbols(symbols)
        if not symbols:
            raise RuntimeError("Available Bars: no Market Watch symbols available")
        tfs = self.selected_timeframes
        if not tfs:
            raise RuntimeError("Available Bars: no timeframe selected")
        self.mcp_client._log(
            f"🔍 Scanning available bars for {len(symbols)} symbols "
            f"{self.date_from:%Y-%m-%d} → {(self.date_to - timedelta(days=1)):%Y-%m-%d}")
        header = "TF:" + "".join(" " * ((self.TABLE_COL_WIDTH - len(tf)) // 2) + tf for tf in self.TIMEFRAME_ORDER)
        self.mcp_client._log(header)
        for idx, symbol in enumerate(symbols, 1):
            counts = {}
            for tf in tfs:
                data = self.fetch_rates(symbol, tf)
                counts[tf] = len(data)
                time.sleep(0.05)
            row = symbol.ljust(12)
            for tf in self.TIMEFRAME_ORDER:
                n = counts.get(tf, 0)
                text = self._fmt_bars(n) if n > 0 else "N/A"
                row += " " * (self.TABLE_COL_WIDTH - len(text)) + text
            self.mcp_client._log(f"{idx}/{len(symbols)} {row}")
            time.sleep(0.1)
        self.mcp_client._log(f"✅ Available bars scan complete: {len(symbols)} symbols")

    @staticmethod
    def _fmt_bars(n):
        if n >= 1_000_000:
            return f"{n / 1_000_000:.1f}M"
        if n >= 1_000:
            return f"{n / 1_000:.1f}K"
        return str(n)


class Mt5FetchMcpGUI(ctk.CTk):
    """Top-level application window; owns all GUI code and lifecycle."""

    TITLE = "MT5 Fetch MCP - Copyright Su Nie | BSD-3C License | https://github.com/can87683"
    APPEARANCE = "dark"
    THEME = "blue"
    BG_COLOR = "#1c1c1c"
    TITLE_FONT = ("TkDefaultFont", 14, "bold")
    TITLE_COLOR = "#ffff00"
    UI_FONT = ("TkDefaultFont", 12)
    TEXT_COLOR = "#ffffff"
    BUTTON_FG = "white"
    BUTTON_BG = "blue"
    BUTTON_BORDER_WIDTH = 2
    BUTTON_ROUNDNESS = 5
    BUTTON_FONT = ("TkDefaultFont", 12, "bold")
    LOG_FONT = ("Consolas", 10)
    LOG_BG_COLOR = "black"
    LOG_TAIL_CHARS = 500000
    LOG_AUTOSCROLL = True
    COLOR_STATUS_OK = "#4CAF50"
    COLOR_STATUS_BAD = "#F44336"
    COLOR_ALL_BTN = "#4285F4"
    COLOR_ALL_BTN_TEXT = "yellow"
    COLOR_SINGLE_BTN = "#E67E22"
    COLOR_SINGLE_BTN_TEXT = "white"

    def __init__(self):
        super().__init__()
        self.ini_path = Config.INI_FILE
        ctk.set_appearance_mode(self.APPEARANCE)
        ctk.set_default_color_theme(self.THEME)
        self.title(self.TITLE)
        self.resizable(False, False)
        self.configure(fg_color=self.BG_COLOR)

        saved_x, saved_y = Config.load_window_geometry(self.ini_path)
        self._apply_geometry(saved_x, saved_y)

        self.grid_rowconfigure(0, weight=0)
        self.grid_rowconfigure(1, weight=0)
        self.grid_rowconfigure(2, weight=1)
        self.grid_columnconfigure(0, weight=1)

        self._build_bottom_log()
        self._build_connection_frame()
        self._build_fetch_frame()

        self.mcp_client = Mt5McpClient(config_path=self.ini_path)
        self.mcp_client.log_callback = self._log_append
        self.mcp_client.on_connect_success = lambda: self.after(0, self._on_mcp_connected)

        self.fetch_data = Mt5FetchData(self.mcp_client, config_path=self.ini_path)

        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self._start_auto_connect()

    # ------------------------------------------------------------------
    # Geometry
    # ------------------------------------------------------------------

    def _apply_geometry(self, saved_x, saved_y):
        width = Config.WINDOW_WIDTH
        height = Config.WINDOW_HEIGHT
        screen_w = self.winfo_screenwidth()
        screen_h = self.winfo_screenheight()
        if saved_x is None or saved_y is None:
            x = (screen_w - width) // 2
            y = (screen_h - height) // 2
        else:
            x, y = saved_x, saved_y
        x = max(0, min(x, screen_w - width))
        y = max(0, min(y, screen_h - height))
        self.geometry(f"{width}x{height}+{x}+{y}")

    # ------------------------------------------------------------------
    # Bottom log frame (only log widget)
    # ------------------------------------------------------------------

    def _build_bottom_log(self):
        log_frame = ctk.CTkFrame(self, border_width=2, border_color=Config.COLOR_BORDER, fg_color=self.BG_COLOR)
        log_frame.grid(row=2, column=0, sticky="nsew", padx=2, pady=(2, 2))

        ctk.CTkLabel(log_frame, text="Log", font=self.TITLE_FONT, text_color=self.TITLE_COLOR, anchor="w").pack(fill="x", padx=2, pady=(2, 2))

        self.log_text = ctk.CTkTextbox(log_frame, font=self.LOG_FONT, wrap="word", border_width=0, fg_color=self.LOG_BG_COLOR)
        self.log_text.pack(fill="both", expand=True, padx=2, pady=(0, 2))
        self.log_text.configure(state="disabled")

        menu = tk.Menu(self.log_text, tearoff=0)
        menu.add_command(label="Copy", command=self._log_copy)
        menu.add_command(label="Clear", command=self._log_clear)
        self.log_text.bind("<Button-3>", lambda e: menu.tk_popup(e.x_root, e.y_root))

    def _log_append(self, message):
        box = self.log_text
        box.configure(state="normal")
        at_bottom = box.yview()[1] >= 0.999
        box.insert("end", f"{message}\n")
        lines = int(box.index("end-1c").split(".")[0])
        if lines > Config.LOG_MAX_LINES:
            box.delete("1.0", f"{lines - Config.LOG_MAX_LINES}.0")
        if at_bottom and self.LOG_AUTOSCROLL:
            box.see("end")
        box.configure(state="disabled")

    def _log_copy(self):
        text = (self.log_text.selection_get() if self.log_text.tag_ranges("sel")
                else self.log_text.get("1.0", "end-1c"))
        self.clipboard_clear()
        self.clipboard_append(text)

    def _log_clear(self):
        self.log_text.configure(state="normal")
        self.log_text.delete("1.0", "end")
        self.log_text.configure(state="disabled")

    # ------------------------------------------------------------------
    # Connection frame
    # ------------------------------------------------------------------

    def _build_connection_frame(self):
        conn_frame = ctk.CTkFrame(self, border_width=2, border_color=Config.COLOR_BORDER, fg_color=self.BG_COLOR)
        conn_frame.grid(row=0, column=0, sticky="ew", padx=2, pady=(2, 2))

        ctk.CTkLabel(conn_frame, text="MT5 MCP Connection", font=self.TITLE_FONT, text_color=self.TITLE_COLOR, anchor="w").pack(fill="x", padx=2, pady=2)

        ip_row = ctk.CTkFrame(conn_frame, fg_color=self.BG_COLOR, border_width=2)
        ip_row.pack(fill="x", padx=0, pady=0)
        ip_row.pack_propagate(False)
        ip_row.configure(height=34)

        ctk.CTkLabel(ip_row, text="Server IP:", font=self.UI_FONT, text_color=self.TEXT_COLOR).pack(side="left", padx=2)
        self.server_ip_var = ctk.StringVar(value=self.mcp_client.server_ip if hasattr(self, 'mcp_client') else "127.0.0.1")
        ctk.CTkEntry(ip_row, textvariable=self.server_ip_var, width=90, font=self.UI_FONT, border_width=2, text_color=self.TEXT_COLOR).pack(side="left", padx=2)

        ctk.CTkLabel(ip_row, text="Port:", font=self.UI_FONT, text_color=self.TEXT_COLOR).pack(side="left", padx=2)
        self.port_var = ctk.StringVar(value="22346")
        ctk.CTkEntry(ip_row, textvariable=self.port_var, width=70, font=self.UI_FONT, border_width=2, text_color=self.TEXT_COLOR).pack(side="left", padx=2)

        ctk.CTkLabel(ip_row, text="Endpoint:", font=self.UI_FONT, text_color=self.TEXT_COLOR).pack(side="left", padx=2)
        self.endpoint_var = ctk.StringVar(value="/mcp")
        ctk.CTkEntry(ip_row, textvariable=self.endpoint_var, width=70, font=self.UI_FONT, border_width=2, text_color=self.TEXT_COLOR).pack(side="left", padx=2)

        token_row = ctk.CTkFrame(conn_frame, fg_color=self.BG_COLOR, border_width=2)
        token_row.pack(fill="x", padx=0, pady=0)
        token_row.pack_propagate(False)
        token_row.configure(height=34)

        ctk.CTkLabel(token_row, text="Token:", font=self.UI_FONT, text_color=self.TEXT_COLOR).pack(side="left", padx=2)
        self.token_var = ctk.StringVar(value="")
        self.token_entry = ctk.CTkEntry(token_row, textvariable=self.token_var, show="*", font=self.UI_FONT, border_width=2, text_color=self.TEXT_COLOR)
        self.token_entry.pack(side="left", fill="x", expand=True, padx=2)

        self.token_visible = False
        self.token_toggle = ctk.CTkButton(token_row, text="👁", width=30, font=self.UI_FONT, fg_color=self.BUTTON_BG, text_color=self.BUTTON_FG, border_width=self.BUTTON_BORDER_WIDTH, corner_radius=self.BUTTON_ROUNDNESS, command=self._toggle_token)
        self.token_toggle.pack(side="left", padx=2)

        ctk.CTkButton(token_row, text="🧹 Clean", width=70, font=self.UI_FONT, fg_color=self.BUTTON_BG, text_color=self.BUTTON_FG, border_width=self.BUTTON_BORDER_WIDTH, corner_radius=self.BUTTON_ROUNDNESS, command=self._clean_token).pack(side="left", padx=2)
        ctk.CTkButton(token_row, text="💾 Save", width=70, font=self.UI_FONT, fg_color=self.BUTTON_BG, text_color=self.BUTTON_FG, border_width=self.BUTTON_BORDER_WIDTH, corner_radius=self.BUTTON_ROUNDNESS, command=self._save_token).pack(side="left", padx=2)
        ctk.CTkButton(token_row, text="🗑", width=30, font=self.UI_FONT, fg_color=self.BUTTON_BG, text_color=self.BUTTON_FG, border_width=self.BUTTON_BORDER_WIDTH, corner_radius=self.BUTTON_ROUNDNESS, command=self._clear_saved_token).pack(side="left", padx=2)

        options_row = ctk.CTkFrame(conn_frame, fg_color=self.BG_COLOR, border_width=2)
        options_row.pack(fill="x", padx=0, pady=0)
        options_row.pack_propagate(False)
        options_row.configure(height=34)

        self.market_watch_only_var = ctk.BooleanVar(value=True)
        ctk.CTkCheckBox(options_row, text="Market Watch Only", variable=self.market_watch_only_var, font=self.UI_FONT, text_color=self.TEXT_COLOR, command=self._on_options_change).pack(side="left", padx=(2, 12))

        self.blacklist_enabled_var = ctk.BooleanVar(value=True)
        ctk.CTkCheckBox(options_row, text="Blacklist:", variable=self.blacklist_enabled_var, font=self.UI_FONT, text_color=self.TEXT_COLOR, command=self._on_options_change).pack(side="left", padx=2)

        self.blacklist_var = ctk.StringVar(value="")
        bl_entry = ctk.CTkEntry(options_row, textvariable=self.blacklist_var, font=self.UI_FONT, border_width=2, text_color=self.TEXT_COLOR, placeholder_text="SYMBOL1,SYMBOL2,SYMBOL3")
        bl_entry.pack(side="left", fill="x", expand=True, padx=2)
        bl_entry.bind("<Return>", lambda e: self._on_blacklist_edit())
        bl_entry.bind("<FocusOut>", lambda e: self._on_blacklist_edit())

        btn_row = ctk.CTkFrame(conn_frame, fg_color=self.BG_COLOR, border_width=2)
        btn_row.pack(fill="x", padx=0, pady=0)
        btn_row.pack_propagate(False)
        btn_row.configure(height=34)

        self.auto_connect_var = ctk.BooleanVar(value=True)
        ctk.CTkCheckBox(btn_row, text="Auto Connect", variable=self.auto_connect_var, font=self.UI_FONT, text_color=self.TEXT_COLOR, command=self._on_auto_connect_change).pack(side="left", padx=2)

        self.connect_btn = ctk.CTkButton(btn_row, text="Connect", command=self._manual_connect, width=100, fg_color=self.BUTTON_BG, text_color=self.BUTTON_FG, font=self.BUTTON_FONT, border_width=self.BUTTON_BORDER_WIDTH, corner_radius=self.BUTTON_ROUNDNESS)
        self.connect_btn.pack(side="left", padx=2)

        ctk.CTkButton(btn_row, text="List Capabilities", command=self._list_mcp_tools, width=130, fg_color=self.BUTTON_BG, text_color=self.BUTTON_FG, font=self.BUTTON_FONT, border_width=self.BUTTON_BORDER_WIDTH, corner_radius=self.BUTTON_ROUNDNESS).pack(side="left", padx=2)
        ctk.CTkButton(btn_row, text="Symbols", command=self._show_marketwatch_symbols, width=80, fg_color=self.BUTTON_BG, text_color=self.BUTTON_FG, font=self.BUTTON_FONT, border_width=self.BUTTON_BORDER_WIDTH, corner_radius=self.BUTTON_ROUNDNESS).pack(side="left", padx=2)
        ctk.CTkButton(btn_row, text="Schema", command=self._show_tool_schemas, width=80, fg_color=self.BUTTON_BG, text_color=self.BUTTON_FG, font=self.BUTTON_FONT, border_width=self.BUTTON_BORDER_WIDTH, corner_radius=self.BUTTON_ROUNDNESS).pack(side="left", padx=2)
        ctk.CTkButton(btn_row, text="Clear Log", command=self._log_clear, width=80, fg_color=self.BUTTON_BG, text_color=self.BUTTON_FG, font=self.BUTTON_FONT, border_width=self.BUTTON_BORDER_WIDTH, corner_radius=self.BUTTON_ROUNDNESS).pack(side="left", padx=2)

        self.conn_status_dot = ctk.CTkLabel(btn_row, text="●", text_color=self.COLOR_STATUS_BAD, font=self.UI_FONT)
        self.conn_status_dot.pack(side="left", padx=6)

    # ------------------------------------------------------------------
    # Fetch Data frame
    # ------------------------------------------------------------------

    def _build_fetch_frame(self):
        fetch_frame = ctk.CTkFrame(self, border_width=2, border_color=Config.COLOR_BORDER, fg_color=self.BG_COLOR)
        fetch_frame.grid(row=1, column=0, sticky="ew", padx=2, pady=2)

        ctk.CTkLabel(fetch_frame, text="Fetch MT5 Data", font=self.TITLE_FONT, text_color=self.TITLE_COLOR, anchor="w").pack(fill="x", padx=2, pady=(2, 2))

        folder_row = ctk.CTkFrame(fetch_frame, fg_color="transparent")
        folder_row.pack(fill="x", padx=2, pady=2)
        ctk.CTkButton(folder_row, text="MT5 Data Folder", font=self.UI_FONT, command=self._select_data_folder).pack(side="left", padx=2)
        self.data_folder_label = ctk.CTkLabel(folder_row, text="", font=self.UI_FONT, anchor="w")
        self.data_folder_label.pack(side="left", padx=2, fill="x", expand=True)

        date_row = ctk.CTkFrame(fetch_frame, fg_color="transparent")
        date_row.pack(fill="x", padx=2, pady=2)
        ctk.CTkLabel(date_row, text="From", font=self.UI_FONT, anchor="w").pack(side="left", padx=(2, 4))
        self.from_year = ctk.CTkEntry(date_row, width=54, font=self.UI_FONT, border_width=2); self.from_year.pack(side="left", padx=1)
        ctk.CTkLabel(date_row, text="-", font=self.UI_FONT).pack(side="left", padx=1)
        self.from_month = ctk.CTkEntry(date_row, width=40, font=self.UI_FONT, border_width=2); self.from_month.pack(side="left", padx=1)
        ctk.CTkLabel(date_row, text="-", font=self.UI_FONT).pack(side="left", padx=1)
        self.from_day = ctk.CTkEntry(date_row, width=40, font=self.UI_FONT, border_width=2); self.from_day.pack(side="left", padx=1)
        ctk.CTkLabel(date_row, text="   ", font=self.UI_FONT).pack(side="left")
        ctk.CTkLabel(date_row, text="To", font=self.UI_FONT, anchor="w").pack(side="left", padx=(4, 4))
        self.to_year = ctk.CTkEntry(date_row, width=54, font=self.UI_FONT, border_width=2); self.to_year.pack(side="left", padx=1)
        ctk.CTkLabel(date_row, text="-", font=self.UI_FONT).pack(side="left", padx=1)
        self.to_month = ctk.CTkEntry(date_row, width=40, font=self.UI_FONT, border_width=2); self.to_month.pack(side="left", padx=1)
        ctk.CTkLabel(date_row, text="-", font=self.UI_FONT).pack(side="left", padx=1)
        self.to_day = ctk.CTkEntry(date_row, width=40, font=self.UI_FONT, border_width=2); self.to_day.pack(side="left", padx=1)
        self.available_bars_btn = ctk.CTkButton(date_row, text="Available Bars", width=110, command=self._show_available_bars, fg_color=self.COLOR_SINGLE_BTN, text_color=self.COLOR_SINGLE_BTN_TEXT, font=self.BUTTON_FONT, state="disabled")
        self.available_bars_btn.pack(side="left", padx=(8, 2))

        action_row = ctk.CTkFrame(fetch_frame, fg_color="transparent")
        action_row.pack(fill="x", padx=2, pady=2)
        self.fetch_all_btn = ctk.CTkButton(action_row, text="All", width=50, command=self._fetch_all_symbols, fg_color=self.COLOR_ALL_BTN, text_color=self.COLOR_ALL_BTN_TEXT, font=self.BUTTON_FONT, state="disabled")
        self.fetch_all_btn.pack(side="left", padx=2)
        self.fetch_single_btn = ctk.CTkButton(action_row, text="Single", width=70, command=self._fetch_single_symbol, fg_color=self.COLOR_SINGLE_BTN, text_color=self.COLOR_SINGLE_BTN_TEXT, font=self.BUTTON_FONT, state="disabled")
        self.fetch_single_btn.pack(side="left", padx=2)
        ctk.CTkLabel(action_row, text="Symbol:", font=self.UI_FONT).pack(side="left", padx=2)
        self.symbol_combo = ctk.CTkComboBox(action_row, width=100, state="readonly", border_width=2, font=self.UI_FONT, dropdown_font=self.UI_FONT, command=lambda _: None)
        self.symbol_combo.pack(side="left", fill="x", expand=True, padx=2)
        self.symbol_combo.set("Select Symbol")
        self.refresh_symbols_btn = ctk.CTkButton(action_row, text="↻", width=32, command=self._refresh_symbols, font=self.UI_FONT, state="disabled")
        self.refresh_symbols_btn.pack(side="left", padx=2)

        tf_row = ctk.CTkFrame(fetch_frame, fg_color="transparent")
        tf_row.pack(fill="x", padx=2, pady=2)
        ctk.CTkLabel(tf_row, text="TF:", width=30, font=self.UI_FONT).pack(side="left", padx=2)
        self.timeframe_vars = {}
        for tf in Mt5FetchData.TIMEFRAME_ORDER:
            var = ctk.BooleanVar(value=True)
            self.timeframe_vars[tf] = var
            ctk.CTkCheckBox(tf_row, text=tf, width=60, variable=var, font=self.UI_FONT, border_width=2).pack(side="left", padx=2, fill="x", expand=True)

    # ------------------------------------------------------------------
    # Sync GUI vars to pure-python objects after construction
    # ------------------------------------------------------------------

    def _sync_gui_to_backend(self):
        self.server_ip_var.set(self.mcp_client.server_ip)
        self.port_var.set(self.mcp_client.port)
        self.endpoint_var.set(self.mcp_client.endpoint)
        self.token_var.set(self.mcp_client.token)
        self.auto_connect_var.set(self.mcp_client.auto_connect)
        self.market_watch_only_var.set(self.mcp_client.market_watch_only)
        self.blacklist_enabled_var.set(self.mcp_client.blacklist_enabled)
        self.blacklist_var.set(self.mcp_client.blacklist_raw)
        self.data_folder_label.configure(text=self.fetch_data.data_folder_path)
        fy = self.fetch_data.date_from
        ty = self.fetch_data.date_to - timedelta(days=1)
        self.from_year.delete(0, "end"); self.from_year.insert(0, f"{fy.year:04d}")
        self.from_month.delete(0, "end"); self.from_month.insert(0, f"{fy.month:02d}")
        self.from_day.delete(0, "end"); self.from_day.insert(0, f"{fy.day:02d}")
        self.to_year.delete(0, "end"); self.to_year.insert(0, f"{ty.year:04d}")
        self.to_month.delete(0, "end"); self.to_month.insert(0, f"{ty.month:02d}")
        self.to_day.delete(0, "end"); self.to_day.insert(0, f"{ty.day:02d}")
        for tf in Mt5FetchData.TIMEFRAME_ORDER:
            self.timeframe_vars[tf].set(tf in self.fetch_data.selected_timeframes)

    # ------------------------------------------------------------------
    # Auto-connect timer
    # ------------------------------------------------------------------

    def _start_auto_connect(self):
        self._sync_gui_to_backend()
        self._auto_connect_job = self.after(Mt5McpClient.AUTO_CONNECT_DELAY_MS, self._auto_connect_check)

    def _auto_connect_check(self):
        self._auto_connect_job = None
        if self.auto_connect_var.get() and not self.mcp_client.connected:
            self._manual_connect()
        self._auto_connect_job = self.after(Mt5McpClient.AUTO_CONNECT_INTERVAL_MS, self._auto_connect_check)

    # ------------------------------------------------------------------
    # Connection control
    # ------------------------------------------------------------------

    def _push_gui_to_client(self):
        self.mcp_client.configure_connection(
            self.server_ip_var.get().strip(),
            self.port_var.get().strip(),
            self.endpoint_var.get().strip(),
            self.token_var.get().strip())
        self.mcp_client.auto_connect = self.auto_connect_var.get()
        self.mcp_client.market_watch_only = self.market_watch_only_var.get()
        self.mcp_client.blacklist_enabled = self.blacklist_enabled_var.get()
        self.mcp_client.blacklist_raw = self.blacklist_var.get()

    def _manual_connect(self):
        if self.mcp_client.connected:
            self.mcp_client.mcp_disconnect()
            self._update_connection_status(False)
            self._log_append("MCP disconnected")
            return
        self._push_gui_to_client()
        self.connect_btn.configure(state="disabled")
        def worker():
            success = self.mcp_client.mcp_connect()
            self.after(0, lambda: self._finish_connect(success))
        threading.Thread(target=worker, daemon=True).start()

    def _finish_connect(self, success):
        self.connect_btn.configure(state="normal")
        self._update_connection_status(success)

    def _update_connection_status(self, connected):
        if connected:
            self.conn_status_dot.configure(text_color=self.COLOR_STATUS_OK)
            self.connect_btn.configure(text="Disconnect")
            self.refresh_symbols_btn.configure(state="normal")
            self.fetch_all_btn.configure(state="normal")
            self.fetch_single_btn.configure(state="normal")
            self.available_bars_btn.configure(state="normal")
            self.after(300, self._after_connected)
        else:
            self.conn_status_dot.configure(text_color=self.COLOR_STATUS_BAD)
            self.connect_btn.configure(text="Connect")
            self.refresh_symbols_btn.configure(state="disabled")
            self.fetch_all_btn.configure(state="disabled")
            self.fetch_single_btn.configure(state="disabled")
            self.available_bars_btn.configure(state="disabled")

    def _on_mcp_connected(self):
        self._update_connection_status(True)

    def _after_connected(self):
        def worker():
            symbols = self.mcp_client.get_market_watch_symbols()
            symbols = self.mcp_client.filter_symbols(symbols)
            if symbols:
                self._log_append(f"✅ Loaded {len(symbols)} symbols")
                self.after(0, lambda: self._update_symbols_dropdown(symbols))
            else:
                self._log_append("⚠️ No symbols available after blacklist filter")
        threading.Thread(target=worker, daemon=True).start()

    # ------------------------------------------------------------------
    # Token helpers
    # ------------------------------------------------------------------

    def _toggle_token(self):
        self.token_visible = not self.token_visible
        if self.token_visible:
            self.token_entry.configure(show="")
            self.token_toggle.configure(text="🙈")
        else:
            self.token_entry.configure(show="*")
            self.token_toggle.configure(text="👁")

    def _clean_token(self):
        self.token_var.set("")
        self.token_entry.focus_set()
        self._log_append("🧹 Token entry cleared (saved token untouched)")

    def _save_token(self):
        self._push_gui_to_client()
        self.mcp_client.save_state()
        self._log_append("🔑 Token saved (obfuscated)")

    def _clear_saved_token(self):
        self.token_var.set("")
        self.mcp_client.token = ""
        self.mcp_client.save_state()
        self._log_append("🗑️ Saved token cleared")

    # ------------------------------------------------------------------
    # Options handlers
    # ------------------------------------------------------------------

    def _on_auto_connect_change(self):
        self.mcp_client.auto_connect = self.auto_connect_var.get()
        self.mcp_client.save_state()
        self._log_append(f"Auto Connect {'enabled' if self.auto_connect_var.get() else 'disabled'}")

    def _on_options_change(self):
        self.mcp_client.market_watch_only = self.market_watch_only_var.get()
        self.mcp_client.blacklist_enabled = self.blacklist_enabled_var.get()
        self.mcp_client.save_state()
        mw = "ON" if self.market_watch_only_var.get() else "OFF"
        bl = "ON" if self.blacklist_enabled_var.get() else "OFF"
        self._log_append(f"Market Watch Only: {mw}  |  Blacklist: {bl}")

    def _on_blacklist_edit(self):
        self.mcp_client.blacklist_raw = self.blacklist_var.get()
        self.mcp_client.save_state()

    # ------------------------------------------------------------------
    # Tool views (write to bottom log)
    # ------------------------------------------------------------------

    def _list_mcp_tools(self):
        if not self.mcp_client.connected:
            self._log_append("❌ Not connected.")
            return
        self._log_append("Fetching server capabilities...")
        tools = self.mcp_client.mcp_list_tools()
        resources = self.mcp_client.mcp_list_resources()
        prompts = self.mcp_client.mcp_list_prompts()
        self._log_append("\n--- Server Capabilities ---")
        self._log_append(f"\nTools ({len(tools)}):")
        for t in tools:
            self._log_append(f"  • {t.get('name', 'unknown')}: {t.get('description', '')}")
        self._log_append(f"\nResources ({len(resources)}):")
        for r in resources:
            self._log_append(f"  • {r.get('name', r.get('uri', 'unknown'))} ({r.get('uri', '')})")
        self._log_append(f"\nPrompts ({len(prompts)}):")
        for p in prompts:
            self._log_append(f"  • {p.get('name', 'unknown')}: {p.get('description', '')}")
        self._log_append("=" * 50)

    def _show_tool_schemas(self):
        if not self.mcp_client.connected:
            self._log_append("❌ Not connected.")
            return
        tools = self.mcp_client.mcp_list_tools()
        if not tools:
            self._log_append("❌ No tools available — connect first.")
            return
        self._log_append(f"Fetching schemas for {len(tools)} tools...")
        self._log_append("\n--- Tool Schemas ---")
        self._log_append("=" * 50)
        for t in tools:
            name = t.get("name", "unknown")
            desc = t.get("description", "")
            schema = t.get("inputSchema", {}) or {}
            self._log_append(f"\n● {name}")
            if desc:
                self._log_append(f"  {desc}")
            pretty = json.dumps(schema, indent=2, ensure_ascii=False)
            for line in pretty.splitlines():
                self._log_append(f"    {line}")
        self._log_append("\n" + "=" * 50)

    def _show_marketwatch_symbols(self):
        if not self.mcp_client.connected:
            self._log_append("❌ Not connected.")
            return
        mw_only = self.market_watch_only_var.get()
        bl = self.blacklist_enabled_var.get()
        label = "Market Watch" if mw_only else "All Symbols"
        if bl:
            label += " (blacklist)"
        self._log_append(f"Fetching {label}...")
        symbols = self.mcp_client.get_market_watch_symbols()
        self._log_append(f"\n--- {label} ({len(symbols)}) ---")
        if symbols:
            self._log_append(", ".join(symbols))
        self._log_append("=" * 50)

    # ------------------------------------------------------------------
    # Fetch actions
    # ------------------------------------------------------------------

    def _select_data_folder(self):
        folder = filedialog.askdirectory(initialdir=self.fetch_data.data_folder_path, parent=self)
        if folder:
            self.fetch_data.data_folder_path = folder
            self.data_folder_label.configure(text=folder)
            self.fetch_data.save_config()

    def _get_selected_timeframes(self):
        return [tf for tf in Mt5FetchData.TIMEFRAME_ORDER if self.timeframe_vars[tf].get()]

    def _sync_fetch_gui_to_backend(self):
        self.fetch_data.data_folder_path = self.data_folder_label.cget("text") or self.fetch_data.data_folder_path
        try:
            fy = int(self.from_year.get().strip()); fm = int(self.from_month.get().strip()); fd = int(self.from_day.get().strip())
            ty = int(self.to_year.get().strip()); tm = int(self.to_month.get().strip()); td = int(self.to_day.get().strip())
            self.fetch_data.date_from = datetime(fy, fm, fd, tzinfo=timezone.utc)
            self.fetch_data.date_to = datetime(ty, tm, td, tzinfo=timezone.utc) + timedelta(days=1) - timedelta(seconds=1)
        except ValueError:
            pass
        self.fetch_data.selected_timeframes = self._get_selected_timeframes()
        self.fetch_data.save_config()

    def _fetch_single_symbol(self):
        if not self.mcp_client.connected:
            raise RuntimeError("Fetch: single not connected")
        symbol = self.symbol_combo.get()
        if not symbol or symbol == "Select Symbol":
            raise RuntimeError("Fetch: no symbol selected")
        self._sync_fetch_gui_to_backend()
        tfs = self._get_selected_timeframes()
        if not tfs:
            raise RuntimeError("Fetch: no timeframe selected")
        self._log_append(f"🚀 Fetching {symbol} {self.fetch_data.date_from:%Y-%m-%d} → {(self.fetch_data.date_to - timedelta(days=1)):%Y-%m-%d}")
        self.fetch_all_btn.configure(state="disabled")
        self.fetch_single_btn.configure(state="disabled")
        self.available_bars_btn.configure(state="disabled")
        def thread():
            self.fetch_data.fetch_single(symbol)
            self.after(0, lambda: self.fetch_all_btn.configure(state="normal"))
            self.after(0, lambda: self.fetch_single_btn.configure(state="normal"))
            self.after(0, lambda: self.available_bars_btn.configure(state="normal"))
        threading.Thread(target=thread, daemon=True).start()

    def _fetch_all_symbols(self):
        if not self.mcp_client.connected:
            raise RuntimeError("Fetch: all not connected")
        self._sync_fetch_gui_to_backend()
        tfs = self._get_selected_timeframes()
        if not tfs:
            raise RuntimeError("Fetch: no timeframe selected")
        symbols = self.fetch_data.fetch_symbols()
        symbols = self.mcp_client.filter_symbols(symbols)
        if not symbols:
            raise RuntimeError("Fetch: no Market Watch symbols available")
        self._log_append(f"🚀 Fetching {len(symbols)} symbols {self.fetch_data.date_from:%Y-%m-%d} → {(self.fetch_data.date_to - timedelta(days=1)):%Y-%m-%d}")
        self.fetch_all_btn.configure(state="disabled")
        self.fetch_single_btn.configure(state="disabled")
        self.available_bars_btn.configure(state="disabled")
        def thread():
            self.fetch_data.fetch_all()
            self.after(0, lambda: self.fetch_all_btn.configure(state="normal"))
            self.after(0, lambda: self.fetch_single_btn.configure(state="normal"))
            self.after(0, lambda: self.available_bars_btn.configure(state="normal"))
        threading.Thread(target=thread, daemon=True).start()

    def _show_available_bars(self):
        if not self.mcp_client.connected:
            raise RuntimeError("Available Bars: not connected")
        self._sync_fetch_gui_to_backend()
        self.available_bars_btn.configure(state="disabled")
        def thread():
            self.fetch_data.scan_available_bars()
            self.after(0, lambda: self.available_bars_btn.configure(state="normal"))
        threading.Thread(target=thread, daemon=True).start()

    def _refresh_symbols(self):
        if not self.mcp_client.connected:
            raise RuntimeError("Fetch: refresh symbols not connected")
        def thread():
            self._log_append("📋 Fetching symbols...")
            symbols = self.fetch_data.fetch_symbols()
            symbols = self.mcp_client.filter_symbols(symbols)
            self.after(0, lambda: self._update_symbols_dropdown(symbols))
            self._log_append(f"✅ Got {len(symbols)} symbols")
        threading.Thread(target=thread, daemon=True).start()

    def _update_symbols_dropdown(self, symbols):
        filtered = sorted(set(symbols))
        self.symbol_combo.configure(values=filtered)
        if filtered:
            current = self.symbol_combo.get()
            self.symbol_combo.set(current if current in filtered else filtered[0])

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def report_callback_exception(self, exc_type, exc_value, exc_tb):
        self._log_append(f"❌ {exc_type.__name__}: {exc_value}")

    def _on_close(self):
        confirmed = messagebox.askyesno(
            "Exit MT5 MCP Trading Client",
            "Are you sure you want to quit?\n\nAny running operations will be stopped.",
            icon="question", parent=self)
        if not confirmed:
            return
        if self._auto_connect_job:
            self.after_cancel(self._auto_connect_job)
        self.mcp_client.stop()
        self.mcp_client.save_state()
        self.fetch_data.save_config()
        Config.save_window_geometry(self.ini_path, self.wm_geometry())
        self.destroy()


if __name__ == "__main__":
    app = Mt5FetchMcpGUI()
    app.mainloop()