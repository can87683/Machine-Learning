#!/usr/bin/env python3
# mt5_fetch_api.py
# Copyright Su Nie | BSD-3C License | https://github.com/can87683

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
from tkinter import filedialog, messagebox
import customtkinter as ctk
import requests
import tkinter as tk

import MetaTrader5 as mt5


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
    WINDOW_HEIGHT = 1020
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


class Mt5ApiClient:
    """Pure Python MT5 API client; no GUI dependencies."""

    DEFAULT_PATH = ""
    AUTO_CONNECT_DEFAULT = True
    AUTO_CONNECT_INTERVAL_MS = 5000

    TF_MAP = {
        "M1": 1, "M2": 2, "M3": 3, "M4": 4, "M5": 5,
        "M6": 6, "M10": 10, "M12": 12, "M15": 15,
        "M20": 20, "M30": 30,
        "H1": 16385, "H2": 16386, "H3": 16387, "H4": 16388,
        "H6": 16390, "H8": 16392, "H12": 16396,
        "D1": 16408, "W1": 32769, "MN1": 49153,
    }

    DEFAULT_BLACKLIST = "AUDUSD,GBPUSD,NZDUSD,USDCHF,USDTRY"

    def __init__(self, config_path=None, log_callback=None, schedule_callback=None):
        self.ini_path = config_path or Config.INI_FILE
        self.log_callback = log_callback
        self.schedule = schedule_callback or (lambda fn: fn())

        self.connected = False
        self.on_connect_success = None
        self._closing = False

        self.ini = configparser.ConfigParser(interpolation=None)
        if os.path.exists(self.ini_path):
            self.ini.read(self.ini_path, encoding="utf-8")

        self._load_state()

    def _load_state(self):
        if not self.ini.has_section("Mt5"):
            self.ini.add_section("Mt5")

        self.path_value = self.ini.get("Mt5", "path", fallback=self.DEFAULT_PATH)
        self.auto_connect_value = self.ini.getboolean(
            "Mt5", "auto_connect", fallback=self.AUTO_CONNECT_DEFAULT,
        )
        self.market_watch_only_value = self.ini.getboolean(
            "Mt5", "market_watch_only", fallback=True,
        )
        self.blacklist_enabled_value = self.ini.getboolean(
            "Mt5", "blacklist_enabled", fallback=True,
        )
        self.blacklist_value = self.ini.get(
            "Mt5", "blacklist", fallback=self.DEFAULT_BLACKLIST,
        )

    def log(self, message):
        if self._closing:
            return
        if self.log_callback:
            self.log_callback(message)

    def connect(self, path_override=None):
        path = path_override or self.path_value.strip() or None
        init_kwargs = {}
        if path:
            init_kwargs["path"] = path

        if not mt5.initialize(**init_kwargs):
            self.log(f"❌ initialize() failed: {mt5.last_error()}")
            return False

        self.connected = True
        info = mt5.account_info()
        if info:
            self.log(
                f"✅ Connected: {info.login} @ {info.server} | "
                f"Balance: {info.balance:.2f} {info.currency}"
            )
        else:
            self.log("✅ Connected (no account info available)")

        self.save_state()
        if callable(self.on_connect_success):
            self.schedule(self.on_connect_success)
        return True

    def disconnect(self):
        mt5.shutdown()
        self.connected = False
        self.log("🔌 Disconnected from MT5")

    def fetch_symbols(self):
        all_symbols = mt5.symbols_get() or []
        mw_only = self.market_watch_only_value
        records = []
        for sym in all_symbols:
            if mw_only and not sym.visible:
                continue
            records.append({"symbol": sym.name})
        symbols = {r["symbol"] for r in records if r["symbol"]}
        blacklist = self._parse_blacklist()
        if blacklist:
            symbols -= blacklist
        return sorted(symbols)

    def filter_symbols(self, symbols):
        if not symbols:
            return []
        blacklist = self._parse_blacklist()
        return sorted({s for s in symbols if s and s not in blacklist})

    def _parse_blacklist(self):
        if not self.blacklist_enabled_value:
            return set()
        raw = self.blacklist_value or ""
        return {s.strip() for s in raw.split(",") if s.strip()}

    def get_rates(self, symbol, timeframe, count):
        count = min(int(count), Config.TF_MAX_BARS.get(timeframe, int(count)))
        tf_const = self.TF_MAP.get(timeframe)
        if tf_const is None:
            raise RuntimeError(f"Unknown timeframe: {timeframe}")

        dt_to = datetime.now(timezone.utc)
        tf_min = Config.TF_MINUTES.get(timeframe, 5)
        span_minutes = tf_min * (count + 5)
        max_span = 5 * 365 * 24 * 60
        if span_minutes <= max_span:
            dt_from = dt_to - timedelta(minutes=span_minutes)
        else:
            dt_from = dt_to - timedelta(days=365 * 5)

        rates = mt5.copy_rates_range(symbol, tf_const, dt_from, dt_to)
        if rates is None:
            self.log(
                f"⚠️ copy_rates_range returned None for {symbol} "
                f"{timeframe}: {mt5.last_error()}"
            )
            return []

        bars = []
        for r in rates:
            bars.append({
                "time": int(r["time"]),
                "open": float(r["open"]),
                "high": float(r["high"]),
                "low": float(r["low"]),
                "close": float(r["close"]),
                "tick_volume": int(r["tick_volume"]),
                "spread": int(r["spread"]),
                "real_volume": int(r["real_volume"]),
            })
        return bars[-count:]

    def count_available_bars(self, symbol, timeframe, probe=500):
        bars = self.get_rates(symbol, timeframe, probe)
        return len(bars) if bars else 0

    def get_symbol_volume_info(self, symbol):
        info = mt5.symbol_info(symbol)
        if info is None:
            return (0.01, 100.0, 0.01)
        return (info.volume_min, info.volume_max, info.volume_step)

    def save_state(self):
        if self.ini.has_section("Mt5"):
            for key in ("login", "password", "server"):
                if self.ini.has_option("Mt5", key):
                    self.ini.remove_option("Mt5", key)
        Config.atomic_merge_ini(self.ini_path, "Mt5", {
            "path": self.path_value,
            "auto_connect": str(self.auto_connect_value),
            "market_watch_only": str(self.market_watch_only_value),
            "blacklist_enabled": str(self.blacklist_enabled_value),
            "blacklist": self.blacklist_value,
        })

    def stop(self):
        self._closing = True
        if self.connected:
            mt5.shutdown()
            self.connected = False


class Mt5ApiFetchData:
    """Pure Python historical bar fetcher; no GUI dependencies."""

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
        "M1": 500_000,
        "M5": 200_000,
        "M15": 100_000,
        "H1": 50_000,
        "H4": 30_000,
        "D1": 5_000,
        "W1": 1_000,
        "MN1": 240,
    }

    RANGE_RETRY_STEPS = 4
    AVAILABILITY_PROBE_BARS = 500

    TABLE_COL_WIDTH = 8

    CSV_COLUMNS = ("time", "open", "high", "low", "close", "tick_volume")

    TOOL_CANDIDATES = {
        "symbols": [
            "get_marketwatch_symbols",
            "market_watch_get_symbols",
            "get_symbols",
        ],
        "rates": [
            "get_chart_history",
            "market_data_get_candles",
            "get_candles",
            "get_bars",
            "get_rates",
        ],
    }

    def __init__(self, mcp_client, config_path=None, log_callback=None, schedule_callback=None):
        self.mcp_client = mcp_client
        self.log_callback = log_callback
        self.schedule = schedule_callback or (lambda fn: fn())

        if config_path is None:
            script_name = os.path.splitext(os.path.basename(__file__))[0]
            config_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                f"{script_name}.ini",
            )
        self.ini_path = config_path
        self.ini = configparser.ConfigParser(interpolation=None)
        if os.path.exists(self.ini_path):
            self.ini.read(self.ini_path, encoding="utf-8")

        self.data_folder_path = self.DEFAULT_DATA_FOLDER
        self.date_from_value = None
        self.date_to_value = None
        self.selected_timeframes = list(self.TIMEFRAME_ORDER)

        self._load_fetch_state()

    def log(self, message):
        if self.log_callback:
            self.log_callback(message)

    def is_connected(self):
        return bool(
            self.mcp_client
            and getattr(self.mcp_client, "connected", False)
        )

    def set_client(self, mcp_client):
        self.mcp_client = mcp_client

    def get_selected_timeframes(self):
        return [tf for tf in self.TIMEFRAME_ORDER if tf in self.selected_timeframes]

    def _load_fetch_state(self):
        today = datetime.now(timezone.utc).date()
        from_date = today - timedelta(days=self.DEFAULT_FROM_OFFSET_DAYS)

        saved_from = f"{from_date.year:04d}-{from_date.month:02d}-{from_date.day:02d}"
        saved_to = f"{today.year:04d}-{today.month:02d}-{today.day:02d}"

        if self.ini.has_section("Fetch"):
            self.data_folder_path = self.ini.get(
                "Fetch", "data_folder", fallback=self.DEFAULT_DATA_FOLDER,
            )
            saved_from = self.ini.get(
                "Fetch", "date_from", fallback=saved_from,
            )
            saved_to = self.ini.get(
                "Fetch", "date_to", fallback=saved_to,
            )
            saved_tfs = [
                tf.strip() for tf in self.ini.get(
                    "Fetch", "fetch_timeframes",
                    fallback=",".join(self.TIMEFRAME_ORDER),
                ).split(",")
                if tf.strip()
            ]
            self.selected_timeframes = [tf for tf in self.TIMEFRAME_ORDER if tf in saved_tfs]

        self.date_from_value = saved_from
        self.date_to_value = saved_to

    def save_config(self):
        date_from = self.date_from_value
        date_to = self.date_to_value
        Config.atomic_merge_ini(self.ini_path, "Fetch", {
            "date_from":        date_from,
            "date_to":          date_to,
            "fetch_timeframes": ",".join(self.get_selected_timeframes()),
            "data_folder":      self.data_folder_path,
        })

    @staticmethod
    def _extract_bars(payload):
        if isinstance(payload, list):
            return [b for b in payload if isinstance(b, dict)]
        if not isinstance(payload, dict):
            return []
        for key in (
            "data", "bars", "items", "candles", "history",
            "rates", "ohlc", "result",
        ):
            value = payload.get(key)
            if isinstance(value, list):
                return [b for b in value if isinstance(b, dict)]
        return []

    def resolve_tool(self, category):
        candidates = self.TOOL_CANDIDATES.get(category, [])
        for cand in candidates:
            if hasattr(self.mcp_client, cand):
                return cand
        raise RuntimeError(f"No tool resolved for category: {category}")

    def mcp_call_tool(self, name, args):
        method = getattr(self.mcp_client, name, None)
        if not callable(method):
            raise RuntimeError(f"MCP tool not found: {name}")
        return method(args)

    def _unwrap_tool_text(self, envelope, name):
        if isinstance(envelope, list):
            return envelope
        if isinstance(envelope, dict):
            content = envelope.get("content") or []
            if content and isinstance(content[0], dict):
                text = content[0].get("text", "")
                try:
                    return json.loads(text)
                except (json.JSONDecodeError, TypeError):
                    return text
            return envelope
        return envelope

    def fetch_symbols(self):
        return self.mcp_client.fetch_symbols()

    def _filter_symbols(self, symbols):
        if not symbols:
            return []
        filter_fn = getattr(self.mcp_client, "filter_symbols", None)
        if callable(filter_fn):
            return filter_fn(symbols)
        return sorted(set(symbols))

    def _resolve_datetime_range(self):
        fy, fm, fd = self.date_from_value.split("-")
        ty, tm, td = self.date_to_value.split("-")
        dt_from = datetime(int(fy), int(fm), int(fd), tzinfo=timezone.utc)
        dt_to = datetime(int(ty), int(tm), int(td), tzinfo=timezone.utc) + timedelta(days=1) - timedelta(seconds=1)

        if dt_from > dt_to:
            raise RuntimeError(
                "Fetch: 'From' date must be before or equal to 'To' date"
            )
        return dt_from, dt_to

    def _clamp_count(self, timeframe, count):
        cap = self.TF_MAX_BARS.get(timeframe)
        return min(int(count), cap) if cap else int(count)

    def _estimate_bars(self, timeframe, dt_from, dt_to):
        tf_min = self.TF_MINUTES.get(timeframe, 5)
        total_minutes = max(1, int((dt_to - dt_from).total_seconds() // 60))
        return self._clamp_count(timeframe, total_minutes // tf_min + 5)

    def _build_rates_request(self, symbol, timeframe, dt_from, dt_to):
        return {
            "symbol": symbol,
            "period": timeframe,
            "datetime_from": dt_from.strftime("%Y-%m-%dT%H:%M:%S"),
            "datetime_to":   dt_to.strftime("%Y-%m-%dT%H:%M:%S"),
            "limit":         self._estimate_bars(timeframe, dt_from, dt_to),
        }

    def _is_range_error(self, envelope):
        content = envelope.get("content") or []
        text = content[0].get("text", "") if content else ""
        return "invalid datetime range" in text.lower()

    def fetch_rates(self, symbol, timeframe, dt_from=None, dt_to=None):
        if dt_from is None or dt_to is None:
            dt_from, dt_to = self._resolve_datetime_range()
        tf_min = self.TF_MINUTES.get(timeframe, 5)
        total_minutes = max(1, int((dt_to - dt_from).total_seconds() // 60))
        count = min(total_minutes // tf_min + 5, self.TF_MAX_BARS.get(timeframe, 10_000))
        return self.mcp_client.get_rates(symbol, timeframe, count)

    def count_available_bars(self, symbol, timeframe):
        dt_from, dt_to = self._resolve_datetime_range()
        return len(self.fetch_rates(symbol, timeframe, dt_from, dt_to))

    def select_data_folder(self, folder):
        self.data_folder_path = folder
        self.save_config()

    def wipe_data_folder(self):
        if os.path.isdir(self.data_folder_path):
            shutil.rmtree(self.data_folder_path)
            self.log(f"🧹 Removed {self.data_folder_path}")
        else:
            self.log(f"🧹 No existing data folder at {self.data_folder_path}")

    def scan_available_bars(self, symbols, selected_tfs, dt_from, dt_to):
        header = "TF:" + "".join(
            "".join([" " * ((self.TABLE_COL_WIDTH - len(tf)) // 2), tf])
            for tf in self.TIMEFRAME_ORDER
        )
        self.log(header)
        for idx, symbol in enumerate(symbols, 1):
            counts = {}
            for tf in selected_tfs:
                data = self.fetch_rates(symbol, tf, dt_from, dt_to)
                counts[tf] = len(data)
                time.sleep(0.05)
            row = symbol.ljust(12)
            for tf in self.TIMEFRAME_ORDER:
                n = counts.get(tf, 0)
                text = self._fmt_bars(n) if n > 0 else "N/A"
                row += " " * (self.TABLE_COL_WIDTH - len(text)) + text
            self.log(f"{idx}/{len(symbols)} {row}")
            time.sleep(0.1)
        self.log(f"✅ Available bars scan complete: {len(symbols)} symbols")

    def fetch_single_symbol(self, symbol, selected_tfs, dt_from, dt_to):
        target_dir = self.data_folder_path
        self.wipe_data_folder()
        for tf in selected_tfs:
            data = self.fetch_rates(symbol, tf, dt_from, dt_to)
            if data:
                self._save_to_csv(symbol, tf, data, target_dir)
            time.sleep(0.1)

    def fetch_all_symbols(self, symbols, selected_tfs, dt_from, dt_to):
        target_dir = self.data_folder_path
        self.wipe_data_folder()
        for idx, symbol in enumerate(symbols, 1):
            self.log(f"📈 {idx}/{len(symbols)}: {symbol}")
            for tf in selected_tfs:
                data = self.fetch_rates(symbol, tf, dt_from, dt_to)
                if data:
                    self._save_to_csv(symbol, tf, data, target_dir)
                time.sleep(0.1)
            time.sleep(0.2)
        self.log(f"✅ Fetch complete: {len(symbols)} symbols")

    @staticmethod
    def _fmt_bars(n):
        if n >= 1_000_000:
            return f"{n / 1_000_000:.1f}M"
        if n >= 1_000:
            return f"{n / 1_000:.1f}K"
        return str(n)

    def _row_for(self, values, label):
        cw = self.TABLE_COL_WIDTH
        row = label
        for tf in self.TIMEFRAME_ORDER:
            n = values.get(tf, 0)
            if n > 0:
                text = self._fmt_bars(n)
                row += " " * (cw - len(text)) + text
            else:
                na = "N/A"
                left = (cw - len(na)) // 2
                right = (cw - len(na)) - left
                row += " " * left + na + " " * right
        return row

    def _update_available_display(self, available):
        cw = self.TABLE_COL_WIDTH
        header = "TF:" + "".join(
            " " * ((cw - len(tf)) // 2) + tf
            + " " * ((cw - len(tf)) - (cw - len(tf)) // 2)
            for tf in self.TIMEFRAME_ORDER
        )
        return header, self._row_for(available, "Avail:")

    def _update_fetched_display(self, fetched):
        return self._row_for(fetched, "Fetched:")

    def _save_to_csv(self, symbol, timeframe, data, target_dir):
        safe_symbol = re.sub(r"[^A-Za-z0-9_+#!$-]", "_", symbol)
        symbol_dir = os.path.join(target_dir, safe_symbol)
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
            writer = csv.DictWriter(
                handle, fieldnames=list(self.CSV_COLUMNS),
                lineterminator=os.linesep,
            )
            writer.writeheader()
            writer.writerows(rows)
        self.log(f"💾 Saved to {filepath}")


class Mt5ApiFetchGUI(ctk.CTk):
    """Top-level application window; owns frames, wiring, and lifecycle."""

    TITLE = "MT5 Fetch API - Copyright Su Nie | BSD-3C License | https://github.com/can87683"
    APPEARANCE = "dark"
    THEME = "blue"
    BG_COLOR = "#1c1c1c"

    TITLE_FONT = ("TkDefaultFont", 14, "bold")
    TITLE_COLOR = "yellow"
    UI_FONT = ("TkDefaultFont", 12)
    MONO_FONT = ("Consolas", 10)
    MONO_BOLD = ("Courier", 12, "bold")
    BUTTON_FONT = ("TkDefaultFont", 12, "bold")

    COLOR_BORDER = Config.COLOR_BORDER
    BORDER_WIDTH = Config.BORDER_WIDTH

    COLOR_CONNECT = "#2196F3"
    COLOR_DISCONNECT = "#F44336"
    COLOR_SYMBOLS = "#FF9800"
    COLOR_CLEAR_LOG = "#546E7A"
    COLOR_STATUS_OK = "#4CAF50"
    COLOR_STATUS_BAD = "#F44336"

    COLOR_ALL_BTN = "#4285F4"
    COLOR_ALL_BTN_TEXT = "yellow"
    COLOR_SINGLE_BTN = "#E67E22"
    COLOR_SINGLE_BTN_TEXT = "white"
    COLOR_AVAIL_TEXT = "#00FFFF"
    COLOR_FETCHED_TEXT = "#00FF00"

    LOG_BG_COLOR = "black"
    LOG_TITLE = "Log"
    CLEAR_SENTINEL = "__clear__"

    def __init__(self):
        super().__init__()

        ctk.set_appearance_mode(self.APPEARANCE)
        ctk.set_default_color_theme(self.THEME)

        self.title(self.TITLE)
        self.resizable(False, False)
        self.configure(fg_color=self.BG_COLOR)

        self.ini_path = Config.INI_FILE
        saved_x, saved_y = Config.load_window_geometry(self.ini_path)
        self._apply_geometry(saved_x, saved_y)

        # Business logic instances
        self.api_client = Mt5ApiClient(
            config_path=self.ini_path,
            log_callback=self._log_append,
            schedule_callback=lambda fn: self.after(0, fn),
        )
        self.fetch_data = Mt5ApiFetchData(
            mcp_client=self.api_client,
            config_path=self.ini_path,
            log_callback=self._log_append,
            schedule_callback=lambda fn: self.after(0, fn),
        )

        # State vars synced between GUI and business logic
        self.path_var = ctk.StringVar(value=self.api_client.path_value)
        self.auto_connect_var = ctk.BooleanVar(value=self.api_client.auto_connect_value)
        self.market_watch_only_var = ctk.BooleanVar(value=self.api_client.market_watch_only_value)
        self.blacklist_enabled_var = ctk.BooleanVar(value=self.api_client.blacklist_enabled_value)
        self.blacklist_var = ctk.StringVar(value=self.api_client.blacklist_value)

        today = datetime.now(timezone.utc).date()
        from_date = today - timedelta(days=Mt5ApiFetchData.DEFAULT_FROM_OFFSET_DAYS)
        self.from_year_var = ctk.StringVar(value=f"{from_date.year:04d}")
        self.from_month_var = ctk.StringVar(value=f"{from_date.month:02d}")
        self.from_day_var = ctk.StringVar(value=f"{from_date.day:02d}")
        self.to_year_var = ctk.StringVar(value=f"{today.year:04d}")
        self.to_month_var = ctk.StringVar(value=f"{today.month:02d}")
        self.to_day_var = ctk.StringVar(value=f"{today.day:02d}")

        self.timeframe_vars = {}
        for tf in Mt5ApiFetchData.TIMEFRAME_ORDER:
            var = ctk.BooleanVar(value=tf in self.fetch_data.selected_timeframes)
            self.timeframe_vars[tf] = var

        self.data_folder_var = ctk.StringVar(value=self.fetch_data.data_folder_path)
        self.symbol_combo_values = []

        self._closing = False
        self._auto_connect_job = None

        self._build_ui()
        self._sync_gui_to_logic()
        self._wire_callbacks()

        if self.auto_connect_var.get():
            self._start_auto_connect()

        self.protocol("WM_DELETE_WINDOW", self._on_close)

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

    def _build_ui(self):
        self.grid_rowconfigure(0, weight=0)
        self.grid_rowconfigure(1, weight=0)
        self.grid_rowconfigure(2, weight=1)
        self.grid_columnconfigure(0, weight=1)

        # Row 0: MT5 Client controls
        self.client_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.client_frame.grid(row=0, column=0, sticky="ew", padx=2, pady=(2, 2))
        self._build_client_ui()

        # Row 1: Fetch controls
        self.fetch_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.fetch_frame.grid(row=1, column=0, sticky="ew", padx=2, pady=2)
        self._build_fetch_ui()

        # Row 2: Integrated Log
        self.log_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.log_frame.grid(row=2, column=0, sticky="nsew", padx=2, pady=(2, 2))
        self._build_log_ui()

    def _build_client_ui(self):
        ctk.CTkLabel(
            self.client_frame, text="MT5 Fetch API",
            font=self.TITLE_FONT, text_color=self.TITLE_COLOR, anchor="w",
        ).pack(fill="x", padx=2, pady=(2, 2))

        opt_row = ctk.CTkFrame(self.client_frame, fg_color="transparent")
        opt_row.pack(fill="x", padx=2, pady=2)

        ctk.CTkCheckBox(
            opt_row, text="Market Watch Only",
            variable=self.market_watch_only_var,
            font=self.UI_FONT, command=self._on_opt_change,
        ).pack(side="left", padx=(2, 12))

        ctk.CTkCheckBox(
            opt_row, text="Blacklist:",
            variable=self.blacklist_enabled_var,
            font=self.UI_FONT, command=self._on_opt_change,
        ).pack(side="left", padx=2)

        ctk.CTkEntry(
            opt_row, textvariable=self.blacklist_var, font=self.UI_FONT,
            border_width=2, placeholder_text="SYMBOL1,SYMBOL2,SYMBOL3",
        ).pack(side="left", fill="x", expand=True, padx=2)

        btn_row = ctk.CTkFrame(self.client_frame, fg_color="transparent")
        btn_row.pack(fill="x", padx=2, pady=2)

        self.auto_connect_cb = ctk.CTkCheckBox(
            btn_row, text="Auto Connect",
            variable=self.auto_connect_var, font=self.UI_FONT,
            command=self._on_auto_connect_change,
        )
        self.auto_connect_cb.pack(side="left", padx=2)

        self.connect_btn = ctk.CTkButton(
            btn_row, text="Connect", command=self._manual_connect,
            width=100, fg_color=self.COLOR_CONNECT, text_color="white",
            font=self.BUTTON_FONT, border_width=2,
        )
        self.connect_btn.pack(side="left", padx=2)

        ctk.CTkButton(
            btn_row, text="Symbols", command=self._show_marketwatch_symbols,
            width=80, fg_color=self.COLOR_SYMBOLS, text_color="white",
            font=self.BUTTON_FONT, border_width=2,
        ).pack(side="left", padx=2)

        ctk.CTkButton(
            btn_row, text="Clear Log", command=self._clear_log,
            width=80, fg_color=self.COLOR_CLEAR_LOG, text_color="white",
            font=self.BUTTON_FONT, border_width=2,
        ).pack(side="left", padx=2)

        self.conn_status_dot = ctk.CTkLabel(
            btn_row, text="●", text_color=self.COLOR_STATUS_BAD,
            font=self.UI_FONT,
        )
        self.conn_status_dot.pack(side="left", padx=6)

    def _build_fetch_ui(self):
        ctk.CTkLabel(
            self.fetch_frame, text="Fetch MT5 Data",
            font=self.TITLE_FONT, text_color=self.TITLE_COLOR, anchor="w",
        ).pack(fill="x", padx=2, pady=(2, 2))

        folder_row = ctk.CTkFrame(self.fetch_frame, fg_color="transparent")
        folder_row.pack(fill="x", padx=2, pady=2)

        ctk.CTkButton(
            folder_row, text="MT5 Data Folder", font=self.UI_FONT,
            command=self._select_data_folder,
        ).pack(side="left", padx=2)

        self.data_folder_label = ctk.CTkLabel(
            folder_row, textvariable=self.data_folder_var, font=self.UI_FONT, anchor="w",
        )
        self.data_folder_label.pack(side="left", padx=2, fill="x", expand=True)

        date_row = ctk.CTkFrame(self.fetch_frame, fg_color="transparent")
        date_row.pack(fill="x", padx=2, pady=2)

        ctk.CTkLabel(date_row, text="From", font=self.UI_FONT, anchor="w").pack(
            side="left", padx=(2, 4),
        )

        ctk.CTkEntry(
            date_row, textvariable=self.from_year_var, width=54,
            font=self.UI_FONT, border_width=2,
        ).pack(side="left", padx=1)

        ctk.CTkLabel(date_row, text="-", font=self.UI_FONT, anchor="center").pack(
            side="left", padx=1,
        )

        ctk.CTkEntry(
            date_row, textvariable=self.from_month_var, width=40,
            font=self.UI_FONT, border_width=2,
        ).pack(side="left", padx=1)

        ctk.CTkLabel(date_row, text="-", font=self.UI_FONT, anchor="center").pack(
            side="left", padx=1,
        )

        ctk.CTkEntry(
            date_row, textvariable=self.from_day_var, width=40,
            font=self.UI_FONT, border_width=2,
        ).pack(side="left", padx=1)

        ctk.CTkLabel(date_row, text="   ", font=self.UI_FONT).pack(side="left")

        ctk.CTkLabel(date_row, text="To", font=self.UI_FONT, anchor="w").pack(
            side="left", padx=(4, 4),
        )

        ctk.CTkEntry(
            date_row, textvariable=self.to_year_var, width=54,
            font=self.UI_FONT, border_width=2,
        ).pack(side="left", padx=1)

        ctk.CTkLabel(date_row, text="-", font=self.UI_FONT, anchor="center").pack(
            side="left", padx=1,
        )

        ctk.CTkEntry(
            date_row, textvariable=self.to_month_var, width=40,
            font=self.UI_FONT, border_width=2,
        ).pack(side="left", padx=1)

        ctk.CTkLabel(date_row, text="-", font=self.UI_FONT, anchor="center").pack(
            side="left", padx=1,
        )

        ctk.CTkEntry(
            date_row, textvariable=self.to_day_var, width=40,
            font=self.UI_FONT, border_width=2,
        ).pack(side="left", padx=1)

        self.available_bars_btn = ctk.CTkButton(
            date_row, text="Available Bars", width=110,
            command=self._show_available_bars,
            fg_color=self.COLOR_SINGLE_BTN,
            text_color=self.COLOR_SINGLE_BTN_TEXT,
            font=self.BUTTON_FONT, state="disabled",
        )
        self.available_bars_btn.pack(side="left", padx=(8, 2))

        action_row = ctk.CTkFrame(self.fetch_frame, fg_color="transparent")
        action_row.pack(fill="x", padx=2, pady=2)

        self.fetch_all_btn = ctk.CTkButton(
            action_row, text="All", width=50,
            command=self._fetch_all_symbols,
            fg_color=self.COLOR_ALL_BTN,
            text_color=self.COLOR_ALL_BTN_TEXT,
            font=self.BUTTON_FONT, state="disabled",
        )
        self.fetch_all_btn.pack(side="left", padx=2)

        self.fetch_single_btn = ctk.CTkButton(
            action_row, text="Single", width=70,
            command=self._fetch_single_symbol,
            fg_color=self.COLOR_SINGLE_BTN,
            text_color=self.COLOR_SINGLE_BTN_TEXT,
            font=self.BUTTON_FONT, state="disabled",
        )
        self.fetch_single_btn.pack(side="left", padx=2)

        ctk.CTkLabel(action_row, text="Symbol:", font=self.UI_FONT).pack(side="left", padx=2)

        self.symbol_combo = ctk.CTkComboBox(
            action_row, width=100, state="readonly", border_width=2,
            font=self.UI_FONT, dropdown_font=self.UI_FONT,
            command=self._on_symbol_selected,
        )
        self.symbol_combo.pack(side="left", fill="x", expand=True, padx=2)
        self.symbol_combo.set("Select Symbol")

        self.refresh_symbols_btn = ctk.CTkButton(
            action_row, text="↻", width=32,
            command=self._refresh_symbols,
            font=self.UI_FONT, state="disabled",
        )
        self.refresh_symbols_btn.pack(side="left", padx=2)

        tf_row = ctk.CTkFrame(self.fetch_frame, fg_color="transparent")
        tf_row.pack(fill="x", padx=2, pady=2)

        ctk.CTkLabel(tf_row, text="TF:", width=30, font=self.UI_FONT).pack(
            side="left", padx=2,
        )

        for tf in Mt5ApiFetchData.TIMEFRAME_ORDER:
            ctk.CTkCheckBox(
                tf_row, text=tf, width=60, variable=self.timeframe_vars[tf],
                font=self.UI_FONT, border_width=2,
                command=self._on_tf_change,
            ).pack(side="left", padx=2, fill="x", expand=True)

    def _build_log_ui(self):
        ctk.CTkLabel(
            self.log_frame, text=self.LOG_TITLE,
            font=self.TITLE_FONT, text_color=self.TITLE_COLOR, anchor="w",
        ).pack(fill="x", padx=2, pady=(2, 2))

        self.log_text = ctk.CTkTextbox(
            self.log_frame, font=self.MONO_FONT, wrap="word", border_width=2,
            fg_color=self.LOG_BG_COLOR,
        )
        self.log_text.pack(fill="both", expand=True, padx=2, pady=(2, 2))
        self.log_text.configure(state="disabled")

        menu = tk.Menu(self.log_text, tearoff=0)
        menu.add_command(label="Copy", command=self._copy_log)
        menu.add_command(label="Clear", command=self._clear_log)
        self.log_text.bind(
            "<Button-3>",
            lambda event: menu.tk_popup(event.x_root, event.y_root),
        )

    def _sync_gui_to_logic(self):
        self.api_client.path_value = self.path_var.get()
        self.api_client.auto_connect_value = self.auto_connect_var.get()
        self.api_client.market_watch_only_value = self.market_watch_only_var.get()
        self.api_client.blacklist_enabled_value = self.blacklist_enabled_var.get()
        self.api_client.blacklist_value = self.blacklist_var.get()

        self.fetch_data.date_from_value = (
            f"{self.from_year_var.get().strip()}-"
            f"{self.from_month_var.get().strip()}-"
            f"{self.from_day_var.get().strip()}"
        )
        self.fetch_data.date_to_value = (
            f"{self.to_year_var.get().strip()}-"
            f"{self.to_month_var.get().strip()}-"
            f"{self.to_day_var.get().strip()}"
        )
        self.fetch_data.data_folder_path = self.data_folder_var.get()
        self.fetch_data.selected_timeframes = [
            tf for tf in Mt5ApiFetchData.TIMEFRAME_ORDER
            if self.timeframe_vars[tf].get()
        ]

    def _wire_callbacks(self):
        self.api_client.on_connect_success = self._after_connected

    def _on_opt_change(self):
        self._sync_gui_to_logic()
        self.api_client.save_state()

    def _on_auto_connect_change(self):
        self._sync_gui_to_logic()
        if self.auto_connect_var.get():
            self._log_append("Auto Connect enabled")
            self._start_auto_connect()
        else:
            self._log_append("Auto Connect disabled")
            self._stop_auto_connect()
        self.api_client.save_state()

    def _start_auto_connect(self):
        self._stop_auto_connect()
        self._auto_connect_job = self.after(
            Mt5ApiClient.AUTO_CONNECT_INTERVAL_MS, self._auto_connect_tick,
        )

    def _stop_auto_connect(self):
        if self._auto_connect_job:
            self.after_cancel(self._auto_connect_job)
            self._auto_connect_job = None

    def _auto_connect_tick(self):
        if self._closing:
            return
        if self.auto_connect_var.get() and not self.api_client.connected:
            self._manual_connect()
        self._auto_connect_job = self.after(
            Mt5ApiClient.AUTO_CONNECT_INTERVAL_MS, self._auto_connect_tick,
        )

    def _manual_connect(self):
        path = self.path_var.get().strip() or None
        success = self.api_client.connect(path_override=path)
        self._update_connection_status(success)

    def _disconnect(self):
        self.api_client.disconnect()
        self._update_connection_status(False)

    def _update_connection_status(self, connected):
        if connected:
            self.conn_status_dot.configure(text_color=self.COLOR_STATUS_OK)
            self.connect_btn.configure(
                text="Disconnect",
                fg_color=self.COLOR_DISCONNECT,
                command=self._disconnect,
            )
            self.refresh_symbols_btn.configure(state="normal")
            self.fetch_all_btn.configure(state="normal")
            self.fetch_single_btn.configure(state="normal")
            self.available_bars_btn.configure(state="normal")
            self.after(300, self._after_connected)
        else:
            self.conn_status_dot.configure(text_color=self.COLOR_STATUS_BAD)
            self.connect_btn.configure(
                text="Connect",
                fg_color=self.COLOR_CONNECT,
                command=self._manual_connect,
            )
            self.refresh_symbols_btn.configure(state="disabled")
            self.fetch_all_btn.configure(state="disabled")
            self.fetch_single_btn.configure(state="disabled")
            self.available_bars_btn.configure(state="disabled")

    def _after_connected(self):
        def worker():
            symbols = self.api_client.fetch_symbols()
            symbols = self.api_client.filter_symbols(symbols)
            if symbols:
                self._log_append(f"✅ Loaded {len(symbols)} symbols")
                self.after(0, lambda: self._update_symbols_dropdown(symbols))
            else:
                self._log_append(
                    "⚠️ No symbols available after blacklist filter",
                )

        threading.Thread(target=worker, daemon=True).start()

    def _show_marketwatch_symbols(self):
        if not self.api_client.connected:
            raise RuntimeError("Symbols: not connected")
        symbols = self.api_client.fetch_symbols()
        label = "Market Watch" if self.market_watch_only_var.get() else "All Symbols"
        if self.blacklist_enabled_var.get():
            label += " (blacklist)"
        self._log_append(f"--- {label} ({len(symbols)}) ---")
        self._log_append(", ".join(symbols) if symbols else "(none)")

    def _select_data_folder(self):
        folder = filedialog.askdirectory(
            initialdir=self.data_folder_var.get(),
            parent=self,
        )
        if folder:
            self.data_folder_var.set(folder)
            self._sync_gui_to_logic()
            self.fetch_data.save_config()

    def _on_symbol_selected(self, event=None):
        return

    def _on_tf_change(self):
        self._sync_gui_to_logic()
        self.fetch_data.save_config()

    def _show_available_bars(self):
        if not self.fetch_data.is_connected():
            raise RuntimeError("Available Bars: not connected")

        self._sync_gui_to_logic()
        symbols = self.fetch_data._filter_symbols(self.fetch_data.fetch_symbols())
        if not symbols:
            raise RuntimeError("Available Bars: no Market Watch symbols available")

        selected_tfs = self.fetch_data.get_selected_timeframes()
        if not selected_tfs:
            raise RuntimeError("Available Bars: no timeframe selected")

        dt_from, dt_to = self.fetch_data._resolve_datetime_range()
        self.fetch_data.save_config()
        self._log_append(
            f"🔍 Scanning available bars for {len(symbols)} symbols "
            f"{dt_from:%Y-%m-%d} → {dt_to:%Y-%m-%d}",
        )
        self.available_bars_btn.configure(state="disabled")

        def scan_thread():
            self.fetch_data.scan_available_bars(symbols, selected_tfs, dt_from, dt_to)
            self.after(0, lambda: self.available_bars_btn.configure(state="normal"))

        threading.Thread(target=scan_thread, daemon=True).start()

    def _refresh_symbols(self):
        if not self.fetch_data.is_connected():
            raise RuntimeError("Fetch: refresh symbols not connected")

        def get_thread():
            self._log_append("📋 Fetching symbols...")
            symbols = self.fetch_data.fetch_symbols()
            symbols = self.fetch_data._filter_symbols(symbols)
            self.after(0, lambda: self._update_symbols_dropdown(symbols))
            self._log_append(f"✅ Got {len(symbols)} symbols")

        threading.Thread(target=get_thread, daemon=True).start()

    def _update_symbols_dropdown(self, symbols):
        filtered = sorted(set(self.fetch_data._filter_symbols(symbols)))
        self.symbol_combo_values = filtered
        self.symbol_combo.configure(values=filtered)
        if filtered:
            current = self.symbol_combo.get()
            self.symbol_combo.set(
                current if current in filtered else filtered[0],
            )

    def update_symbols(self, symbols):
        filtered = sorted(set(self.fetch_data._filter_symbols(symbols)))
        self.symbol_combo.configure(values=filtered)
        if filtered:
            self.symbol_combo.set(filtered[0])
        self.refresh_symbols_btn.configure(state="normal")
        self.fetch_all_btn.configure(state="normal")
        self.fetch_single_btn.configure(state="normal")
        self.available_bars_btn.configure(state="normal")

    def _fetch_single_symbol(self):
        if not self.fetch_data.is_connected():
            raise RuntimeError("Fetch: single not connected")
        symbol = self.symbol_combo.get()
        if not symbol or symbol == "Select Symbol":
            raise RuntimeError("Fetch: no symbol selected")

        self._sync_gui_to_logic()
        selected_tfs = self.fetch_data.get_selected_timeframes()
        if not selected_tfs:
            raise RuntimeError("Fetch: no timeframe selected")

        dt_from, dt_to = self.fetch_data._resolve_datetime_range()
        self.fetch_data.save_config()
        self._log_append(
            f"🚀 Fetching {symbol} "
            f"{dt_from:%Y-%m-%d} → {dt_to:%Y-%m-%d}"
        )
        self.fetch_all_btn.configure(state="disabled")
        self.fetch_single_btn.configure(state="disabled")
        self.available_bars_btn.configure(state="disabled")

        def fetch_thread():
            self.fetch_data.fetch_single_symbol(symbol, selected_tfs, dt_from, dt_to)
            self.after(0, lambda: self.fetch_all_btn.configure(state="normal"))
            self.after(0, lambda: self.fetch_single_btn.configure(state="normal"))
            self.after(0, lambda: self.available_bars_btn.configure(state="normal"))

        threading.Thread(target=fetch_thread, daemon=True).start()

    def _fetch_all_symbols(self):
        if not self.fetch_data.is_connected():
            raise RuntimeError("Fetch: all not connected")

        self._sync_gui_to_logic()
        symbols = self.fetch_data._filter_symbols(self.fetch_data.fetch_symbols())
        if not symbols:
            raise RuntimeError("Fetch: no Market Watch symbols available")

        selected_tfs = self.fetch_data.get_selected_timeframes()
        if not selected_tfs:
            raise RuntimeError("Fetch: no timeframe selected")

        dt_from, dt_to = self.fetch_data._resolve_datetime_range()
        self.fetch_data.save_config()
        self._log_append(
            f"🚀 Fetching {len(symbols)} symbols "
            f"{dt_from:%Y-%m-%d} → {dt_to:%Y-%m-%d}"
        )
        self.fetch_all_btn.configure(state="disabled")
        self.fetch_single_btn.configure(state="disabled")
        self.available_bars_btn.configure(state="disabled")

        def fetch_thread():
            self.fetch_data.fetch_all_symbols(symbols, selected_tfs, dt_from, dt_to)
            self.after(0, lambda: self.fetch_all_btn.configure(state="normal"))
            self.after(0, lambda: self.fetch_single_btn.configure(state="normal"))
            self.after(0, lambda: self.available_bars_btn.configure(state="normal"))

        threading.Thread(target=fetch_thread, daemon=True).start()

    def _log_append(self, message):
        if self._closing:
            return
        if message == self.CLEAR_SENTINEL:
            self._clear_log()
            return
        box = self.log_text
        box.configure(state="normal")

        at_bottom = box.yview()[1] >= 0.999
        box.insert("end", f"{message}\n")
        lines = int(box.index("end-1c").split(".")[0])
        if lines > Config.LOG_MAX_LINES:
            box.delete("1.0", f"{lines - Config.LOG_MAX_LINES}.0")
        if at_bottom:
            box.see("end")
        box.configure(state="disabled")

    def _copy_log(self):
        text = (
            self.log_text.selection_get()
            if self.log_text.tag_ranges("sel")
            else self.log_text.get("1.0", "end-1c")
        )
        self.clipboard_clear()
        self.clipboard_append(text)

    def _clear_log(self):
        self.log_text.configure(state="normal")
        self.log_text.delete("1.0", "end")
        self.log_text.configure(state="disabled")

    def stop(self):
        self._closing = True
        self._stop_auto_connect()
        self.api_client.stop()

    def destroy(self):
        self.stop()
        self.api_client.save_state()
        self.fetch_data.save_config()
        super().destroy()

    def report_callback_exception(self, exc_type, exc_value, exc_tb):
        self._log_append(f"❌ {exc_type.__name__}: {exc_value}")

    def _on_close(self):
        confirmed = messagebox.askyesno(
            "Exit MT5 Fetch API",
            "Are you sure you want to quit?\n\n"
            "Any running operations will be stopped.",
            icon="question",
            parent=self,
        )
        if not confirmed:
            return

        self.destroy()
        Config.save_window_geometry(self.ini_path, self.wm_geometry())


if __name__ == "__main__":
    Mt5ApiFetchGUI().mainloop()