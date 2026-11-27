#!/usr/bin/env python3
# yfinance_fetch.py
# Copyright Su Nie | BSD-3C License | https://github.com/can87683

"""
customtkinter, setproctitle, yfinance
"""

import os
import sys
import setproctitle
setproctitle.setproctitle(os.path.basename(sys.argv[0]))

import configparser
import csv
import datetime
import pathlib
import queue
import re
import shutil
import tempfile
import threading
import time
import tkinter as tk
import tkinter.filedialog
import tkinter.messagebox
import customtkinter as ctk
import yfinance


class Config:
    """Static configuration and INI merge helper."""

    WINDOW_WIDTH = 640
    WINDOW_HEIGHT = 1020
    INI_FILE = str(pathlib.Path(sys.argv[0]).resolve().with_suffix(".ini"))
    LOG_MAX_LINES = 1000

    INI_SECTION_ORDER = ("Window", "Yfinance", "Fetch")
    INI_SECTION_WINDOW = "Window"

    COLOR_BACKGROUND = "#1c1c1c"
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

        preferred = [item.lower() for item in Config.INI_SECTION_ORDER]
        ordered = configparser.ConfigParser(interpolation=None)
        ordered.optionxform = str

        for name in preferred:
            if disk.has_section(name):
                ordered.add_section(name)
                for key, value in disk.items(name):
                    ordered.set(name, key, value)

        for name in disk.sections():
            if name not in preferred:
                ordered.add_section(name)
                for key, value in disk.items(name):
                    ordered.set(name, key, value)

        directory = os.path.dirname(os.path.abspath(path)) or "."
        fd, temporary_path = tempfile.mkstemp(
            prefix=".yf_", suffix=".ini", dir=directory,
        )

        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            ordered.write(handle)

        if os.name == "posix":
            os.chmod(temporary_path, 0o600)

        os.replace(temporary_path, path)

    @staticmethod
    def load_window_geometry(path):
        parser = configparser.ConfigParser(interpolation=None)

        if os.path.exists(path):
            parser.read(path, encoding="utf-8")

        if not parser.has_section(Config.INI_SECTION_WINDOW):
            return None, None

        geometry = parser.get(
            Config.INI_SECTION_WINDOW, "geometry", fallback="",
        )

        if geometry:
            _size, position = geometry.split("+", 1)
            x_value, y_value = position.split("+", 1)
            return int(x_value), int(y_value)

        x_value = parser.getint(
            Config.INI_SECTION_WINDOW, "x", fallback=None,
        )
        y_value = parser.getint(
            Config.INI_SECTION_WINDOW, "y", fallback=None,
        )
        return x_value, y_value

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
            fd, temporary_path = tempfile.mkstemp(
                prefix=".yf_", suffix=".ini", dir=directory,
            )
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                parser.write(handle)

            if os.name == "posix":
                os.chmod(temporary_path, 0o600)

            os.replace(temporary_path, path)

        Config.atomic_merge_ini(
            path,
            Config.INI_SECTION_WINDOW,
            {"geometry": geometry},
        )


class YfinanceClient:
    """Pure-Python yfinance client (no GUI)."""

    AUTO_CONNECT_DEFAULT = True
    PROBE_TICKER = "BTC-USD"

    TF_INTERVAL = {
        "M5": "5m",
        "M15": "15m",
        "H1": "60m",
        "H4": "60m",
        "D1": "1d",
        "W1": "1wk",
    }

    TF_RESAMPLE = {"H4": "4h"}

    INTRADAY_MAX_DAYS = {
        "M5": 59,
        "M15": 59,
        "H1": 729,
        "H4": 729,
    }

    OHLC_AGG = {
        "Open": "first",
        "High": "max",
        "Low": "min",
        "Close": "last",
        "Volume": "sum",
    }

    def __init__(self, config_path=None, log_callback=None):
        self.ini_path = config_path or Config.INI_FILE
        self.log_callback = log_callback

        self.connected = False
        self.on_connect_success = None

        self.ini = configparser.ConfigParser(interpolation=None)
        if os.path.exists(self.ini_path):
            self.ini.read(self.ini_path, encoding="utf-8")

        self._load_state()

    def _load_state(self):
        if not self.ini.has_section("Yfinance"):
            self.ini.add_section("Yfinance")

        self.auto_connect = self.ini.getboolean(
            "Yfinance",
            "auto_connect",
            fallback=self.AUTO_CONNECT_DEFAULT,
        )

    def save_state(self):
        Config.atomic_merge_ini(
            self.ini_path,
            "Yfinance",
            {"auto_connect": str(self.auto_connect)},
        )

    def log(self, message):
        if self.log_callback:
            self.log_callback(message)

    def probe(self):
        """Blocking probe of yfinance reachability. Returns bar count."""
        return len(
            yfinance.Ticker(self.PROBE_TICKER).history(
                period="5d",
                interval="1d",
            )
        )

    def finish_connect(self, bars):
        """Apply probe result. Returns True if connected."""
        if bars == 0:
            self.log(
                f"❌ yfinance probe failed: no data for {self.PROBE_TICKER}",
            )
            self.connected = False
            return False

        self.connected = True
        self.log(
            f"✅ Connected: yfinance reachable ({self.PROBE_TICKER})",
        )
        self.save_state()

        if callable(self.on_connect_success):
            self.on_connect_success()

        return True

    def disconnect(self):
        self.connected = False
        self.log("🔌 Disconnected from yfinance")

    def stop(self):
        self.connected = False

    def get_rates(self, ticker, timeframe, dt_from, dt_to):
        max_days = self.INTRADAY_MAX_DAYS.get(timeframe)

        if max_days:
            dt_from = max(
                dt_from,
                datetime.datetime.now(datetime.timezone.utc)
                - datetime.timedelta(days=max_days),
            )

        if dt_from >= dt_to:
            self.log(
                f"⚠️ {ticker} {timeframe}: "
                "range outside yfinance history limit",
            )
            return []

        frame = yfinance.Ticker(ticker).history(
            start=dt_from,
            end=dt_to,
            interval=self.TF_INTERVAL[timeframe],
            auto_adjust=False,
        )

        if timeframe in self.TF_RESAMPLE:
            frame = (
                frame.resample(self.TF_RESAMPLE[timeframe])
                .agg(self.OHLC_AGG)
                .dropna(subset=["Open"])
            )

        if frame.empty:
            self.log(
                f"⚠️ yfinance returned no data for {ticker} {timeframe}",
            )
            return []

        return [
            {
                "time": int(timestamp.timestamp()),
                "open": float(row["Open"]),
                "high": float(row["High"]),
                "low": float(row["Low"]),
                "close": float(row["Close"]),
                "tick_volume": int(row["Volume"]),
            }
            for timestamp, row in frame.iterrows()
        ]


class YfinanceFetchData:
    """Pure-Python historical bar fetcher (no GUI)."""

    DEFAULT_DATA_FOLDER = os.path.join(
        str(pathlib.Path.home()),
        "yfinance_data",
    )

    DEFAULT_TICKERS = (
        "BTC-USD,^NDX,^GSPC,GC=F,SI=F,CL=F,EURUSD=X,CAD=X,JPY=X"
    )

    DEFAULT_FROM_OFFSET_DAYS = 30
    TIMEFRAME_ORDER = ["M5", "M15", "H1", "H4", "D1", "W1"]
    TABLE_COL_WIDTH = 8
    CSV_COLUMNS = ("time", "open", "high", "low", "close", "tick_volume")

    def __init__(self, client, config_path=None, log_callback=None):
        self.client = client
        self.log_callback = log_callback
        self._stop_event = threading.Event()

        if config_path is None:
            script_name = os.path.splitext(
                os.path.basename(__file__),
            )[0]
            config_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                f"{script_name}.ini",
            )

        self.ini_path = config_path
        self.ini = configparser.ConfigParser(interpolation=None)
        if os.path.exists(self.ini_path):
            self.ini.read(self.ini_path, encoding="utf-8")

        self.data_folder_path = self.DEFAULT_DATA_FOLDER
        self.tickers = self._split_tickers(self.DEFAULT_TICKERS)
        self.timeframes = list(self.TIMEFRAME_ORDER)
        self.date_from = None
        self.date_to = None

        self._load_fetch_state()

    def log(self, message):
        if self.log_callback:
            self.log_callback(message)

    @staticmethod
    def _split_tickers(text):
        return [
            ticker.strip()
            for ticker in text.split(",")
            if ticker.strip()
        ]

    def _load_fetch_state(self):
        today = datetime.datetime.now(datetime.timezone.utc).date()
        from_date = today - datetime.timedelta(
            days=self.DEFAULT_FROM_OFFSET_DAYS,
        )
        self.date_from = from_date
        self.date_to = today

        if not self.ini.has_section("Fetch"):
            return

        self.data_folder_path = self.ini.get(
            "Fetch", "data_folder", fallback=self.DEFAULT_DATA_FOLDER,
        )
        self.tickers = self._split_tickers(
            self.ini.get("Fetch", "tickers", fallback=self.DEFAULT_TICKERS),
        )

        saved_timeframes = [
            item.strip()
            for item in self.ini.get(
                "Fetch",
                "fetch_timeframes",
                fallback=",".join(self.TIMEFRAME_ORDER),
            ).split(",")
            if item.strip()
        ]
        self.timeframes = [
            timeframe
            for timeframe in self.TIMEFRAME_ORDER
            if timeframe in saved_timeframes
        ]

        date_from_str = self.ini.get("Fetch", "date_from", fallback="")
        if date_from_str:
            self.date_from = datetime.date.fromisoformat(date_from_str)

        date_to_str = self.ini.get("Fetch", "date_to", fallback="")
        if date_to_str:
            self.date_to = datetime.date.fromisoformat(date_to_str)

    def save_config(self):
        Config.atomic_merge_ini(
            self.ini_path,
            "Fetch",
            {
                "date_from": self.date_from.isoformat(),
                "date_to": self.date_to.isoformat(),
                "fetch_timeframes": ",".join(self.timeframes),
                "data_folder": self.data_folder_path,
                "tickers": ",".join(self.tickers),
            },
        )

    def is_connected(self):
        return self.client.connected

    def resolve_datetime_range(self):
        dt_from = datetime.datetime(
            self.date_from.year,
            self.date_from.month,
            self.date_from.day,
            tzinfo=datetime.timezone.utc,
        )
        dt_to = datetime.datetime(
            self.date_to.year,
            self.date_to.month,
            self.date_to.day,
            tzinfo=datetime.timezone.utc,
        ) + datetime.timedelta(days=1) - datetime.timedelta(seconds=1)

        if dt_from > dt_to:
            raise RuntimeError(
                "Fetch: 'From' date must be before or equal to 'To' date",
            )
        return dt_from, dt_to

    def fetch_rates(self, ticker, timeframe, dt_from=None, dt_to=None):
        if dt_from is None or dt_to is None:
            dt_from, dt_to = self.resolve_datetime_range()
        return self.client.get_rates(ticker, timeframe, dt_from, dt_to)

    @staticmethod
    def format_bars(count):
        if count >= 1_000_000:
            return f"{count / 1_000_000:.1f}M"
        if count >= 1_000:
            return f"{count / 1_000:.1f}K"
        return str(count)

    def wipe_data_folder(self):
        if os.path.isdir(self.data_folder_path):
            shutil.rmtree(self.data_folder_path)
            self.log(f"🧹 Removed {self.data_folder_path}")
        else:
            self.log(
                f"🧹 No existing data folder at {self.data_folder_path}",
            )

    def scan_worker(
        self, tickers, selected_timeframes, dt_from, dt_to,
        on_complete=None, on_line=None,
    ):
        self._stop_event.clear()

        header = "TF:" + "".join(
            " " * ((self.TABLE_COL_WIDTH - len(timeframe)) // 2)
            + timeframe
            for timeframe in self.TIMEFRAME_ORDER
        )
        self.log(header)

        for index, ticker in enumerate(tickers, 1):
            if self._stop_event.is_set():
                break

            counts = {}
            for timeframe in selected_timeframes:
                if self._stop_event.is_set():
                    break
                data = self.fetch_rates(
                    ticker, timeframe, dt_from, dt_to,
                )
                counts[timeframe] = len(data)
                time.sleep(0.05)

            row = ticker.ljust(12)
            for timeframe in self.TIMEFRAME_ORDER:
                count = counts.get(timeframe, 0)
                value = (
                    self.format_bars(count) if count > 0 else "N/A"
                )
                row += (
                    " " * (self.TABLE_COL_WIDTH - len(value)) + value
                )
            self.log(f"{index}/{len(tickers)} {row}")
            if on_line:
                on_line(ticker, counts)
            time.sleep(0.1)

        if not self._stop_event.is_set():
            self.log(
                f"✅ Available bars scan complete: {len(tickers)} tickers",
            )
        if on_complete:
            on_complete()

    def fetch_worker(
        self, tickers, selected_timeframes, dt_from, dt_to, target_dir,
        on_complete=None,
    ):
        self._stop_event.clear()
        self.wipe_data_folder()

        for index, ticker in enumerate(tickers, 1):
            if self._stop_event.is_set():
                break

            self.log(f"📈 {index}/{len(tickers)}: {ticker}")
            for timeframe in selected_timeframes:
                if self._stop_event.is_set():
                    break
                data = self.fetch_rates(
                    ticker, timeframe, dt_from, dt_to,
                )
                if data:
                    self._save_to_csv(
                        ticker, timeframe, data, target_dir,
                    )
                time.sleep(0.1)
            time.sleep(0.2)

        if not self._stop_event.is_set():
            self.log(f"✅ Fetch complete: {len(tickers)} tickers")
        if on_complete:
            on_complete()

    def stop(self):
        self._stop_event.set()

    def _save_to_csv(self, ticker, timeframe, data, target_dir):
        safe_ticker = re.sub(
            r"[^A-Za-z0-9_+#!$-]",
            "_",
            ticker,
        )

        ticker_dir = os.path.join(target_dir, safe_ticker)
        os.makedirs(ticker_dir, exist_ok=True)

        rows = [
            {column: bar[column] for column in self.CSV_COLUMNS if column in bar}
            for bar in data
        ]
        rows = [row for row in rows if row]
        if not rows:
            return

        filepath = os.path.join(
            ticker_dir, f"{safe_ticker}_{timeframe}.csv",
        )

        with open(filepath, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=list(self.CSV_COLUMNS),
                lineterminator=os.linesep,
            )
            writer.writeheader()
            writer.writerows(rows)

        self.log(f"💾 Saved to {filepath}")


class YfinanceFetchGUI(ctk.CTk):
    """Standalone yfinance GUI."""

    TITLE = "Yfinance Fetch - Copyright Su Nie | BSD-3C License | https://github.com/can87683"

    TITLE_FONT = ("TkDefaultFont", 14, "bold")
    TITLE_COLOR = "yellow"
    UI_FONT = ("TkDefaultFont", 12)
    BUTTON_FONT = ("TkDefaultFont", 12, "bold")
    MONO_FONT = ("Courier", 12)
    LOG_FONT = ("Consolas", 10)

    COLOR_CONNECT = "#2196F3"
    COLOR_DISCONNECT = "#F44336"
    COLOR_CLEAR_LOG = "#546E7A"
    COLOR_STATUS_OK = "#4CAF50"
    COLOR_STATUS_BAD = "#F44336"

    COLOR_ALL_BTN = "#4285F4"
    COLOR_ALL_BTN_TEXT = "yellow"
    COLOR_SINGLE_BTN = "#E67E22"
    COLOR_SINGLE_BTN_TEXT = "white"

    LOG_BACKGROUND = "#1c1c1c"
    LOG_FOREGROUND = "white"
    LOG_MAX_LINES = Config.LOG_MAX_LINES

    BORDER_WIDTH = Config.BORDER_WIDTH
    BORDER_COLOR = Config.COLOR_BORDER

    LOG_POLL_INTERVAL_MS = 50
    AUTO_CONNECT_INTERVAL_MS = 5000

    FRAME_W = 636
    FRAME_H = 105

    def __init__(self, config_path=None, **kwargs):
        super().__init__(**kwargs)

        self.title(self.TITLE)
        self.geometry(f"{Config.WINDOW_WIDTH}x{Config.WINDOW_HEIGHT}")
        self.resizable(False, False)
        self.configure(fg_color=Config.COLOR_BACKGROUND)

        self.ini_path = config_path or Config.INI_FILE

        x, y = Config.load_window_geometry(self.ini_path)
        if x is not None and y is not None:
            self.geometry(f"+{x}+{y}")
        else:
            self.eval("tk::PlaceWindow . center")

        self._closing = False
        self._ui_queue = queue.Queue()
        self._log_poll_job = None
        self._auto_connect_job = None
        self._probing = False

        self.client = YfinanceClient(
            config_path=self.ini_path,
            log_callback=self.log,
        )
        self.fetch_data = YfinanceFetchData(
            client=self.client,
            config_path=self.ini_path,
            log_callback=self.log,
        )
        self.client.on_connect_success = self._on_connect_success

        self._build_ui()
        self._apply_state_to_ui()

        if self.client.auto_connect:
            self._auto_connect_job = self.after(
                self.AUTO_CONNECT_INTERVAL_MS,
                self._auto_connect_check,
            )

        self._log_poll_job = self.after(
            self.LOG_POLL_INTERVAL_MS,
            self._drain_ui_queue,
        )

        self.protocol("WM_DELETE_WINDOW", self.request_standalone_close)

    def _build_ui(self):
        self._build_client_frame()
        self._build_fetch_frame()
        self._build_log_frame()

        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(2, weight=1)

    def _build_client_frame(self):
        frame = ctk.CTkFrame(
            self,
            border_width=self.BORDER_WIDTH,
            border_color=self.BORDER_COLOR,
            fg_color=Config.COLOR_BACKGROUND,
        )
        frame.grid(row=0, column=0, sticky="ew", padx=2, pady=2)
        frame.configure(width=self.FRAME_W, height=self.FRAME_H)
        frame.pack_propagate(False)
        frame.grid_propagate(False)
        self.client_frame = frame

        ctk.CTkLabel(
            frame,
            text="Yfinance Fetch API",
            font=self.TITLE_FONT,
            text_color=self.TITLE_COLOR,
            anchor="w",
        ).pack(fill="x", padx=2, pady=2)

        row = ctk.CTkFrame(frame, fg_color="transparent")
        row.pack(fill="x", padx=2, pady=2)

        self.auto_connect_var = ctk.BooleanVar(
            value=self.client.auto_connect,
        )
        self.auto_connect_cb = ctk.CTkCheckBox(
            row,
            text="Auto Connect",
            variable=self.auto_connect_var,
            font=self.UI_FONT,
            command=self._on_auto_connect_change,
        )
        self.auto_connect_cb.pack(side="left", padx=2, pady=2)

        self.connect_btn = ctk.CTkButton(
            row,
            text="Connect",
            command=self._manual_connect,
            width=100,
            fg_color=self.COLOR_CONNECT,
            text_color="white",
            font=self.BUTTON_FONT,
            border_width=2,
        )
        self.connect_btn.pack(side="left", padx=2, pady=2)

        ctk.CTkButton(
            row,
            text="Clear Log",
            command=self.clear_log,
            width=80,
            fg_color=self.COLOR_CLEAR_LOG,
            text_color="white",
            font=self.BUTTON_FONT,
            border_width=2,
        ).pack(side="left", padx=2, pady=2)

        self.conn_status_dot = ctk.CTkLabel(
            row,
            text="●",
            text_color=self.COLOR_STATUS_BAD,
            font=self.UI_FONT,
        )
        self.conn_status_dot.pack(side="left", padx=2, pady=2)

    def _build_fetch_frame(self):
        frame = ctk.CTkFrame(
            self,
            border_width=self.BORDER_WIDTH,
            border_color=self.BORDER_COLOR,
            fg_color="transparent",
        )
        frame.grid(row=1, column=0, sticky="ew", padx=2, pady=2)
        self.fetch_frame = frame

        ctk.CTkLabel(
            frame,
            text="Fetch Yfinance Data",
            font=self.TITLE_FONT,
            text_color=self.TITLE_COLOR,
            anchor="w",
        ).pack(fill="x", padx=2, pady=2)

        self._build_folder_row(frame)
        self._build_date_range_row(frame)
        self._build_action_row(frame)
        self._build_timeframe_row(frame)

    def _build_folder_row(self, parent):
        row = ctk.CTkFrame(parent, fg_color="transparent")
        row.pack(fill="x", padx=2, pady=2)

        ctk.CTkButton(
            row,
            text="Yfinance Data Folder",
            font=self.UI_FONT,
            command=self._select_data_folder,
        ).pack(side="left", padx=2, pady=2)

        self.data_folder_label = ctk.CTkLabel(
            row,
            text="",
            font=self.UI_FONT,
            anchor="w",
        )
        self.data_folder_label.pack(
            side="left", padx=2, pady=2, fill="x", expand=True,
        )

    def _build_date_range_row(self, parent):
        row = ctk.CTkFrame(parent, fg_color="transparent")
        row.pack(fill="x", padx=2, pady=2)

        ctk.CTkLabel(row, text="From", font=self.UI_FONT, anchor="w").pack(
            side="left", padx=2, pady=2,
        )
        self.from_year = ctk.CTkEntry(
            row, width=54, font=self.UI_FONT, border_width=2,
        )
        self.from_year.pack(side="left", padx=2, pady=2)
        ctk.CTkLabel(row, text="-", font=self.UI_FONT).pack(
            side="left", padx=2, pady=2,
        )
        self.from_month = ctk.CTkEntry(
            row, width=40, font=self.UI_FONT, border_width=2,
        )
        self.from_month.pack(side="left", padx=2, pady=2)
        ctk.CTkLabel(row, text="-", font=self.UI_FONT).pack(
            side="left", padx=2, pady=2,
        )
        self.from_day = ctk.CTkEntry(
            row, width=40, font=self.UI_FONT, border_width=2,
        )
        self.from_day.pack(side="left", padx=2, pady=2)

        ctk.CTkLabel(row, text="To", font=self.UI_FONT).pack(
            side="left", padx=4, pady=2,
        )
        self.to_year = ctk.CTkEntry(
            row, width=54, font=self.UI_FONT, border_width=2,
        )
        self.to_year.pack(side="left", padx=2, pady=2)
        ctk.CTkLabel(row, text="-", font=self.UI_FONT).pack(
            side="left", padx=2, pady=2,
        )
        self.to_month = ctk.CTkEntry(
            row, width=40, font=self.UI_FONT, border_width=2,
        )
        self.to_month.pack(side="left", padx=2, pady=2)
        ctk.CTkLabel(row, text="-", font=self.UI_FONT).pack(
            side="left", padx=2, pady=2,
        )
        self.to_day = ctk.CTkEntry(
            row, width=40, font=self.UI_FONT, border_width=2,
        )
        self.to_day.pack(side="left", padx=2, pady=2)

        self.available_bars_btn = ctk.CTkButton(
            row,
            text="Available Bars",
            width=110,
            command=self._show_available_bars,
            fg_color=self.COLOR_SINGLE_BTN,
            text_color=self.COLOR_SINGLE_BTN_TEXT,
            font=self.BUTTON_FONT,
            state="disabled",
        )
        self.available_bars_btn.pack(side="left", padx=2, pady=2)

    def _build_action_row(self, parent):
        row = ctk.CTkFrame(parent, fg_color="transparent")
        row.pack(fill="x", padx=2, pady=2)

        self.fetch_btn = ctk.CTkButton(
            row,
            text="Fetch",
            width=70,
            command=self._fetch_tickers,
            fg_color=self.COLOR_ALL_BTN,
            text_color=self.COLOR_ALL_BTN_TEXT,
            font=self.BUTTON_FONT,
            state="disabled",
        )
        self.fetch_btn.pack(side="left", padx=2, pady=2)

        ctk.CTkLabel(row, text="Ticker:", font=self.UI_FONT).pack(
            side="left", padx=2, pady=2,
        )
        self.ticker_entry = ctk.CTkEntry(
            row, font=self.UI_FONT, border_width=2,
        )
        self.ticker_entry.pack(
            side="left", fill="x", expand=True, padx=2, pady=2,
        )

    def _build_timeframe_row(self, parent):
        row = ctk.CTkFrame(parent, fg_color="transparent")
        row.pack(fill="x", padx=2, pady=2)

        ctk.CTkLabel(row, text="TF:", width=30, font=self.UI_FONT).pack(
            side="left", padx=2, pady=2,
        )

        self.timeframe_vars = {}
        for timeframe in YfinanceFetchData.TIMEFRAME_ORDER:
            var = ctk.BooleanVar(value=True)
            self.timeframe_vars[timeframe] = var
            ctk.CTkCheckBox(
                row,
                text=timeframe,
                width=60,
                variable=var,
                font=self.UI_FONT,
                border_width=2,
            ).pack(
                side="left", padx=2, pady=2, fill="x", expand=True,
            )

    def _build_log_frame(self):
        self.log_frame = ctk.CTkFrame(
            self,
            fg_color=Config.COLOR_BACKGROUND,
            border_width=self.BORDER_WIDTH,
            border_color=self.BORDER_COLOR,
        )
        self.log_frame.grid(
            row=2, column=0, sticky="nsew", padx=2, pady=2,
        )

        ctk.CTkLabel(
            self.log_frame,
            text="Log",
            font=self.TITLE_FONT,
            text_color=self.TITLE_COLOR,
            anchor="w",
        ).pack(fill="x", padx=2, pady=2)

        self.log_text = tk.Text(
            self.log_frame,
            font=self.LOG_FONT,
            wrap="word",
            bd=0,
            highlightthickness=0,
            bg=self.LOG_BACKGROUND,
            fg=self.LOG_FOREGROUND,
            insertbackground=self.LOG_FOREGROUND,
            selectbackground="#345b80",
            selectforeground="white",
            undo=False,
        )
        self.log_text.pack(fill="both", expand=True, padx=2, pady=2)

        self.log_text.bind("<Key>", self._block_log_edit)
        self.log_text.bind("<Control-c>", self._copy_log)
        self.log_text.bind("<Control-C>", self._copy_log)
        self.log_text.bind("<Control-a>", self._select_all_log)
        self.log_text.bind("<Control-A>", self._select_all_log)

        menu = tk.Menu(self.log_text, tearoff=0)
        menu.add_command(label="Copy", command=self.copy_log)
        menu.add_command(label="Copy All", command=self.copy_all_log)
        menu.add_command(label="Clear", command=self.clear_log)
        self.log_menu = menu
        self.log_text.bind("<Button-3>", self._show_log_menu)

    def _apply_state_to_ui(self):
        fd = self.fetch_data

        self.data_folder_label.configure(text=fd.data_folder_path)
        self.ticker_entry.delete(0, "end")
        self.ticker_entry.insert(0, ",".join(fd.tickers))

        self.from_year.delete(0, "end")
        self.from_year.insert(0, f"{fd.date_from.year:04d}")
        self.from_month.delete(0, "end")
        self.from_month.insert(0, f"{fd.date_from.month:02d}")
        self.from_day.delete(0, "end")
        self.from_day.insert(0, f"{fd.date_from.day:02d}")

        self.to_year.delete(0, "end")
        self.to_year.insert(0, f"{fd.date_to.year:04d}")
        self.to_month.delete(0, "end")
        self.to_month.insert(0, f"{fd.date_to.month:02d}")
        self.to_day.delete(0, "end")
        self.to_day.insert(0, f"{fd.date_to.day:02d}")

        for timeframe, var in self.timeframe_vars.items():
            var.set(timeframe in fd.timeframes)

    def _collect_state_from_ui(self):
        fd = self.fetch_data
        fd.tickers = [
            ticker.strip()
            for ticker in self.ticker_entry.get().split(",")
            if ticker.strip()
        ]
        fd.timeframes = [
            timeframe
            for timeframe in YfinanceFetchData.TIMEFRAME_ORDER
            if self.timeframe_vars[timeframe].get()
        ]
        fd.date_from = datetime.date(
            int(self.from_year.get().strip()),
            int(self.from_month.get().strip()),
            int(self.from_day.get().strip()),
        )
        fd.date_to = datetime.date(
            int(self.to_year.get().strip()),
            int(self.to_month.get().strip()),
            int(self.to_day.get().strip()),
        )

    def save_config(self):
        self._collect_state_from_ui()
        self.fetch_data.save_config()
        self.client.auto_connect = bool(self.auto_connect_var.get())
        self.client.save_state()

    def log(self, message):
        if not self._closing:
            self._ui_queue.put(("log", message))

    def queue_ui(self, callback, *args, **kwargs):
        if not self._closing:
            self._ui_queue.put(("callback", callback, args, kwargs))

    def _drain_ui_queue(self):
        if self._closing:
            return
        while not self._ui_queue.empty():
            item = self._ui_queue.get_nowait()
            if item[0] == "log":
                self._append_log(item[1])
            elif item[0] == "callback":
                callback, args, kwargs = item[1], item[2], item[3]
                callback(*args, **kwargs)
        self._log_poll_job = self.after(
            self.LOG_POLL_INTERVAL_MS, self._drain_ui_queue,
        )

    def _append_log(self, message):
        if message == "__clear__":
            self.clear_log()
            return
        box = self.log_text
        at_bottom = box.yview()[1] >= 0.999
        box.insert("end", f"{message}\n")
        line_count = int(box.index("end-1c").split(".")[0])
        if line_count > self.LOG_MAX_LINES:
            box.delete("1.0", f"{line_count - self.LOG_MAX_LINES}.0")
        if at_bottom:
            box.see("end")

    def _block_log_edit(self, _event):
        return "break"

    def _copy_log(self, _event):
        self.copy_log()
        return "break"

    def _select_all_log(self, _event):
        self.log_text.tag_add("sel", "1.0", "end-1c")
        return "break"

    def _show_log_menu(self, event):
        self.log_menu.tk_popup(event.x_root, event.y_root)

    def copy_log(self):
        if self.log_text.tag_ranges("sel"):
            text = self.log_text.selection_get()
        else:
            text = self.log_text.get("1.0", "end-1c")
        self.clipboard_clear()
        self.clipboard_append(text)

    def copy_all_log(self):
        text = self.log_text.get("1.0", "end-1c")
        self.clipboard_clear()
        self.clipboard_append(text)

    def clear_log(self):
        self.log_text.delete("1.0", "end")

    def _on_auto_connect_change(self):
        self.client.auto_connect = bool(self.auto_connect_var.get())

        if self.auto_connect_var.get():
            self.log("Auto Connect enabled")
            if self._auto_connect_job:
                self.after_cancel(self._auto_connect_job)
                self._auto_connect_job = None
            self._auto_connect_job = self.after(
                self.AUTO_CONNECT_INTERVAL_MS,
                self._auto_connect_check,
            )
        else:
            self.log("Auto Connect disabled")
            if self._auto_connect_job:
                self.after_cancel(self._auto_connect_job)
                self._auto_connect_job = None

        self.client.save_state()

    def _auto_connect_check(self):
        if self._closing:
            return
        if self.auto_connect_var.get() and not self.client.connected:
            self._manual_connect()
        self._auto_connect_job = self.after(
            self.AUTO_CONNECT_INTERVAL_MS, self._auto_connect_check,
        )

    def _manual_connect(self):
        if self._probing or self._closing:
            return
        self._probing = True
        threading.Thread(target=self._probe_worker, daemon=True).start()

    def _probe_worker(self):
        bars = self.client.probe()
        self.queue_ui(self._finish_connect, bars)

    def _finish_connect(self, bars):
        self._probing = False
        if self._closing:
            return
        connected = self.client.finish_connect(bars)
        self._update_connection_status(connected)

    def _update_connection_status(self, connected):
        if connected:
            self.conn_status_dot.configure(text_color=self.COLOR_STATUS_OK)
            self.connect_btn.configure(
                text="Disconnect",
                fg_color=self.COLOR_DISCONNECT,
                command=self._disconnect,
            )
        else:
            self.conn_status_dot.configure(text_color=self.COLOR_STATUS_BAD)
            self.connect_btn.configure(
                text="Connect",
                fg_color=self.COLOR_CONNECT,
                command=self._manual_connect,
            )

    def _disconnect(self):
        self.client.disconnect()
        self._update_connection_status(False)

    def _on_connect_success(self):
        self.fetch_btn.configure(state="normal")
        self.available_bars_btn.configure(state="normal")

    def _select_data_folder(self):
        folder = tkinter.filedialog.askdirectory(
            initialdir=self.fetch_data.data_folder_path,
            parent=self,
        )
        if folder:
            self.fetch_data.data_folder_path = folder
            self.data_folder_label.configure(text=folder)
            self.save_config()

    def _show_available_bars(self):
        if not self.client.connected:
            raise RuntimeError("Available Bars: not connected")

        self._collect_state_from_ui()
        dt_from, dt_to = self.fetch_data.resolve_datetime_range()

        tickers = list(self.fetch_data.tickers)
        selected_timeframes = list(self.fetch_data.timeframes)

        if not tickers:
            self.log("❌ Available Bars: no tickers entered")
            return
        if not selected_timeframes:
            self.log("❌ Available Bars: no timeframe selected")
            return

        self.save_config()
        self.log(
            f"🔍 Scanning available bars for {len(tickers)} tickers "
            f"{dt_from:%Y-%m-%d} → {dt_to:%Y-%m-%d}",
        )
        self.available_bars_btn.configure(state="disabled")

        threading.Thread(
            target=self.fetch_data.scan_worker,
            args=(tickers, selected_timeframes, dt_from, dt_to),
            kwargs={
                "on_complete": lambda: self.queue_ui(
                    self.available_bars_btn.configure, state="normal",
                ),
            },
            daemon=True,
        ).start()

    def _fetch_tickers(self):
        if not self.client.connected:
            raise RuntimeError("Fetch: not connected")

        self._collect_state_from_ui()
        dt_from, dt_to = self.fetch_data.resolve_datetime_range()

        tickers = list(self.fetch_data.tickers)
        selected_timeframes = list(self.fetch_data.timeframes)
        target_dir = self.fetch_data.data_folder_path

        if not tickers:
            self.log("❌ Fetch: no tickers entered")
            return
        if not selected_timeframes:
            self.log("❌ Fetch: no timeframe selected")
            return

        self.save_config()
        self.log(
            f"🚀 Fetching {len(tickers)} tickers "
            f"{dt_from:%Y-%m-%d} → {dt_to:%Y-%m-%d}",
        )
        self.fetch_btn.configure(state="disabled")
        self.available_bars_btn.configure(state="disabled")

        def _done():
            self.queue_ui(self.fetch_btn.configure, state="normal")
            self.queue_ui(
                self.available_bars_btn.configure, state="normal",
            )

        threading.Thread(
            target=self.fetch_data.fetch_worker,
            args=(
                tickers, selected_timeframes, dt_from, dt_to, target_dir,
            ),
            kwargs={"on_complete": _done},
            daemon=True,
        ).start()

    def shutdown(self):
        if self._closing:
            return
        self._closing = True

        if self._log_poll_job:
            self.after_cancel(self._log_poll_job)
            self._log_poll_job = None
        if self._auto_connect_job:
            self.after_cancel(self._auto_connect_job)
            self._auto_connect_job = None

        self.fetch_data.stop()
        self.client.stop()
        self.save_config()

    def request_standalone_close(self):
        confirmed = tkinter.messagebox.askyesno(
            "Exit Yfinance Fetch API",
            "Are you sure you want to quit?\n\n"
            "Background operations will stop after any current "
            "network request finishes.",
            icon="question",
            parent=self,
        )
        if not confirmed:
            return
        Config.save_window_geometry(self.ini_path, self.geometry())
        self.shutdown()
        self.destroy()

    def destroy(self):
        self.shutdown()
        super().destroy()


if __name__ == "__main__":
    YfinanceFetchGUI().mainloop()