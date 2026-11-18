#!/usr/bin/env python3
# ai_predict_train_tree.py
# Copyright Su Nie | BSD-3C License | https://github.com/can87683

"""
catboost customtkinter lightgbm numpy onnx onnxconverter-common onnxmltools pandas psutil setproctitle xgboost
"""
import os
import sys
import setproctitle
setproctitle.setproctitle(os.path.basename(sys.argv[0]))

import catboost
import configparser
import customtkinter as ctk
from customtkinter import CTkLabel
import datetime
import gc
import json
import lightgbm
import numpy as np
import onnx
import onnxconverter_common
import onnxmltools
import pandas as pd
import pathlib
import psutil
import pickle
import queue
import re
import subprocess
import threading
import tkinter as tk
import tkinter.filedialog as filedialog
import tkinter.messagebox as messagebox
import traceback
import types
import xgboost


class Config:
    HOME = pathlib.Path.home()
    MODEL_DIR = HOME / "ai_model_ml_symbol"
    DATASET_DIR = HOME / "mt5_data"

    REQUIRED_COLUMNS = ['time', 'open', 'high', 'low', 'close', 'tick_volume']
    SYMBOL_PATTERN = r"[A-Za-z0-9._#+-]+"
    MAX_ROWS_PER_SYMBOL = 20000
    WARMUP_BARS = 300
    MIN_ROWS_AFTER_WARMUP = 100
    FEATURE_VERSION = 2
    VALIDATION_FRACTION = 0.15

    DEFAULT_HORIZON = 10
    DEFAULT_ITERATIONS = 1000
    DEFAULT_LEARNING_RATE = 0.05
    DEFAULT_MAX_DEPTH = 6

    TIMEFRAME_HYPERPARAMS = {
        "M5":  {"max_depth": 6, "iterations": 1500, "reg_lambda": 1.0},
        "M15": {"max_depth": 6, "iterations": 1500, "reg_lambda": 1.0},
        "H1":  {"max_depth": 6, "iterations": 1200, "reg_lambda": 2.0},
        "H4":  {"max_depth": 5, "iterations": 1000, "reg_lambda": 3.0},
        "D1":  {"max_depth": 4, "iterations": 600,  "reg_lambda": 4.0},
        "W1":  {"max_depth": 3, "iterations": 300,  "reg_lambda": 6.0},
    }
    TIMEFRAMES = ['M5', 'M15', 'H1', 'H4', 'D1', 'W1']

    APP_TITLE = "Tree ML Trainer"
    THEME = "dark"
    COLOR_THEME = "dark-blue"

    @classmethod
    def ensure_directories(cls):
        cls.MODEL_DIR.mkdir(parents=True, exist_ok=True)
        cls.DATASET_DIR.mkdir(parents=True, exist_ok=True)


class GPUManager:
    def __init__(self, verbose=True, log_callback=None):
        self.verbose = verbose
        self.gpu_available = False
        self.gpu_name = None
        self.gpu_memory_gb = 0.0
        self.log_callback = log_callback or print
        self._detect_gpu()

    def _log(self, message):
        if self.log_callback:
            self.log_callback(message)
        elif self.verbose:
            print(f"[GPUManager] {message}")

    def _detect_gpu(self):
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=3,
        )
        if result.returncode == 0 and result.stdout.strip():
            self.gpu_available = True
            self.gpu_name = result.stdout.strip().splitlines()[0]
            mem_result = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=3
            )
            self.gpu_memory_gb = float(mem_result.stdout.strip().split()[0]) / 1024.0
            self._log(f"✅ GPU detected: {self.gpu_name} ({self.gpu_memory_gb:.2f} GB)")
        else:
            self._log("ℹ️ No GPU detected, using CPU")

    def log_system_info(self):
        self._log("=" * 60)
        self._log("🖥️ System & Library Information")
        self._log("=" * 60)
        self._log(f"🐍 Python: {sys.version.split()[0]}")
        self._log(f"🚀 LightGBM: {lightgbm.__version__}")
        self._log("=" * 60)

    def preferred_devices(self):
        if self.gpu_available:
            return ["gpu", "cpu"]
        return ["cpu"]


class UsageRow:
    FRAME_HEIGHT = 40
    FRAME_CORNER_RADIUS = 5
    FRAME_PADY = (0, 10)

    FONT = ("TkDefaultFont", 20)
    TEXT_COLOR = "Lime"
    PLACEHOLDER_TEXT = "Loading system usage..."

    LABEL_PADX = 10
    LABEL_PADY = 5

    REFRESH_MS = 2000
    CPU_SAMPLE_INTERVAL = 0.5

    def __init__(self, parent):
        self.usage_frame = ctk.CTkFrame(
            parent,
            height=self.FRAME_HEIGHT,
            corner_radius=self.FRAME_CORNER_RADIUS,
        )
        self.usage_frame.pack(fill="x", pady=self.FRAME_PADY)
        self.usage_frame.pack_propagate(False)

        self.usage_label = CTkLabel(
            self.usage_frame,
            text=self.PLACEHOLDER_TEXT,
            font=self.FONT,
            text_color=self.TEXT_COLOR,
        )
        self.usage_label.pack(
            expand=True, fill="x",
            padx=self.LABEL_PADX, pady=self.LABEL_PADY,
        )
        self._update_usage()

    def _update_usage(self):
        cpu = psutil.cpu_percent(interval=self.CPU_SAMPLE_INTERVAL)
        dram = psutil.virtual_memory().percent
        gpu_percent = vram_percent = 0
        try:
            import GPUtil
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu_percent = gpus[0].load * 100
                vram_percent = gpus[0].memoryUtil * 100
        except ImportError:
            pass
        self.usage_label.configure(
            text=f"CPU: {cpu:.1f}%  DRAM: {dram:.1f}%  "
                 f"GPU: {gpu_percent:.1f}%  VRAM: {vram_percent:.1f}%"
        )
        self.usage_label.after(self.REFRESH_MS, self._update_usage)


class ConsoleLogger:
    def __init__(self, gui_logger, original_stream, level):
        self.gui_logger = gui_logger
        self.original_stream = original_stream
        self.level = level

    def write(self, message):
        text = message.strip()
        if text:
            self.gui_logger.log(text, self.level)

    def flush(self):
        self.original_stream.flush()


class MemoryOptimizer:
    @staticmethod
    def get_system_memory_available_gb():
        return psutil.virtual_memory().available / 1e9

    @staticmethod
    def clear_cache():
        gc.collect()


class FeatureEngineer:
    RETURN_LAGS = (1, 2, 3, 5, 10, 20, 60)
    VOL_WINDOWS = (10, 20, 60)
    PARKINSON_WINDOW = 20
    ZSCORE_WINDOW = 20
    VOLUME_WINDOW = 20
    RANGE_WINDOWS = (20, 60)
    ER_WINDOW = 20
    SLOPE_WINDOW = 20
    DIST_WINDOW = 60
    DD_WINDOW = 60
    GK_WINDOW = 20

    ADX_PERIOD = 14
    ATR_PERIOD = 14
    BB_PERIOD = 20
    BB_STD = 2.0
    CCI_PERIOD = 20
    EMA_SPANS = (12, 26, 50, 200)
    KC_PERIOD = 20
    KC_MULT = 2.0
    MACD_FAST = 12
    MACD_SLOW = 26
    MACD_SIGNAL = 9
    MFI_PERIOD = 14
    OBV_WINDOW = 20
    ROC_PERIOD = 12
    RSI_PERIOD = 14
    SMA_WINDOWS = (20, 50, 200)
    STOCH_PERIOD = 14
    STOCH_SMOOTH = 3
    VOLMA_WINDOW = 20
    VWAP_WINDOW = 20
    WILLR_PERIOD = 14

    INDICATORS = {
        'enable_adx': True,
        'enable_atr': True,
        'enable_bb': True,
        'enable_cci': False,
        'enable_ema': True,
        'enable_kc': False,
        'enable_macd': False,
        'enable_mfi': True,
        'enable_obv': True,
        'enable_roc': False,
        'enable_rsi': True,
        'enable_sma': False,
        'enable_stochastic': False,
        'enable_vol_ma': False,
        'enable_vwap': False,
        'enable_willr': False,
    }

    RAW_FEATURES = [c for c in Config.REQUIRED_COLUMNS if c != 'time']
    TIME_FEATURES = ['hour_sin', 'hour_cos', 'day_sin', 'day_cos', 'month_sin', 'month_cos']
    BASE_FEATURES = (
        RAW_FEATURES
        + TIME_FEATURES
        + ['high_low_ratio', 'close_high_ratio', 'close_low_ratio', 'body_ratio', 'upper_wick', 'lower_wick']
        + [f"ret_{n}" for n in RETURN_LAGS]
        + [f"vol_{w}" for w in VOL_WINDOWS]
        + ['vol_ratio', 'parkinson', 'ret_z', 'tick_volume_ratio', 'tick_volume_chg']
        + ['gap', 'garman_klass', f"er_{ER_WINDOW}", f"slope_atr_{SLOPE_WINDOW}", f"skew_{DIST_WINDOW}", f"kurt_{DIST_WINDOW}", f"drawdown_{DD_WINDOW}"]
        + [f"range_pos_{w}" for w in RANGE_WINDOWS]
        + [f"since_high_{w}" for w in RANGE_WINDOWS]
        + [f"since_low_{w}" for w in RANGE_WINDOWS]
    )

    FLAG_TO_FEATURES = {
        'enable_adx': ['adx', 'plus_di', 'minus_di'],
        'enable_atr': ['atr_pct'],
        'enable_bb': ['bb_percent', 'bb_width'],
        'enable_cci': ['cci'],
        'enable_ema': [f"ema_{s}_rel" for s in EMA_SPANS],
        'enable_kc': ['kc_percent', 'kc_width'],
        'enable_macd': ['macd', 'macd_signal', 'macd_hist'],
        'enable_mfi': ['mfi'],
        'enable_obv': ['obv_rel'],
        'enable_roc': ['roc'],
        'enable_rsi': ['rsi'],
        'enable_sma': [f"sma_{s}_rel" for s in SMA_WINDOWS],
        'enable_stochastic': ['stoch_k', 'stoch_d'],
        'enable_vol_ma': [f"volume_ema_{VOLMA_WINDOW}_rel", f"volume_sma_{VOLMA_WINDOW}_rel"],
        'enable_vwap': ['vwap_rel'],
        'enable_willr': ['willr'],
    }

    @staticmethod
    def feature_names():
        names = list(FeatureEngineer.BASE_FEATURES)
        for flag, cols in FeatureEngineer.FLAG_TO_FEATURES.items():
            if FeatureEngineer.INDICATORS[flag]:
                names.extend(cols)
        return names

    @classmethod
    def add_features(cls, df):
        df = df.copy()
        t = df['time']
        o, h, l, c, v = df['open'], df['high'], df['low'], df['close'], df['tick_volume']

        hour = t.dt.hour + t.dt.minute / 60
        df['hour_sin'] = np.sin(2 * np.pi * hour / 24)
        df['hour_cos'] = np.cos(2 * np.pi * hour / 24)
        df['day_sin'] = np.sin(2 * np.pi * t.dt.dayofweek / 7)
        df['day_cos'] = np.cos(2 * np.pi * t.dt.dayofweek / 7)
        df['month_sin'] = np.sin(2 * np.pi * t.dt.month / 12)
        df['month_cos'] = np.cos(2 * np.pi * t.dt.month / 12)

        rng = (h - l).replace(0, np.nan)
        df['high_low_ratio'] = h / l
        df['close_high_ratio'] = c / h
        df['close_low_ratio'] = c / l
        df['body_ratio'] = (c - o) / rng
        df['upper_wick'] = (h - np.maximum(o, c)) / rng
        df['lower_wick'] = (np.minimum(o, c) - l) / rng

        log_close = np.log(c)
        ret1 = log_close.diff()

        for n in cls.RETURN_LAGS:
            df[f"ret_{n}"] = log_close.diff(n)

        for w in cls.VOL_WINDOWS:
            df[f"vol_{w}"] = ret1.rolling(w).std()

        df['vol_ratio'] = (
            df[f"vol_{cls.VOL_WINDOWS[0]}"]
            / df[f"vol_{cls.VOL_WINDOWS[-1]}"].replace(0, np.nan)
        )
        df['parkinson'] = np.sqrt(
            (np.log(h / l) ** 2).rolling(cls.PARKINSON_WINDOW).mean()
            / (4 * np.log(2))
        )
        df['ret_z'] = (
            (ret1 - ret1.rolling(cls.ZSCORE_WINDOW).mean())
            / ret1.rolling(cls.ZSCORE_WINDOW).std().replace(0, np.nan)
        )
        df['tick_volume_ratio'] = v / v.rolling(cls.VOLUME_WINDOW).mean().replace(0, np.nan)
        df['tick_volume_chg'] = np.log(v.replace(0, np.nan)).diff()

        typical = (h + l + c) / 3
        prev_close = c.shift(1)
        true_range = pd.concat(
            [h - l, (h - prev_close).abs(), (l - prev_close).abs()],
            axis=1,
        ).max(axis=1)

        df['gap'] = o / prev_close - 1
        gk = (
            0.5 * np.log(h / l) ** 2
            - (2 * np.log(2) - 1) * np.log(c / o) ** 2
        )
        df['garman_klass'] = np.sqrt(
            gk.rolling(cls.GK_WINDOW).mean().clip(lower=0)
        )
        df[f"er_{cls.ER_WINDOW}"] = (
            (c - c.shift(cls.ER_WINDOW)).abs()
            / c.diff().abs().rolling(cls.ER_WINDOW).sum().replace(0, np.nan)
        )

        slope_x = np.arange(cls.SLOPE_WINDOW) - (cls.SLOPE_WINDOW - 1) / 2
        slope = c.rolling(cls.SLOPE_WINDOW).apply(
            lambda y: np.dot(slope_x, y) / np.dot(slope_x, slope_x),
            raw=True,
        )
        atr_slope = true_range.ewm(
            alpha=1 / cls.ATR_PERIOD,
            min_periods=1,
            adjust=False,
        ).mean()
        df[f"slope_atr_{cls.SLOPE_WINDOW}"] = slope / atr_slope.replace(0, np.nan)

        df[f"skew_{cls.DIST_WINDOW}"] = ret1.rolling(cls.DIST_WINDOW).skew()
        df[f"kurt_{cls.DIST_WINDOW}"] = ret1.rolling(cls.DIST_WINDOW).kurt()
        df[f"drawdown_{cls.DD_WINDOW}"] = c / c.rolling(cls.DD_WINDOW).max() - 1

        for w in cls.RANGE_WINDOWS:
            lowest = l.rolling(w).min()
            highest = h.rolling(w).max()
            df[f"range_pos_{w}"] = (
                (c - lowest) / (highest - lowest).replace(0, np.nan)
            )
            df[f"since_high_{w}"] = h.rolling(w).apply(
                lambda x: np.argmax(x[::-1]) / len(x),
                raw=True,
            )
            df[f"since_low_{w}"] = l.rolling(w).apply(
                lambda x: np.argmin(x[::-1]) / len(x),
                raw=True,
            )

        if cls.INDICATORS['enable_adx']:
            a = 1 / cls.ADX_PERIOD
            up = h.diff()
            down = -l.diff()
            plus_dm = up.where((up > down) & (up > 0), 0.0)
            minus_dm = down.where((down > up) & (down > 0), 0.0)
            atr = true_range.ewm(alpha=a, min_periods=1, adjust=False).mean().replace(0, np.nan)
            plus_di = 100 * plus_dm.ewm(alpha=a, min_periods=1, adjust=False).mean() / atr
            minus_di = 100 * minus_dm.ewm(alpha=a, min_periods=1, adjust=False).mean() / atr
            dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
            df['plus_di'] = plus_di
            df['minus_di'] = minus_di
            df['adx'] = dx.ewm(alpha=a, min_periods=1, adjust=False).mean()

        if cls.INDICATORS['enable_atr']:
            df['atr_pct'] = (
                true_range.ewm(
                    alpha=1 / cls.ATR_PERIOD,
                    min_periods=1,
                    adjust=False,
                ).mean() / c
            )

        if cls.INDICATORS['enable_bb']:
            middle = c.rolling(cls.BB_PERIOD, min_periods=1).mean()
            std = c.rolling(cls.BB_PERIOD, min_periods=1).std()
            upper = middle + cls.BB_STD * std
            lower = middle - cls.BB_STD * std
            df['bb_width'] = (upper - lower) / middle.replace(0, np.nan)
            df['bb_percent'] = (c - lower) / (upper - lower).replace(0, np.nan)

        if cls.INDICATORS['enable_cci']:
            sma_typ = typical.rolling(cls.CCI_PERIOD, min_periods=1).mean()
            mean_dev = typical.rolling(cls.CCI_PERIOD, min_periods=1).apply(
                lambda x: np.mean(np.abs(x - np.mean(x))),
                raw=True,
            )
            df['cci'] = (typical - sma_typ) / (0.015 * mean_dev.replace(0, np.nan))

        if cls.INDICATORS['enable_ema']:
            for s in cls.EMA_SPANS:
                df[f"ema_{s}_rel"] = (
                    c / c.ewm(span=s, min_periods=1, adjust=False).mean() - 1
                )

        if cls.INDICATORS['enable_kc']:
            middle = typical.rolling(cls.KC_PERIOD, min_periods=1).mean()
            atr = true_range.ewm(
                alpha=1 / cls.KC_PERIOD,
                min_periods=1,
                adjust=False,
            ).mean()
            upper = middle + cls.KC_MULT * atr
            lower = middle - cls.KC_MULT * atr
            df['kc_width'] = (upper - lower) / middle.replace(0, np.nan)
            df['kc_percent'] = (c - lower) / (upper - lower).replace(0, np.nan)

        if cls.INDICATORS['enable_macd']:
            macd = (
                c.ewm(span=cls.MACD_FAST, adjust=False).mean()
                - c.ewm(span=cls.MACD_SLOW, adjust=False).mean()
            )
            signal = macd.ewm(span=cls.MACD_SIGNAL, adjust=False).mean()
            df['macd'] = macd / c
            df['macd_signal'] = signal / c
            df['macd_hist'] = (macd - signal) / c

        if cls.INDICATORS['enable_mfi']:
            raw_money = typical * v
            diff = typical.diff()
            positive = raw_money.where(diff > 0, 0.0).rolling(
                cls.MFI_PERIOD,
                min_periods=1,
            ).sum()
            negative = raw_money.where(diff < 0, 0.0).rolling(
                cls.MFI_PERIOD,
                min_periods=1,
            ).sum()

            money_ratio = positive / negative.replace(0, np.nan)
            mfi = 100 - 100 / (1 + money_ratio)
            mfi = mfi.mask((negative == 0) & (positive > 0), 100.0)
            mfi = mfi.mask((positive == 0) & (negative > 0), 0.0)
            mfi = mfi.mask((positive == 0) & (negative == 0), 50.0)
            df['mfi'] = mfi

        if cls.INDICATORS['enable_obv']:
            signed = np.sign(c.diff()).fillna(0.0) * v
            df['obv_rel'] = (
                signed.rolling(cls.OBV_WINDOW).sum()
                / v.rolling(cls.OBV_WINDOW).sum().replace(0, np.nan)
            )

        if cls.INDICATORS['enable_roc']:
            df['roc'] = (
                (c - c.shift(cls.ROC_PERIOD))
                / c.shift(cls.ROC_PERIOD).replace(0, np.nan)
            ) * 100

        if cls.INDICATORS['enable_rsi']:
            delta = c.diff()
            avg_gain = delta.clip(lower=0).ewm(
                alpha=1 / cls.RSI_PERIOD,
                min_periods=1,
                adjust=False,
            ).mean()
            avg_loss = (-delta).clip(lower=0).ewm(
                alpha=1 / cls.RSI_PERIOD,
                min_periods=1,
                adjust=False,
            ).mean()

            relative_strength = avg_gain / avg_loss.replace(0, np.nan)
            rsi = 100 - 100 / (1 + relative_strength)
            rsi = rsi.mask((avg_loss == 0) & (avg_gain > 0), 100.0)
            rsi = rsi.mask((avg_gain == 0) & (avg_loss > 0), 0.0)
            rsi = rsi.mask((avg_gain == 0) & (avg_loss == 0), 50.0)
            df['rsi'] = rsi

        if cls.INDICATORS['enable_sma']:
            for s in cls.SMA_WINDOWS:
                df[f"sma_{s}_rel"] = c / c.rolling(s, min_periods=1).mean() - 1

        if cls.INDICATORS['enable_stochastic']:
            lowest = l.rolling(cls.STOCH_PERIOD, min_periods=1).min()
            highest = h.rolling(cls.STOCH_PERIOD, min_periods=1).max()
            k = 100 * (c - lowest) / (highest - lowest).replace(0, np.nan)
            df['stoch_k'] = k
            df['stoch_d'] = k.rolling(cls.STOCH_SMOOTH, min_periods=1).mean()

        if cls.INDICATORS['enable_vol_ma']:
            df[f"volume_ema_{cls.VOLMA_WINDOW}_rel"] = (
                v / v.ewm(
                    span=cls.VOLMA_WINDOW,
                    min_periods=1,
                    adjust=False,
                ).mean().replace(0, np.nan) - 1
            )
            df[f"volume_sma_{cls.VOLMA_WINDOW}_rel"] = (
                v / v.rolling(cls.VOLMA_WINDOW, min_periods=1).mean().replace(0, np.nan) - 1
            )

        if cls.INDICATORS['enable_vwap']:
            vwap = (
                (typical * v).rolling(cls.VWAP_WINDOW, min_periods=1).sum()
                / v.rolling(cls.VWAP_WINDOW, min_periods=1).sum().replace(0, np.nan)
            )
            df['vwap_rel'] = c / vwap - 1

        if cls.INDICATORS['enable_willr']:
            lowest = l.rolling(cls.WILLR_PERIOD, min_periods=1).min()
            highest = h.rolling(cls.WILLR_PERIOD, min_periods=1).max()
            df['willr'] = (
                -100 * (highest - c) / (highest - lowest).replace(0, np.nan)
            )

        return df


class GUILogger:
    MAX_LOG_LINES = 2000
    ERROR_COLOR = "yellow"

    def __init__(self):
        self.messages = queue.Queue()
        self.log_widget = None
        self.ready = False
        self._original_stdout = sys.stdout
        self._original_stderr = sys.stderr

    def set_log_widget(self, widget):
        self.log_widget = widget
        self.log_widget.tag_config("error", foreground=self.ERROR_COLOR)
        self.ready = True

    def _insert(self, entry, tag):
        self.log_widget.insert("end", entry, tag)
        lines = int(self.log_widget.index("end-1c").split(".")[0])

        if lines > self.MAX_LOG_LINES:
            self.log_widget.delete(
                "1.0",
                f"{lines - self.MAX_LOG_LINES + 1}.0",
            )

        self.log_widget.see("end")

    def log(self, message, level="info"):
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        self._original_stdout.write(
            f"[{timestamp}] [{level.upper()}] {message}\n"
        )
        self._original_stdout.flush()

        entry = f"[{timestamp}] {message}\n"
        tag = "error" if level == "error" else None
        self.messages.put((entry, tag))

    def drain(self):
        while not self.messages.empty():
            entry, tag = self.messages.get_nowait()
            self._insert(entry, tag)

    def enable_console_redirect(self):
        sys.stdout = ConsoleLogger(
            self,
            self._original_stdout,
            "console",
        )
        sys.stderr = ConsoleLogger(
            self,
            self._original_stderr,
            "error",
        )

    def close(self):
        self.ready = False
        sys.stdout = self._original_stdout
        sys.stderr = self._original_stderr


class StreamingDataPreparer:
    def __init__(self, log_callback):
        self.log_callback = log_callback
        self.feature_cols = []

    def _log(self, message):
        self.log_callback(message)

    def load_csv_data(self, symbol, timeframe, csv_folder):
        csv_path = pathlib.Path(csv_folder) / symbol / f"{symbol}_{timeframe}.csv"
        df = pd.read_csv(csv_path)

        missing = [c for c in Config.REQUIRED_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(
                f"{csv_path.name} is missing required columns: {missing}"
            )

        original_count = len(df)

        if pd.api.types.is_numeric_dtype(df["time"]):
            df["time"] = pd.to_datetime(
                df["time"],
                unit="s",
                errors="coerce",
            )
        else:
            df["time"] = pd.to_datetime(
                df["time"],
                errors="coerce",
            )

        numeric_columns = [
            "open",
            "high",
            "low",
            "close",
            "tick_volume",
        ]

        for column in numeric_columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

        price_columns = ["open", "high", "low", "close"]

        numeric_values = df[numeric_columns].to_numpy(dtype=np.float64)
        finite_mask = np.isfinite(numeric_values).all(axis=1)

        positive_prices = (df[price_columns] > 0).all(axis=1)
        valid_volume = df["tick_volume"] >= 0

        valid_range = (
            (df["high"] >= df["low"])
            & (df["high"] >= df["open"])
            & (df["high"] >= df["close"])
            & (df["low"] <= df["open"])
            & (df["low"] <= df["close"])
        )

        valid_time = df["time"].notna()

        valid_rows = (
            finite_mask
            & positive_prices.to_numpy()
            & valid_volume.to_numpy()
            & valid_range.to_numpy()
            & valid_time.to_numpy()
        )

        invalid_count = int((~valid_rows).sum())

        if invalid_count:
            invalid_indices = df.index[~valid_rows].tolist()[:10]
            df = df.loc[valid_rows].copy()

            self._log(
                f"⚠️ {csv_path.name}: discarded {invalid_count:,} invalid rows "
                f"({invalid_indices} shown)"
            )

        if df.empty:
            raise ValueError(
                f"{csv_path.name} contains no valid OHLCV rows after cleaning"
            )

        df = df.sort_values("time")

        duplicate_count = int(df["time"].duplicated(keep="last").sum())

        if duplicate_count:
            df = df.drop_duplicates("time", keep="last")
            self._log(
                f"⚠️ {csv_path.name}: removed {duplicate_count:,} duplicate timestamps"
            )

        df = df.reset_index(drop=True)

        removed_count = original_count - len(df)

        if removed_count:
            self._log(
                f"🧹 {csv_path.name}: retained {len(df):,}/{original_count:,} "
                f"valid rows ({removed_count:,} removed)"
            )

        if len(df) < Config.WARMUP_BARS + Config.MIN_ROWS_AFTER_WARMUP:
            self._log(
                f"⚠️ {csv_path.name}: only {len(df):,} valid rows available"
            )

        return df

    def scan_folder(self, csv_folder):
        symbols = []
        all_timeframes = set()

        for symbol_dir in sorted(pathlib.Path(csv_folder).iterdir()):
            if not symbol_dir.is_dir():
                continue

            symbol = symbol_dir.name
            found = [
                tf for tf in Config.TIMEFRAMES
                if (symbol_dir / f"{symbol}_{tf}.csv").is_file()
            ]

            if not found:
                continue

            if not re.fullmatch(Config.SYMBOL_PATTERN, symbol):
                raise ValueError(
                    f"Unsafe symbol directory name: {symbol!r}"
                )

            symbols.append((symbol, found))
            all_timeframes.update(found)
            self._log(f"   ✓ {symbol}: {', '.join(found)}")

        return symbols, sorted(all_timeframes)

    def build_dataframe_for_symbol(self, csv_folder, symbol, timeframe, horizon):
        if horizon < 1:
            raise ValueError("Prediction horizon must be at least 1 bar")

        df = self.load_csv_data(symbol, timeframe, csv_folder)

        if len(df) < Config.WARMUP_BARS + horizon + Config.MIN_ROWS_AFTER_WARMUP:
            self._log(
                f"⚠️ {symbol}_{timeframe}: insufficient valid rows after cleaning"
            )
            return None

        df = df.iloc[-Config.MAX_ROWS_PER_SYMBOL:].copy()
        df = FeatureEngineer.add_features(df)

        features = FeatureEngineer.feature_names()

        df[features] = (
            df[features]
            .replace([np.inf, -np.inf], np.nan)
            .ffill()
            .fillna(0.0)
        )

        df["target"] = df["close"].shift(-horizon)
        df = df.iloc[Config.WARMUP_BARS:].dropna(subset=["target"])

        if df.empty:
            self._log(
                f"⚠️ {symbol}_{timeframe}: no training rows remain after feature preparation"
            )
            return None

        self.feature_cols = features

        self._log(
            f"📊 {symbol}_{timeframe}: {len(df):,} rows, "
            f"{len(features)} features"
        )

        return df[["time", "target"] + features].reset_index(drop=True)


class StreamingLightGBMTrainer:
    MODEL_PREFIX = "ai_predict_lightgbm"
    MODEL_EXT = ".lgb.txt"
    ONNX_EXT = ".onnx"
    META_EXT = ".meta.txt"
    INFO_EXT = ".info.json"
    ONNX_OPSET = 14
    LOG_EVERY_ROUNDS = 10
    TOP_IMPORTANCE = 10
    EARLY_STOPPING_ROUNDS = 100
    MIN_CHILD_SAMPLES = 100
    NUM_LEAVES = 63
    SUBSAMPLE = 0.8
    SUBSAMPLE_FREQ = 1
    COLSAMPLE_BYTREE = 0.8
    REG_LAMBDA = 1.0
    RANDOM_STATE = 42

    def __init__(self, log_callback, symbol, timeframe):
        self.log_callback = log_callback
        self.symbol = symbol
        self.timeframe = timeframe
        self.model = None
        self.is_trained = False
        self.stop_flag = False
        self.training_history = {}
        self.importance = {}
        self.parity_row = None
        self._log(f"🚀 LightGBM trainer [{symbol} {timeframe}] initialized (CPU)")

    def _log(self, message):
        self.log_callback(message)

    def stop_training(self):
        self.stop_flag = True
        self._log("⏹️ Training stop requested")

    def train_single_timeframe(self, data, feature_cols, iterations, learning_rate, max_depth, horizon, status_callback):
        self.stop_flag = False
        tag = f"{self.symbol} {self.timeframe}"

        hp = Config.TIMEFRAME_HYPERPARAMS[self.timeframe]
        n_estimators = iterations or hp["iterations"]
        depth = max_depth or hp["max_depth"]
        reg_lambda = self.REG_LAMBDA if iterations else hp["reg_lambda"]

        self.parity_row = data.iloc[-1][["time"] + feature_cols]

        n_val = max(1, int(len(data) * Config.VALIDATION_FRACTION))
        train_df = data.iloc[:len(data) - n_val - horizon]
        val_df = data.iloc[len(data) - n_val:]

        X_train = train_df[feature_cols].to_numpy(dtype=np.float32)
        y_train = (train_df["target"] / train_df["close"] - 1.0).to_numpy(dtype=np.float32)
        X_val = val_df[feature_cols].to_numpy(dtype=np.float32)
        y_val = (val_df["target"] / val_df["close"] - 1.0).to_numpy(dtype=np.float32)
        del train_df, val_df

        self._log(f"[{tag}] Train rows: {len(y_train):,}, Val rows: {len(y_val):,}, Embargo: {horizon} bars")
        self._log(f"[{tag}] Model features ({len(feature_cols)}): {', '.join(feature_cols)}")
        self._log(f"[{tag}] Trees: {n_estimators}, depth: {depth}, lr: {learning_rate}, L2: {reg_lambda}")

        def progress(env):
            if self.stop_flag:
                raise lightgbm.callback.EarlyStopException(env.iteration, env.evaluation_result_list)
            if (env.iteration + 1) % self.LOG_EVERY_ROUNDS == 0:
                scores = {name: value for name, _, value, _ in env.evaluation_result_list}
                status_callback(f"[{tag}] Round {env.iteration + 1}/{n_estimators}  train={scores['train']:.6g}  val={scores['val']:.6g}")
        progress.order = 30

        self.model = lightgbm.LGBMRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=depth,
            num_leaves=self.NUM_LEAVES,
            min_child_samples=self.MIN_CHILD_SAMPLES,
            subsample=self.SUBSAMPLE,
            subsample_freq=self.SUBSAMPLE_FREQ,
            colsample_bytree=self.COLSAMPLE_BYTREE,
            reg_lambda=reg_lambda,
            random_state=self.RANDOM_STATE,
            verbose=-1,
        )
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_train, y_train), (X_val, y_val)],
            eval_names=["train", "val"],
            eval_metric="l2",
            callbacks=[lightgbm.early_stopping(self.EARLY_STOPPING_ROUNDS, verbose=False), progress],
        )
        self.is_trained = True

        gain = self.model.booster_.feature_importance(importance_type="gain")
        gain = 100 * gain / max(gain.sum(), 1e-12)
        self.importance = dict(zip(feature_cols, gain.round(4).tolist()))

        predicted = self.model.predict(X_val)
        valid_mask = y_val != 0
        self.training_history.update({
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "best_rounds": int(self.model.best_iteration_),
            "rounds_run": int(self.model.booster_.current_iteration()),
            "val_l2": float(np.mean((predicted - y_val) ** 2)),
            "directional_accuracy": float((np.sign(predicted[valid_mask]) == np.sign(y_val[valid_mask])).mean()),
        })
        top = sorted(self.importance.items(), key=lambda item: -item[1])[:self.TOP_IMPORTANCE]
        self._log(f"[{tag}] Top gain %: " + ", ".join(f"{k}={v:.1f}" for k, v in top))
        self._log(f"[{tag}] Zero-importance features: {sum(1 for v in self.importance.values() if v == 0)}")
        self._log(f"[{tag}] Directional accuracy: {self.training_history['directional_accuracy'] * 100:.1f}%")
        return self.is_trained

    def save_model(self, model_dir, filename_stem, feature_cols, horizon, export_onnx):
        output_dir = pathlib.Path(model_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        model_path = output_dir / f"{filename_stem}{self.MODEL_EXT}"
        onnx_path = output_dir / f"{filename_stem}{self.ONNX_EXT}"
        pkl_path = output_dir / f"{filename_stem}.pkl"

        self.model.booster_.save_model(str(model_path))

        onnx_input = None
        onnx_output = None

        if export_onnx:
            onnx_model = onnxmltools.convert_lightgbm(
                self.model,
                initial_types=[
                    (
                        "input",
                        onnxconverter_common.data_types.FloatTensorType(
                            [1, len(feature_cols)]
                        ),
                    )
                ],
                target_opset=self.ONNX_OPSET,
            )

            for key, value in (
                ("features", ",".join(feature_cols)),
                ("feature_version", str(Config.FEATURE_VERSION)),
            ):
                entry = onnx_model.metadata_props.add()
                entry.key = key
                entry.value = value

            onnx_path.write_bytes(onnx_model.SerializeToString())
            onnx_input = onnx_model.graph.input[0].name
            onnx_output = onnx_model.graph.output[0].name
            del onnx_model

        artifact = {
            "model_type": "lightgbm.LGBMRegressor",
            "target_type": "forward_return",
            "feature_version": Config.FEATURE_VERSION,
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "horizon": horizon,
            "warmup_bars": Config.WARMUP_BARS,
            "feature_columns": list(feature_cols),
            "feature_importance": self.importance,
            "model_path": str(model_path),
            "onnx_path": str(onnx_path) if export_onnx else None,
            "training_history": self.training_history,
            "training_time": datetime.datetime.now().isoformat(),
            "dtype": "float32",
            "fill": "inf_to_nan,ffill,zero",
            "target": "close[t+horizon]/close[t]-1",
            "time": "bar_open_csv_timezone",
            "onnx_input": onnx_input,
            "onnx_output": onnx_output,
            "parity_timestamp": self.parity_row["time"].value // 10**9,
            "parity_features": {
                column: float(self.parity_row[column])
                for column in feature_cols
            },
        }

        with open(pkl_path, "wb") as file:
            pickle.dump(artifact, file, protocol=pickle.HIGHEST_PROTOCOL)

        self._log(f"✅ LightGBM model saved: {model_path}")

        if export_onnx:
            self._log(f"✅ ONNX model saved: {onnx_path}")

        self._log(f"✅ PKL metadata saved: {pkl_path}")


class StreamingXGBoostTrainer:
    MODEL_PREFIX = "ai_predict_xgboost"
    MODEL_EXT = ".xgb.json"
    ONNX_EXT = ".onnx"
    META_EXT = ".meta.txt"
    INFO_EXT = ".info.json"
    ONNX_OPSET = 14
    LOG_EVERY_ROUNDS = 10
    TOP_IMPORTANCE = 10
    EARLY_STOPPING_ROUNDS = 100
    MIN_CHILD_WEIGHT = 10
    SUBSAMPLE = 0.8
    COLSAMPLE_BYTREE = 0.8
    REG_LAMBDA = 1.0
    RANDOM_STATE = 42

    def __init__(self, log_callback, symbol, timeframe):
        self.log_callback = log_callback
        self.symbol = symbol
        self.timeframe = timeframe
        self.model = None
        self.is_trained = False
        self.stop_flag = False
        self.training_history = {}
        self.importance = {}
        self.parity_row = None
        self._log(f"Tree ML trainer [{symbol} {timeframe}] initialized")

    def _log(self, message):
        self.log_callback(message)

    def stop_training(self):
        self.stop_flag = True
        self._log("Training stop requested")

    def train_single_timeframe(
        self,
        data,
        feature_cols,
        iterations,
        learning_rate,
        max_depth,
        horizon,
        status_callback,
    ):
        self.stop_flag = False
        self.is_trained = False
        tag = f"{self.symbol} {self.timeframe}"

        if self.timeframe not in Config.TIMEFRAME_HYPERPARAMS:
            raise ValueError(f"Unsupported timeframe: {self.timeframe}")

        if horizon < 1:
            raise ValueError("Prediction horizon must be at least 1 bar")

        if not feature_cols:
            raise ValueError("No training features were supplied")

        if len(data) <= horizon + 1:
            raise ValueError("Insufficient rows for training and validation")

        hp = Config.TIMEFRAME_HYPERPARAMS[self.timeframe]
        n_estimators = iterations or hp["iterations"]
        depth = max_depth or hp["max_depth"]
        reg_lambda = self.REG_LAMBDA if iterations else hp["reg_lambda"]

        if n_estimators < 1 or depth < 1 or learning_rate <= 0:
            raise ValueError("Invalid XGBoost training parameters")

        self.parity_row = data.iloc[-1][["time"] + feature_cols].copy()

        n_val = max(1, int(len(data) * Config.VALIDATION_FRACTION))
        train_end = len(data) - n_val - horizon

        if train_end < 2 or n_val < 1:
            raise ValueError(
                f"Insufficient rows after validation split: "
                f"total={len(data)}, validation={n_val}, embargo={horizon}"
            )

        train_df = data.iloc[:train_end]
        val_df = data.iloc[len(data) - n_val:]

        X_train = train_df[feature_cols].to_numpy(dtype=np.float32)
        y_train = (
            train_df["target"] / train_df["close"] - 1.0
        ).to_numpy(dtype=np.float32)

        X_val = val_df[feature_cols].to_numpy(dtype=np.float32)
        y_val = (
            val_df["target"] / val_df["close"] - 1.0
        ).to_numpy(dtype=np.float32)

        if not np.isfinite(X_train).all() or not np.isfinite(y_train).all():
            raise ValueError("Training data contains NaN or infinite values")

        if not np.isfinite(X_val).all() or not np.isfinite(y_val).all():
            raise ValueError("Validation data contains NaN or infinite values")

        del train_df, val_df

        self._log(
            f"[{tag}] Train rows: {len(y_train):,}, "
            f"Val rows: {len(y_val):,}, Embargo: {horizon} bars"
        )
        self._log(
            f"[{tag}] Model features ({len(feature_cols)}): "
            f"{', '.join(feature_cols)}"
        )
        self._log(
            f"[{tag}] Trees: {n_estimators}, depth: {depth}, "
            f"lr: {learning_rate}, L2: {reg_lambda}"
        )

        trainer = self

        class Progress(xgboost.callback.TrainingCallback):
            def after_iteration(self, booster, epoch, evals_log):
                if (epoch + 1) % trainer.LOG_EVERY_ROUNDS == 0:
                    train_rmse = evals_log["validation_0"]["rmse"][-1]
                    val_rmse = evals_log["validation_1"]["rmse"][-1]
                    status_callback(
                        f"[{tag}] Round {epoch + 1}/{n_estimators}  "
                        f"train={train_rmse:.6g}  val={val_rmse:.6g}"
                    )

                return trainer.stop_flag

        self.model = xgboost.XGBRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=depth,
            min_child_weight=self.MIN_CHILD_WEIGHT,
            subsample=self.SUBSAMPLE,
            colsample_bytree=self.COLSAMPLE_BYTREE,
            reg_lambda=reg_lambda,
            tree_method="hist",
            eval_metric="rmse",
            early_stopping_rounds=self.EARLY_STOPPING_ROUNDS,
            callbacks=[Progress()],
            random_state=self.RANDOM_STATE,
            verbosity=0,
        )

        self.model.fit(
            X_train,
            y_train,
            eval_set=[(X_train, y_train), (X_val, y_val)],
            verbose=False,
        )

        if self.stop_flag:
            self._log(f"[{tag}] Training stopped")
            return False

        self.is_trained = True

        gain = self.model.feature_importances_.astype(np.float64)
        gain_total = gain.sum()

        if gain_total > 0:
            gain = 100 * gain / gain_total

        self.importance = dict(
            zip(feature_cols, gain.round(4).tolist())
        )

        predicted = self.model.predict(X_val)
        valid_mask = y_val != 0

        directional_accuracy = (
            float(
                (
                    np.sign(predicted[valid_mask])
                    == np.sign(y_val[valid_mask])
                ).mean()
            )
            if valid_mask.any()
            else 0.0
        )

        self.training_history = {
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "best_rounds": int(self.model.best_iteration) + 1,
            "rounds_run": int(
                self.model.get_booster().num_boosted_rounds()
            ),
            "val_l2": float(np.mean((predicted - y_val) ** 2)),
            "directional_accuracy": directional_accuracy,
        }

        top = sorted(
            self.importance.items(),
            key=lambda item: -item[1],
        )[:self.TOP_IMPORTANCE]

        self._log(
            f"[{tag}] Top gain %: "
            + ", ".join(f"{key}={value:.1f}" for key, value in top)
        )
        self._log(
            f"[{tag}] Zero-importance features: "
            f"{sum(1 for value in self.importance.values() if value == 0)}"
        )
        self._log(
            f"[{tag}] Directional accuracy: "
            f"{directional_accuracy * 100:.1f}%"
        )

        return self.is_trained

    def save_model(
        self,
        model_dir,
        filename_stem,
        feature_cols,
        horizon,
        export_onnx,
    ):
        if not self.is_trained or self.model is None:
            raise RuntimeError("Cannot save an untrained XGBoost model")

        if not self.training_history:
            raise RuntimeError("Training history is missing")

        best_rounds = self.training_history["best_rounds"]

        if best_rounds < 1:
            raise RuntimeError("XGBoost best iteration is invalid")

        output_dir = pathlib.Path(model_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        model_path = output_dir / f"{filename_stem}{self.MODEL_EXT}"
        onnx_path = output_dir / f"{filename_stem}{self.ONNX_EXT}"
        pkl_path = output_dir / f"{filename_stem}.pkl"

        best_booster = self.model.get_booster()[:best_rounds]
        best_booster.save_model(str(model_path))

        onnx_input = None
        onnx_output = None

        if export_onnx:
            onnx_model = onnxmltools.convert_xgboost(
                best_booster,
                initial_types=[
                    (
                        "input",
                        onnxconverter_common.data_types.FloatTensorType(
                            [1, len(feature_cols)]
                        ),
                    )
                ],
                target_opset=self.ONNX_OPSET,
            )

            for key, value in (
                ("features", ",".join(feature_cols)),
                ("feature_version", str(Config.FEATURE_VERSION)),
                ("best_rounds", str(best_rounds)),
            ):
                entry = onnx_model.metadata_props.add()
                entry.key = key
                entry.value = value

            onnx.checker.check_model(onnx_model)
            onnx_path.write_bytes(onnx_model.SerializeToString())

            onnx_input = onnx_model.graph.input[0].name
            onnx_output = onnx_model.graph.output[0].name

            del onnx_model

        artifact = {
            "model_type": "xgboost.XGBRegressor",
            "target_type": "forward_return",
            "feature_version": Config.FEATURE_VERSION,
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "horizon": horizon,
            "warmup_bars": Config.WARMUP_BARS,
            "feature_columns": list(feature_cols),
            "feature_importance": self.importance,
            "model_path": str(model_path),
            "onnx_path": str(onnx_path) if export_onnx else None,
            "training_history": self.training_history,
            "training_time": datetime.datetime.now().isoformat(),
            "best_rounds": best_rounds,
            "dtype": "float32",
            "fill": "inf_to_nan,ffill,zero",
            "target": "close[t+horizon]/close[t]-1",
            "time": "bar_open_csv_timezone",
            "onnx_input": onnx_input,
            "onnx_output": onnx_output,
            "parity_timestamp": self.parity_row["time"].value // 10**9,
            "parity_features": {
                column: float(self.parity_row[column])
                for column in feature_cols
            },
        }

        with open(pkl_path, "wb") as file:
            pickle.dump(artifact, file, protocol=pickle.HIGHEST_PROTOCOL)

        self._log(f"✅ XGBoost model saved: {model_path}")

        if export_onnx:
            self._log(f"✅ ONNX model saved: {onnx_path}")

        self._log(f"✅ PKL metadata saved: {pkl_path}")


class StreamingCatBoostTrainer:
    MODEL_PREFIX = "ai_predict_catboost"
    MODEL_EXT = ".cbm"
    ONNX_EXT = ".onnx"
    META_EXT = ".meta.txt"
    INFO_EXT = ".info.json"
    LOG_EVERY_ROUNDS = 10
    TOP_IMPORTANCE = 10
    EARLY_STOPPING_ROUNDS = 100
    RANDOM_STATE = 42

    def __init__(self, log_callback, symbol, timeframe):
        self.log_callback = log_callback
        self.symbol = symbol
        self.timeframe = timeframe
        self.model = None
        self.is_trained = False
        self.stop_flag = False
        self.training_history = {}
        self.importance = {}
        self.parity_row = None
        self._log(f"🚀 CatBoost trainer [{symbol} {timeframe}] initialized (CPU)")

    def _log(self, message):
        self.log_callback(message)

    def stop_training(self):
        self.stop_flag = True
        self._log("⏹️ Training stop requested")

    def train_single_timeframe(self, data, feature_cols, iterations, learning_rate, max_depth, horizon, status_callback):
        self.stop_flag = False
        tag = f"{self.symbol} {self.timeframe}"

        hp = Config.TIMEFRAME_HYPERPARAMS[self.timeframe]
        n_estimators = iterations or hp["iterations"]
        depth = max_depth or hp["max_depth"]
        reg_lambda = 3.0 if iterations else hp["reg_lambda"]

        self.parity_row = data.iloc[-1][["time"] + feature_cols]

        n_val = max(1, int(len(data) * Config.VALIDATION_FRACTION))
        train_df = data.iloc[:len(data) - n_val - horizon]
        val_df = data.iloc[len(data) - n_val:]

        X_train = train_df[feature_cols].to_numpy(dtype=np.float32)
        y_train = (train_df["target"] / train_df["close"] - 1.0).to_numpy(dtype=np.float32)
        X_val = val_df[feature_cols].to_numpy(dtype=np.float32)
        y_val = (val_df["target"] / val_df["close"] - 1.0).to_numpy(dtype=np.float32)
        del train_df, val_df

        self._log(f"[{tag}] Train rows: {len(y_train):,}, Val rows: {len(y_val):,}, Embargo: {horizon} bars")
        self._log(f"[{tag}] Model features ({len(feature_cols)}): {', '.join(feature_cols)}")
        self._log(f"[{tag}] Trees: {n_estimators}, depth: {depth}, lr: {learning_rate}, L2: {reg_lambda}")

        def after_iteration(info):
            if (info.iteration + 1) % self.LOG_EVERY_ROUNDS == 0:
                status_callback(f"[{tag}] Round {info.iteration + 1}/{n_estimators}  train={info.metrics['learn']['RMSE'][-1]:.6g}  val={info.metrics['validation']['RMSE'][-1]:.6g}")
            return not self.stop_flag

        self.model = catboost.CatBoostRegressor(
            iterations=n_estimators,
            learning_rate=learning_rate,
            depth=depth,
            l2_leaf_reg=reg_lambda,
            loss_function="RMSE",
            od_type="Iter",
            od_wait=self.EARLY_STOPPING_ROUNDS,
            use_best_model=True,
            random_seed=self.RANDOM_STATE,
            allow_writing_files=False,
            verbose=False,
        )
        self.model.fit(
            X_train, y_train,
            eval_set=(X_val, y_val),
            callbacks=[types.SimpleNamespace(after_iteration=after_iteration)],
        )
        self.is_trained = True

        gain = np.asarray(self.model.get_feature_importance(), dtype=np.float64)
        gain = 100 * gain / max(gain.sum(), 1e-12)
        self.importance = dict(zip(feature_cols, gain.round(4).tolist()))

        predicted = self.model.predict(X_val)
        valid_mask = y_val != 0
        self.training_history.update({
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "best_rounds": int(self.model.tree_count_),
            "rounds_run": len(self.model.get_evals_result()["validation"]["RMSE"]),
            "val_l2": float(np.mean((predicted - y_val) ** 2)),
            "directional_accuracy": float((np.sign(predicted[valid_mask]) == np.sign(y_val[valid_mask])).mean()),
        })
        top = sorted(self.importance.items(), key=lambda item: -item[1])[:self.TOP_IMPORTANCE]
        self._log(f"[{tag}] Top importance %: " + ", ".join(f"{k}={v:.1f}" for k, v in top))
        self._log(f"[{tag}] Zero-importance features: {sum(1 for v in self.importance.values() if v == 0)}")
        self._log(f"[{tag}] Directional accuracy: {self.training_history['directional_accuracy'] * 100:.1f}%")
        return self.is_trained

    def save_model(self, model_dir, filename_stem, feature_cols, horizon, export_onnx):
        output_dir = pathlib.Path(model_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        model_path = output_dir / f"{filename_stem}{self.MODEL_EXT}"
        onnx_path = output_dir / f"{filename_stem}{self.ONNX_EXT}"
        pkl_path = output_dir / f"{filename_stem}.pkl"

        self.model.save_model(str(model_path))

        onnx_input = None
        onnx_output = None

        if export_onnx:
            self.model.save_model(str(onnx_path), format="onnx")
            onnx_model = onnx.load(str(onnx_path))

            for key, value in (
                ("features", ",".join(feature_cols)),
                ("feature_version", str(Config.FEATURE_VERSION)),
            ):
                entry = onnx_model.metadata_props.add()
                entry.key = key
                entry.value = value

            onnx.save(onnx_model, str(onnx_path))

            onnx_input = onnx_model.graph.input[0].name
            onnx_output = onnx_model.graph.output[0].name

            del onnx_model

        artifact = {
            "model_type": "catboost.CatBoostRegressor",
            "target_type": "forward_return",
            "feature_version": Config.FEATURE_VERSION,
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "horizon": horizon,
            "warmup_bars": Config.WARMUP_BARS,
            "feature_columns": list(feature_cols),
            "feature_importance": self.importance,
            "model_path": str(model_path),
            "onnx_path": str(onnx_path) if export_onnx else None,
            "training_history": self.training_history,
            "training_time": datetime.datetime.now().isoformat(),
            "dtype": "float32",
            "fill": "inf_to_nan,ffill,zero",
            "target": "close[t+horizon]/close[t]-1",
            "time": "bar_open_csv_timezone",
            "onnx_input": onnx_input,
            "onnx_output": onnx_output,
            "parity_timestamp": self.parity_row["time"].value // 10**9,
            "parity_features": {
                column: float(self.parity_row[column])
                for column in feature_cols
            },
        }

        with open(pkl_path, "wb") as file:
            pickle.dump(artifact, file, protocol=pickle.HIGHEST_PROTOCOL)

        self._log(f"✅ CatBoost model saved: {model_path}")

        if export_onnx:
            self._log(f"✅ ONNX model saved: {onnx_path}")

        self._log(f"✅ PKL metadata saved: {pkl_path}")


class PredictTreeTrainerGUI(ctk.CTk):
    INI_PATH = pathlib.Path(__file__).with_suffix(".ini")

    WINDOW_W = 640
    WINDOW_H = 1020
    WRAP_W = 600
    PAD = 2
    BG = "#1c1c1c"
    ENTRY_BG = "#2b2b2b"
    BORDER_COLOR = "#444444"
    TEXT_COLOR = "white"
    TITLE_COLOR = "yellow"
    FONT_TEXT = ("TkDefaultFont", 12)
    FONT_TITLE = ("TkDefaultFont", 14, "bold")
    FONT_LOG = ("Courier", 12)
    TRAIN_COLOR = "#2e7d32"
    TRAIN_HOVER = "#1b5e20"
    STOP_COLOR = "#c62828"
    STOP_HOVER = "#b71c1c"
    FOLDER_BTN_W = 140
    BUTTON_W = 100
    ENTRY_W = 60
    CHECK_W = 100
    TF_CHECK_W = 80
    STATUS_H = 60

    TRAINERS = {
        "LightGBM": StreamingLightGBMTrainer,
        "XGBoost": StreamingXGBoostTrainer,
        "CatBoost": StreamingCatBoostTrainer,
    }

    def __init__(self):
        ctk.set_appearance_mode(Config.THEME)
        ctk.set_default_color_theme(Config.COLOR_THEME)
        Config.ensure_directories()

        super().__init__()
        self.title(Config.APP_TITLE)
        self.resizable(False, False)
        self.configure(fg_color=self.BG)
        self.withdraw()

        threading.excepthook = lambda args: self.report_callback_exception(args.exc_type, args.exc_value, args.exc_traceback)

        self.gui_logger = GUILogger()
        self.gui_logger.enable_console_redirect()

        self.symbol_timeframes = {}
        self.is_training = False
        self.stop_flag = False
        self.current_trainer = None
        self.worker_events = queue.Queue()
        self._closing_requested = False
        self.data_preparer = StreamingDataPreparer(self.gui_logger.log)

        self._create_ui()
        self._load_state()
        self.gui_logger.set_log_widget(self.log_text)
        self.after(50, self._process_worker_events)
        self.after(500, self.scan_data_source)
        self.protocol("WM_DELETE_WINDOW", self._on_closing)

        self.update_idletasks()
        self.deiconify()

    def _process_worker_events(self):
        self.gui_logger.drain()

        while not self.worker_events.empty():
            event, value = self.worker_events.get_nowait()

            if event == "status":
                self.status_label.configure(text=value)

            elif event == "finished":
                self._set_training(False)

                if self._closing_requested:
                    self._save_state()
                    self.gui_logger.close()
                    self.destroy()
                    return

        self.after(50, self._process_worker_events)

    def report_callback_exception(self, exc_type, exc_value, exc_traceback):
        traceback.print_exception(exc_type, exc_value, exc_traceback)
        os._exit(1)

    def _section(self, parent, title=None, expand=False):
        frame = ctk.CTkFrame(
            parent, fg_color=self.BG, border_color=self.BORDER_COLOR,
            border_width=self.PAD, corner_radius=self.PAD,
        )
        frame.pack(fill="both" if expand else "x", expand=expand, padx=self.PAD, pady=self.PAD)
        if title:
            self._title(frame, title)
        return frame

    def _title(self, parent, text):
        ctk.CTkLabel(
            parent, text=text, font=self.FONT_TITLE, text_color=self.TITLE_COLOR, anchor="w",
        ).pack(fill="x", padx=self.PAD, pady=self.PAD)

    def _row(self, parent):
        row = ctk.CTkFrame(parent, fg_color="transparent", border_width=0, corner_radius=0)
        row.pack(fill="x", padx=self.PAD, pady=self.PAD)
        return row

    def _label(self, parent, text, **kw):
        return ctk.CTkLabel(parent, text=text, font=self.FONT_TEXT, text_color=self.TEXT_COLOR, width=0, **kw)

    def _entry(self, parent, var, **kw):
        return ctk.CTkEntry(
            parent, textvariable=var, font=self.FONT_TEXT, text_color=self.TEXT_COLOR,
            fg_color=self.ENTRY_BG, border_width=self.PAD, **kw,
        )

    def _check(self, parent, text, var, width):
        return ctk.CTkCheckBox(
            parent, text=text, variable=var, font=self.FONT_TEXT, text_color=self.TEXT_COLOR,
            border_width=self.PAD, width=width,
        )

    def _button(self, parent, text, command, **kw):
        return ctk.CTkButton(
            parent, text=text, command=command, font=self.FONT_TEXT, text_color=self.TEXT_COLOR,
            border_width=0, **kw,
        )

    def _textbox(self, parent, scrollbar=False, **kw):
        return ctk.CTkTextbox(
            parent, font=self.FONT_LOG, text_color=self.TEXT_COLOR, fg_color=self.ENTRY_BG,
            border_width=self.PAD, activate_scrollbars=scrollbar, **kw,
        )

    def _create_ui(self):
        main = ctk.CTkFrame(self, fg_color=self.BG, corner_radius=0, border_width=0)
        main.pack(fill="both", expand=True, padx=self.PAD, pady=self.PAD)

        title_frame = ctk.CTkFrame(main, corner_radius=6)
        title_frame.pack(fill="x", pady=(0, 4))

        ctk.CTkLabel(
            title_frame,
            text="Copyright Su Nie | BSD-3C License | https://github.com/can87683",
            font=("TkDefaultFont", 14),
            text_color="yellow",
        ).pack(pady=(0, 2))

        ctk.CTkLabel(
            title_frame,
            text="Tree ML Trainer",
            font=("TkDefaultFont", 16, "bold"),
        ).pack(pady=4)

        self.usage_row = UsageRow(main)

        self.csv_path_var = tk.StringVar(value=str(Config.DATASET_DIR))
        row = self._row(self._section(main))
        self._button(row, "📁 CSV Folder", self.select_csv_folder, width=self.FOLDER_BTN_W).pack(side="left", padx=self.PAD, pady=self.PAD)
        self._entry(row, self.csv_path_var, state="readonly").pack(side="left", fill="x", expand=True, padx=self.PAD, pady=self.PAD)

        self.model_path_var = tk.StringVar(value=str(Config.MODEL_DIR))
        row = self._row(self._section(main))
        self._button(row, "📁 Model Folder", self.select_model_folder, width=self.FOLDER_BTN_W).pack(side="left", padx=self.PAD, pady=self.PAD)
        self._entry(row, self.model_path_var, state="readonly").pack(side="left", fill="x", expand=True, padx=self.PAD, pady=self.PAD)

        self.horizon = tk.StringVar(value=str(Config.DEFAULT_HORIZON))
        self.iterations = tk.StringVar(value=str(Config.DEFAULT_ITERATIONS))
        self.learning_rate = tk.StringVar(value=str(Config.DEFAULT_LEARNING_RATE))
        self.max_depth = tk.StringVar(value=str(Config.DEFAULT_MAX_DEPTH))
        self.auto_tune_var = tk.BooleanVar(value=True)
        frame = self._section(main, "Parameters")
        row = self._row(frame)
        params = (("Horizon:", self.horizon), ("Iterations:", self.iterations), ("LR:", self.learning_rate), ("Depth:", self.max_depth))
        for i, (text, var) in enumerate(params):
            row.grid_columnconfigure(i * 2 + 1, weight=1)
            self._label(row, text).grid(row=0, column=i * 2, padx=self.PAD, pady=self.PAD, sticky="w")
            self._entry(row, var, width=self.ENTRY_W).grid(row=0, column=i * 2 + 1, padx=self.PAD, pady=self.PAD, sticky="ew")
        self._check(frame, "Auto-tune depth / iterations / L2 per timeframe", self.auto_tune_var, self.CHECK_W).pack(fill="x", padx=self.PAD, pady=self.PAD)

        self.backend_vars = {}
        frame = self._section(main, "Models (one per symbol/timeframe each)")
        row = self._row(frame)
        for i, name in enumerate(self.TRAINERS):
            self.backend_vars[name] = tk.BooleanVar(value=True)
            row.grid_columnconfigure(i, weight=1)
            self._check(row, name, self.backend_vars[name], self.CHECK_W).grid(row=0, column=i, padx=self.PAD, pady=self.PAD, sticky="ew")

        self.tf_vars = {}
        frame = self._section(main, "Timeframes (one model per symbol)")
        row = self._row(frame)
        for i, tf in enumerate(Config.TIMEFRAMES):
            self.tf_vars[tf] = tk.BooleanVar(value=True)
            row.grid_columnconfigure(i, weight=1)
            self._check(row, tf, self.tf_vars[tf], self.TF_CHECK_W).grid(row=0, column=i, padx=self.PAD, pady=self.PAD, sticky="ew")

        names = FeatureEngineer.feature_names()
        frame = self._section(main, f"Training Features ({len(names)})")
        self._label(frame, ", ".join(names), anchor="w", justify="left", wraplength=self.WRAP_W).pack(fill="x", padx=self.PAD, pady=self.PAD)

        self.onnx_var = tk.BooleanVar(value=True)
        frame = self._section(main, "Controls")
        row = self._row(frame)
        for c in range(3):
            row.grid_columnconfigure(c, weight=1)
        self._check(row, "ONNX Export", self.onnx_var, self.CHECK_W).grid(row=0, column=0, padx=self.PAD, pady=self.PAD, sticky="ew")
        self.train_btn = self._button(row, "🚀 Start", self.start_training, width=self.BUTTON_W, fg_color=self.TRAIN_COLOR, hover_color=self.TRAIN_HOVER)
        self.train_btn.grid(row=0, column=1, padx=self.PAD, pady=self.PAD, sticky="ew")
        self.stop_btn = self._button(row, "⏹️ Stop", self.stop_training, width=self.BUTTON_W, fg_color=self.STOP_COLOR, hover_color=self.STOP_HOVER, state="disabled")
        self.stop_btn.grid(row=0, column=2, padx=self.PAD, pady=self.PAD, sticky="ew")

        frame = self._section(main, "Status")
        self.status_label = self._label(frame, "📊 No folder selected", anchor="w")
        self.status_label.pack(fill="x", padx=self.PAD, pady=self.PAD)
        self.status_text = self._textbox(frame, height=self.STATUS_H)
        self.status_text.pack(fill="x", padx=self.PAD, pady=self.PAD)
        self.status_text.insert("1.0", "Ready")
        self.status_text.configure(state="disabled")

        frame = self._section(main, "Log", expand=True)
        self.log_text = self._textbox(frame, scrollbar=True)
        self.log_text.pack(fill="both", expand=True, padx=self.PAD, pady=self.PAD)
        self.copy_menu = tk.Menu(self, tearoff=0, bg=self.ENTRY_BG, fg=self.TEXT_COLOR, font=self.FONT_TEXT)
        self.copy_menu.add_command(label="Copy selected", command=self._copy_selected)
        self.copy_menu.add_command(label="Copy all", command=self._copy_all)
        self.log_text.bind("<Button-3>", self._show_copy_menu)

    def _show_copy_menu(self, event):
        self.copy_menu.tk_popup(event.x_root, event.y_root)

    def _copy_selected(self):
        if self.log_text.tag_ranges("sel"):
            self._to_clipboard(self.log_text.get("sel.first", "sel.last"))

    def _copy_all(self):
        self._to_clipboard(self.log_text.get("1.0", "end-1c"))

    def _to_clipboard(self, text):
        self.clipboard_clear()
        self.clipboard_append(text)

    def _load_state(self):
        if not self.INI_PATH.exists():
            self.update_idletasks()
            x = max(0, (self.winfo_screenwidth() - self.WINDOW_W) // 2)
            y = max(0, (self.winfo_screenheight() - self.WINDOW_H) // 2)
            self.geometry(f"{self.WINDOW_W}x{self.WINDOW_H}+{x}+{y}")
            return

        cp = configparser.ConfigParser()
        cp.read(str(self.INI_PATH))

        if cp.has_option("window", "geometry"):
            self.geometry(cp.get("window", "geometry"))
        else:
            x = cp.getint("window", "x")
            y = cp.getint("window", "y")
            self.geometry(f"{self.WINDOW_W}x{self.WINDOW_H}+{x}+{y}")

        self.csv_path_var.set(cp["paths"]["csv_folder"])
        self.model_path_var.set(cp["paths"]["model_folder"])
        self.horizon.set(cp["params"]["horizon"])
        self.iterations.set(cp["params"]["iterations"])
        self.learning_rate.set(cp["params"]["learning_rate"])
        self.max_depth.set(cp["params"]["max_depth"])
        self.auto_tune_var.set(cp.getboolean("params", "auto_tune"))
        self.onnx_var.set(cp.getboolean("params", "onnx_export"))

        for name, var in self.backend_vars.items():
            var.set(cp.getboolean("backends", name))

        for tf, var in self.tf_vars.items():
            var.set(cp.getboolean("timeframes", tf))

    def _save_state(self):
        cp = configparser.ConfigParser()
        cp["window"] = {
            "geometry": self.geometry()
        }
        cp["paths"] = {
            "csv_folder": self.csv_path_var.get(),
            "model_folder": self.model_path_var.get()
        }
        cp["params"] = {
            "horizon": self.horizon.get(),
            "iterations": self.iterations.get(),
            "learning_rate": self.learning_rate.get(),
            "max_depth": self.max_depth.get(),
            "auto_tune": str(self.auto_tune_var.get()),
            "onnx_export": str(self.onnx_var.get()),
        }
        cp["backends"] = {
            name: str(var.get()) for name, var in self.backend_vars.items()
        }
        cp["timeframes"] = {
            tf: str(var.get()) for tf, var in self.tf_vars.items()
        }

        with open(self.INI_PATH, "w") as file:
            cp.write(file)

    def update_status(self, message):
        self.worker_events.put(("status", message))

    def _set_training(self, active):
        self.is_training = active
        self.train_btn.configure(state="disabled" if active else "normal")
        self.stop_btn.configure(state="normal" if active else "disabled")

    def select_csv_folder(self):
        folder = filedialog.askdirectory(title="Select CSV Data Folder", initialdir=self.csv_path_var.get())
        if folder:
            self.csv_path_var.set(folder)
            self.scan_data_source()

    def select_model_folder(self):
        folder = filedialog.askdirectory(title="Select Model Save Folder", initialdir=self.model_path_var.get())
        if folder:
            pathlib.Path(folder).mkdir(parents=True, exist_ok=True)
            self.model_path_var.set(folder)
            self.gui_logger.log(f"💾 Model folder set to: {folder}")

    def scan_data_source(self):
        csv_folder = pathlib.Path(self.csv_path_var.get())
        if not csv_folder.exists():
            self.gui_logger.log(f"❌ CSV folder not found: {csv_folder}", "error")
            self.status_label.configure(text="❌ Folder not found")
            self.train_btn.configure(state="disabled")
            return
        self.gui_logger.log(f"📁 Scanning CSV folder: {csv_folder}")
        symbols, timeframes = self.data_preparer.scan_folder(csv_folder)
        if not symbols:
            self.gui_logger.log("⚠️ No symbols found", "warning")
            self.status_label.configure(text="⚠️ No symbols found")
            self.train_btn.configure(state="disabled")
            return
        self.symbol_timeframes = dict(symbols)
        self.status_label.configure(text=f"✅ {len(symbols)} symbols, {len(timeframes)} timeframes")
        self.gui_logger.log(f"✅ Loaded {len(symbols)} symbols", "success")
        self.train_btn.configure(state="normal")

    def start_training(self):
        if self.is_training:
            return

        selected_timeframes = [
            tf for tf, var in self.tf_vars.items() if var.get()
        ]
        selected_backends = [
            name for name, var in self.backend_vars.items() if var.get()
        ]

        if not selected_timeframes or not selected_backends:
            self.gui_logger.log(
                "❌ Select at least one backend and one timeframe",
                "error",
            )
            return

        csv_folder = self.csv_path_var.get()
        model_folder = self.model_path_var.get()

        horizon = int(self.horizon.get())
        learning_rate = float(self.learning_rate.get())
        gui_iterations = int(self.iterations.get())
        gui_max_depth = int(self.max_depth.get())

        auto_tune = self.auto_tune_var.get()
        export_onnx = self.onnx_var.get()
        symbol_timeframes = dict(self.symbol_timeframes)

        if horizon < 1:
            raise ValueError("Prediction horizon must be at least 1")

        if learning_rate <= 0:
            raise ValueError("Learning rate must be greater than zero")

        if gui_iterations < 1:
            raise ValueError("Iterations must be at least 1")

        if gui_max_depth < 1:
            raise ValueError("Maximum depth must be at least 1")

        if not pathlib.Path(csv_folder).is_dir():
            raise ValueError(f"CSV folder does not exist: {csv_folder}")

        pathlib.Path(model_folder).mkdir(parents=True, exist_ok=True)

        self.stop_flag = False
        self._set_training(True)

        def training_task():
            self.gui_logger.log("=" * 60)
            self.gui_logger.log("🚀 Starting Per-Symbol Per-Timeframe Training")
            self.gui_logger.log("=" * 60)
            self.gui_logger.log(f"📂 CSV Folder: {csv_folder}")
            self.gui_logger.log(
                f"💾 System RAM: "
                f"{MemoryOptimizer.get_system_memory_available_gb():.1f} GB available"
            )
            self.gui_logger.log(
                f"🧮 Mandatory columns: {', '.join(Config.REQUIRED_COLUMNS)}"
            )
            self.gui_logger.log(
                f"🤖 Backends: {', '.join(selected_backends)}"
            )

            feature_names = FeatureEngineer.feature_names()

            self.gui_logger.log(
                f"📊 Training features ({len(feature_names)}): "
                f"{', '.join(feature_names)}"
            )
            self.gui_logger.log(
                f"🧩 ONNX export: {'enabled' if export_onnx else 'disabled'}"
            )

            jobs = [
                (symbol, tf)
                for tf in selected_timeframes
                for symbol, tfs in symbol_timeframes.items()
                if tf in tfs
            ]

            self.gui_logger.log(
                f"🧱 Models to train: {len(jobs) * len(selected_backends)}"
            )

            results = {}

            for symbol, timeframe in jobs:
                if self.stop_flag:
                    self.gui_logger.log(
                        "⏹️ Training stopped by user",
                        "warning",
                    )
                    break

                tag = f"{symbol} {timeframe}"

                self.gui_logger.log("-" * 60)
                self.gui_logger.log(f"▶️ {tag}")
                self.update_status(f"[{tag}] Building dataset...")

                data = self.data_preparer.build_dataframe_for_symbol(
                    csv_folder,
                    symbol,
                    timeframe,
                    horizon,
                )

                if data is None:
                    self.gui_logger.log(
                        f"⚠️ [{tag}] Skipped: not enough rows after warmup",
                        "warning",
                    )
                    results[tag] = "skipped"
                    continue

                feature_columns = self.data_preparer.feature_cols

                for backend in selected_backends:
                    if self.stop_flag:
                        break

                    job_tag = f"{backend} {tag}"

                    self.update_status(f"[{job_tag}] Training...")

                    trainer = self.TRAINERS[backend](
                        self.gui_logger.log,
                        symbol,
                        timeframe,
                    )
                    self.current_trainer = trainer

                    success = trainer.train_single_timeframe(
                        data=data,
                        feature_cols=feature_columns,
                        iterations=None if auto_tune else gui_iterations,
                        learning_rate=learning_rate,
                        max_depth=None if auto_tune else gui_max_depth,
                        horizon=horizon,
                        status_callback=self.update_status,
                    )

                    if trainer.stop_flag:
                        results[job_tag] = "stopped"
                        self.stop_flag = True

                    elif success:
                        filename_stem = (
                            f"{trainer.MODEL_PREFIX}_{symbol}_{timeframe}"
                        )

                        trainer.save_model(
                            model_dir=model_folder,
                            filename_stem=filename_stem,
                            feature_cols=feature_columns,
                            horizon=horizon,
                            export_onnx=export_onnx,
                        )

                        self.gui_logger.log(
                            f"✅ [{job_tag}] Model saved: "
                            f"{filename_stem}{trainer.MODEL_EXT}",
                            "success",
                        )
                        results[job_tag] = "trained"

                    else:
                        results[job_tag] = "stopped"

                    del trainer
                    self.current_trainer = None
                    MemoryOptimizer.clear_cache()

                del data

            counts = {
                status: list(results.values()).count(status)
                for status in set(results.values())
            }

            summary = ", ".join(
                f"{status}: {count}"
                for status, count in sorted(counts.items())
            )

            self.gui_logger.log("=" * 60)
            self.gui_logger.log(f"🏁 Summary: {summary}")
            self.update_status(f"Done. {summary}")

            self.worker_events.put(("finished", summary))

        threading.Thread(
            target=training_task,
            name="tree-ml-training",
            daemon=True,
        ).start()

    def stop_training(self):
        self.stop_flag = True
        if self.current_trainer is not None:
            self.current_trainer.stop_training()
        self.gui_logger.log("⏹️ Stopping training...", "warning")

    def _on_closing(self):
        message = (
            "Training is running.\n\nStop training and quit?"
            if self.is_training
            else "Quit AI Predict Train?"
        )

        if not messagebox.askyesno("Quit", message):
            return

        if self.is_training:
            self._closing_requested = True
            self.stop_training()
            return

        self._save_state()
        self.gui_logger.close()
        self.destroy()


if __name__ == "__main__":
    app = PredictTreeTrainerGUI()
    app.mainloop()