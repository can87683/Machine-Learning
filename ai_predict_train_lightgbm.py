#!/usr/bin/env python3
# ai_predict_train_lightgbm_ctk.py
# BSD-C-3 License

import setproctitle
setproctitle.setproctitle("ai_predict_train_lightgbm_ctk")

import customtkinter as ctk
from customtkinter import CTk, CTkFrame, CTkLabel, CTkButton, CTkEntry, CTkTextbox
import tkinter as tk
from tkinter import filedialog, messagebox
import threading
import subprocess
from pathlib import Path
import numpy as np
import pandas as pd
import pickle
import json
import gc
import psutil
from datetime import datetime
import traceback
import sys
import os
import time
import warnings

import lightgbm as lgb

# Suppress FutureWarnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


class Config:
    """Central configuration for all settings"""

    # Paths
    HOME = Path.home()
    MODEL_DIR = HOME / "ai_model_ml"
    DATASET_DIR = HOME / "mt5_data"

    # Data settings
    MAX_ROWS_PER_SYMBOL = 20000
    MAX_DATASET_ROWS = 200000
    DEFAULT_FEATURES = ['time', 'open', 'high', 'low', 'close', 'volume']

    # Model settings (LightGBM)
    DEFAULT_HORIZON = 10
    DEFAULT_ITERATIONS = 1000
    DEFAULT_LEARNING_RATE = 0.05
    DEFAULT_MAX_DEPTH = 6
    DEFAULT_NUM_LEAVES = 63
    DEFAULT_L2_REG = 1.0
    DEFAULT_SUBSAMPLE = 0.8
    DEFAULT_COLSAMPLE = 0.8
    DEFAULT_BAGGING_FREQ = 1

    TIMEFRAME_HYPERPARAMS = {
        "M5":  {"max_depth": 6, "iterations": 1500, "reg_lambda": 1.0},
        "M15": {"max_depth": 6, "iterations": 1500, "reg_lambda": 1.0},
        "H1":  {"max_depth": 6, "iterations": 1200, "reg_lambda": 2.0},
        "H4":  {"max_depth": 5, "iterations": 1000, "reg_lambda": 3.0},
        "D1":  {"max_depth": 4, "iterations": 600,  "reg_lambda": 4.0},
        "W1":  {"max_depth": 3, "iterations": 300,  "reg_lambda": 6.0},
    }

    EARLY_STOPPING_ROUNDS = 50
    VALIDATION_FRACTION = 0.15
    MEMORY_WARNING_THRESHOLD_GB = 0.5
    TIMEFRAMES = ['M5', 'M15', 'H1', 'H4', 'D1', 'W1']
    MODEL_PREFIX = "ai_predict_lightgbm"
    MODEL_EXT = ".txt"
    METADATA_EXT = ".pkl"

    APP_TITLE = "🚀 LightGBM Model Trainer"
    APP_GEOMETRY = "640x1050"
    THEME = "dark"
    COLOR_THEME = "dark-blue"

    @classmethod
    def ensure_directories(cls):
        cls.MODEL_DIR.mkdir(parents=True, exist_ok=True)
        cls.DATASET_DIR.mkdir(parents=True, exist_ok=True)


Config.ensure_directories()

ctk.set_appearance_mode(Config.THEME)
ctk.set_default_color_theme(Config.COLOR_THEME)

HOME = Config.HOME
MODEL_DIR = Config.MODEL_DIR
DATASET_DIR = Config.DATASET_DIR


class GPUManager:
    def __init__(self, verbose=True):
        self.verbose = verbose
        self.gpu_available = False
        self.gpu_name = None
        self._detect_gpu()

    def _log(self, message):
        if self.verbose:
            print(f"[GPUManager] {message}")

    def _detect_gpu(self):
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                timeout=3,
            )
            if result.returncode == 0 and result.stdout.strip():
                self.gpu_available = True
                self.gpu_name = result.stdout.strip().splitlines()[0]
                self._log(f"✅ GPU detected: {self.gpu_name}")
            else:
                self._log("ℹ️ No GPU detected, using CPU")
        except (FileNotFoundError, subprocess.TimeoutExpired, Exception):
            self.gpu_available = False
            self._log("ℹ️ No GPU detected, using CPU")

    def preferred_devices(self):
        if self.gpu_available:
            return ["cuda", "gpu", "cpu"]
        return ["cpu"]


class UsageRow:
    def __init__(self, parent):
        self.parent = parent
        self.usage_frame = None
        self.usage_label = None
        self._create_usage_row()

    def _create_usage_row(self):
        self.usage_frame = CTkFrame(self.parent, height=35, corner_radius=5)
        self.usage_frame.pack(fill="x", pady=(0, 5))

        self.usage_label = CTkLabel(
            self.usage_frame,
            text="Loading system usage...",
            font=("Ubuntu Mono", 18),
            text_color="Lime"
        )
        self.usage_label.pack(expand=True, fill="x", padx=10, pady=3)

        self._update_usage()

    def _update_usage(self):
        try:
            cpu = psutil.cpu_percent(interval=0.5)
            dram = psutil.virtual_memory().percent

            gpu_percent = 0
            vram_percent = 0
            try:
                import GPUtil
                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu = gpus[0]
                    gpu_percent = gpu.load * 100
                    vram_percent = gpu.memoryUtil * 100
            except (ImportError, Exception):
                pass

            self.usage_label.configure(
                text=f"CPU: {cpu:.1f}%  DRAM: {dram:.1f}%  GPU: {gpu_percent:.1f}%  VRAM: {vram_percent:.1f}%"
            )
        except Exception:
            self.usage_label.configure(text="⚠️ System monitoring unavailable")

        self.usage_label.after(2000, self._update_usage)


class ConsoleLogger:
    def __init__(self, gui_logger, original_stream):
        self.gui_logger = gui_logger
        self.original_stream = original_stream

    def write(self, message):
        if message and message.strip():
            self.original_stream.write(message)
            self.original_stream.flush()
            if self.gui_logger:
                self.gui_logger.log(message.strip(), "console")

    def flush(self):
        self.original_stream.flush()


class MemoryOptimizer:
    @staticmethod
    def get_system_memory_gb():
        return psutil.virtual_memory().total / 1e9

    @staticmethod
    def get_system_memory_available_gb():
        return psutil.virtual_memory().available / 1e9

    @staticmethod
    def clear_cache():
        gc.collect()

    @staticmethod
    def monitor_memory(threshold_gb=0.5):
        return MemoryOptimizer.get_system_memory_available_gb() < threshold_gb


class FeatureEngineer:
    FLAG_TO_FEATURES = {
        'enable_adx': ['adx', 'plus_di', 'minus_di'],
        'enable_atr': ['atr'],
        'enable_bb': ['bb_upper', 'bb_lower', 'bb_middle', 'bb_width', 'bb_percent'],
        'enable_cci': ['cci'],
        'enable_ema': ['ema_12', 'ema_26', 'ema_50', 'ema_200'],
        'enable_kc': ['kc_upper', 'kc_lower', 'kc_middle', 'kc_width'],
        'enable_macd': ['macd', 'macd_signal', 'macd_hist'],
        'enable_mfi': ['mfi'],
        'enable_obv': ['obv'],
        'enable_roc': ['roc'],
        'enable_rsi': ['rsi'],
        'enable_sma': ['sma_20', 'sma_50', 'sma_200'],
        'enable_stochastic': ['stoch_k', 'stoch_d'],
        'enable_vol_ma': ['volume_ema_20', 'volume_sma_20'],
        'enable_vwap': ['vwap'],
        'enable_willr': ['willr'],
    }

    @staticmethod
    def add_time_features(df):
        if 'time' not in df.columns:
            return df

        if not pd.api.types.is_datetime64_any_dtype(df['time']):
            df['time'] = pd.to_datetime(df['time'])

        df['hour'] = df['time'].dt.hour
        df['day_of_week'] = df['time'].dt.dayofweek
        df['day_of_month'] = df['time'].dt.day
        df['month'] = df['time'].dt.month
        df['quarter'] = df['time'].dt.quarter
        df['year'] = df['time'].dt.year

        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

        return df

    @staticmethod
    def add_technical_features(df, **flags):
        df = df.copy()

        if all(c in df.columns for c in ['high', 'low', 'close']):
            df['high_low_ratio'] = df['high'] / df['low']
            df['close_high_ratio'] = df['close'] / df['high']
            df['close_low_ratio'] = df['close'] / df['low']

        if 'close' in df.columns:
            df['returns_1'] = df['close'].pct_change()
            df['returns_5'] = df['close'].pct_change(periods=5)
            df['returns_10'] = df['close'].pct_change(periods=10)

        if 'volume' in df.columns:
            df['volume_ratio'] = df['volume'] / df['volume'].rolling(5, min_periods=1).mean()

        if all(c in df.columns for c in ['high', 'low', 'close']):
            prev_close = df['close'].shift(1)
            df['true_range'] = pd.concat([
                df['high'] - df['low'],
                (df['high'] - prev_close).abs(),
                (df['low'] - prev_close).abs(),
            ], axis=1).max(axis=1)

        if flags.get('enable_adx', False) and all(c in df.columns for c in ['high', 'low', 'close']):
            df = FeatureEngineer._add_adx(df, period=14)

        if flags.get('enable_atr', False) and 'true_range' in df.columns:
            df['atr'] = df['true_range'].ewm(alpha=1/14, min_periods=1, adjust=False).mean()

        if flags.get('enable_bb', False) and 'close' in df.columns:
            df = FeatureEngineer._add_bb(df, period=20)

        if flags.get('enable_cci', False) and all(c in df.columns for c in ['high', 'low', 'close']):
            df = FeatureEngineer._add_cci(df, period=20)

        if flags.get('enable_ema', False) and 'close' in df.columns:
            df['ema_12'] = df['close'].ewm(span=12, min_periods=1, adjust=False).mean()
            df['ema_26'] = df['close'].ewm(span=26, min_periods=1, adjust=False).mean()
            df['ema_50'] = df['close'].ewm(span=50, min_periods=1, adjust=False).mean()
            df['ema_200'] = df['close'].ewm(span=200, min_periods=1, adjust=False).mean()

        if flags.get('enable_kc', False) and all(c in df.columns for c in ['high', 'low', 'close']):
            df = FeatureEngineer._add_kc(df, period=20)

        if flags.get('enable_macd', False) and 'close' in df.columns:
            ema_fast = df['close'].ewm(span=12, adjust=False).mean()
            ema_slow = df['close'].ewm(span=26, adjust=False).mean()
            macd_line = ema_fast - ema_slow
            signal_line = macd_line.ewm(span=9, adjust=False).mean()
            df['macd'] = macd_line
            df['macd_signal'] = signal_line
            df['macd_hist'] = macd_line - signal_line

        if flags.get('enable_mfi', False) and all(c in df.columns for c in ['high', 'low', 'close', 'volume']):
            df = FeatureEngineer._add_mfi(df, period=14)

        if flags.get('enable_obv', False) and all(c in df.columns for c in ['close', 'volume']):
            df = FeatureEngineer._add_obv(df)

        if flags.get('enable_roc', False) and 'close' in df.columns:
            df['roc'] = ((df['close'] - df['close'].shift(12)) / df['close'].shift(12).replace(0, np.nan)) * 100

        if flags.get('enable_rsi', False) and 'close' in df.columns:
            df = FeatureEngineer._add_rsi(df, period=14)

        if flags.get('enable_sma', False) and 'close' in df.columns:
            df['sma_20'] = df['close'].rolling(20, min_periods=1).mean()
            df['sma_50'] = df['close'].rolling(50, min_periods=1).mean()
            df['sma_200'] = df['close'].rolling(200, min_periods=1).mean()

        if flags.get('enable_stochastic', False) and all(c in df.columns for c in ['high', 'low', 'close']):
            df = FeatureEngineer._add_stochastic(df, period=14, smooth=3)

        if flags.get('enable_vol_ma', False) and 'volume' in df.columns:
            df['volume_ema_20'] = df['volume'].ewm(span=20, min_periods=1, adjust=False).mean()
            df['volume_sma_20'] = df['volume'].rolling(20, min_periods=1).mean()

        if flags.get('enable_vwap', False) and all(c in df.columns for c in ['high', 'low', 'close', 'volume']):
            typical = (df['high'] + df['low'] + df['close']) / 3
            df['vwap'] = (typical * df['volume']).cumsum() / df['volume'].cumsum()

        if flags.get('enable_willr', False) and all(c in df.columns for c in ['high', 'low', 'close']):
            lowest = df['low'].rolling(14, min_periods=1).min()
            highest = df['high'].rolling(14, min_periods=1).max()
            denom = (highest - lowest).replace(0, np.nan)
            df['willr'] = -100 * (highest - df['close']) / denom

        return df

    @staticmethod
    def _add_rsi(df, period=14):
        delta = df['close'].diff()
        gain = delta.clip(lower=0)
        loss = (-delta).clip(upper=0)
        avg_gain = gain.ewm(alpha=1/period, min_periods=1, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/period, min_periods=1, adjust=False).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        df['rsi'] = 100 - (100 / (1 + rs))
        return df

    @staticmethod
    def _add_stochastic(df, period=14, smooth=3):
        lowest_low = df['low'].rolling(period, min_periods=1).min()
        highest_high = df['high'].rolling(period, min_periods=1).max()
        denom = (highest_high - lowest_low).replace(0, np.nan)
        df['stoch_k'] = 100 * (df['close'] - lowest_low) / denom
        df['stoch_d'] = df['stoch_k'].rolling(smooth, min_periods=1).mean()
        return df

    @staticmethod
    def _add_bb(df, period=20):
        middle = df['close'].rolling(period, min_periods=1).mean()
        std = df['close'].rolling(period, min_periods=1).std()
        df['bb_middle'] = middle
        df['bb_upper'] = middle + 2 * std
        df['bb_lower'] = middle - 2 * std
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / middle.replace(0, np.nan)
        df['bb_percent'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower']).replace(0, np.nan)
        return df

    @staticmethod
    def _add_adx(df, period=14):
        plus_dm = df['high'].diff()
        minus_dm = -df['low'].diff()
        plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
        minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
        tr = df['true_range']
        atr = tr.ewm(alpha=1/period, min_periods=1, adjust=False).mean()
        plus_di = 100 * (plus_dm.ewm(alpha=1/period, min_periods=1, adjust=False).mean() / atr.replace(0, np.nan))
        minus_di = 100 * (minus_dm.ewm(alpha=1/period, min_periods=1, adjust=False).mean() / atr.replace(0, np.nan))
        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
        df['plus_di'] = plus_di
        df['minus_di'] = minus_di
        df['adx'] = dx.ewm(alpha=1/period, min_periods=1, adjust=False).mean()
        return df

    @staticmethod
    def _add_cci(df, period=20):
        typical = (df['high'] + df['low'] + df['close']) / 3
        sma_typ = typical.rolling(period, min_periods=1).mean()
        mean_dev = typical.rolling(period, min_periods=1).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True)
        df['cci'] = (typical - sma_typ) / (0.015 * mean_dev.replace(0, np.nan))
        return df

    @staticmethod
    def _add_kc(df, period=20):
        typical = (df['high'] + df['low'] + df['close']) / 3
        middle = typical.rolling(period, min_periods=1).mean()
        atr = df['true_range'].ewm(alpha=1/period, min_periods=1, adjust=False).mean()
        df['kc_middle'] = middle
        df['kc_upper'] = middle + 2 * atr
        df['kc_lower'] = middle - 2 * atr
        df['kc_width'] = (df['kc_upper'] - df['kc_lower']) / middle.replace(0, np.nan)
        return df

    @staticmethod
    def _add_mfi(df, period=14):
        typical = (df['high'] + df['low'] + df['close']) / 3
        raw_money = typical * df['volume']
        diff = typical.diff()
        positive = raw_money.where(diff > 0, 0)
        negative = raw_money.where(diff < 0, 0)
        pos_sum = positive.rolling(period, min_periods=1).sum()
        neg_sum = negative.rolling(period, min_periods=1).sum()
        mfi_ratio = pos_sum / neg_sum.replace(0, np.nan)
        df['mfi'] = 100 - (100 / (1 + mfi_ratio))
        return df

    @staticmethod
    def _add_obv(df):
        obv = [0]
        for i in range(1, len(df)):
            if df['close'].iloc[i] > df['close'].iloc[i-1]:
                obv.append(obv[-1] + df['volume'].iloc[i])
            elif df['close'].iloc[i] < df['close'].iloc[i-1]:
                obv.append(obv[-1] - df['volume'].iloc[i])
            else:
                obv.append(obv[-1])
        df['obv'] = pd.Series(obv, index=df.index)
        return df


class GUILogger:
    def __init__(self):
        self.messages = []
        self.log_widget = None
        self.ready = False
        self._original_stdout = sys.stdout
        self._original_stderr = sys.stderr

    def set_log_widget(self, widget):
        self.log_widget = widget
        self.ready = True
        self.log_widget.configure(font=("Ubuntu Mono", 11))
        for msg, level in self.messages:
            self._write_to_widget(msg, level)
        self.messages.clear()

    def _write_to_widget(self, message, level):
        if self.log_widget:
            timestamp = datetime.now().strftime("%H:%M:%S")
            log_entry = f"[{timestamp}] {message}\n"
            try:
                self.log_widget.insert("end", log_entry)
                self.log_widget.see("end")
                self.log_widget.update_idletasks()
            except Exception:
                print(log_entry, flush=True)

    def log(self, message, level="info"):
        timestamp = datetime.now().strftime("%H:%M:%S")
        console_entry = f"[{timestamp}] [{level.upper()}] {message}"
        try:
            self._original_stdout.write(console_entry + "\n")
            self._original_stdout.flush()
        except Exception:
            print(console_entry, flush=True)
        if self.ready and self.log_widget:
            self._write_to_widget(message, level)
        else:
            self.messages.append((message, level))

    def enable_console_redirect(self):
        sys.stdout = ConsoleLogger(self, self._original_stdout)
        sys.stderr = ConsoleLogger(self, self._original_stderr)

    def disable_console_redirect(self):
        sys.stdout = self._original_stdout
        sys.stderr = self._original_stderr


class StreamingDataPreparer:
    def __init__(self, log_callback=None):
        self.log_callback = log_callback or print
        self.feature_cols = []
        self.categorical_features = ["group_id"]
        self.target_column = "target"
        self.indicator_flags = {k: True for k in FeatureEngineer.FLAG_TO_FEATURES.keys()}

    def _log(self, message):
        if self.log_callback:
            self.log_callback(message)

    def set_indicator_flags(self, **kwargs):
        for k, v in kwargs.items():
            if k in self.indicator_flags:
                self.indicator_flags[k] = bool(v)

    def load_csv_data(self, symbol, timeframe, csv_folder):
        csv_file = Path(csv_folder) / symbol / f"{symbol}_{timeframe}.csv"
        if not csv_file.exists():
            return None
        try:
            df = pd.read_csv(csv_file)
            if "time" not in df.columns:
                self._log(f"⚠️ Missing time column: {csv_file}")
                return None
            if pd.api.types.is_numeric_dtype(df["time"]):
                df["time"] = pd.to_datetime(df["time"], unit="s")
            else:
                df["time"] = pd.to_datetime(df["time"])
            df = df.sort_values("time").drop_duplicates("time").reset_index(drop=True)
            return df
        except Exception as exc:
            self._log(f"❌ Error loading {csv_file}: {exc}")
            return None

    def scan_folder(self, csv_folder):
        csv_folder = Path(csv_folder)
        if not csv_folder.exists():
            return [], []
        symbols = []
        all_timeframes = set()
        for symbol_dir in sorted(csv_folder.iterdir()):
            if not symbol_dir.is_dir():
                continue
            symbol = symbol_dir.name
            found_timeframes = []
            for timeframe in Config.TIMEFRAMES:
                csv_file = symbol_dir / f"{symbol}_{timeframe}.csv"
                if csv_file.exists():
                    found_timeframes.append(timeframe)
                    all_timeframes.add(timeframe)
            if found_timeframes:
                symbols.append((symbol, found_timeframes))
                self._log(f"   ✓ {symbol}: {', '.join(found_timeframes)}")
        return symbols, sorted(all_timeframes)

    def _prepare_single_series(self, df, symbol, horizon, requested_features):
        if df is None or len(df) < 100:
            return None
        df = df.copy()
        if len(df) > Config.MAX_ROWS_PER_SYMBOL:
            df = df.iloc[-Config.MAX_ROWS_PER_SYMBOL:].copy()

        df = FeatureEngineer.add_time_features(df)
        df = FeatureEngineer.add_technical_features(df, **self.indicator_flags)

        requested_features = [c for c in requested_features if c != "time"]

        base_engineered = [
            "hour_sin", "hour_cos", "day_sin", "day_cos",
            "month_sin", "month_cos",
            "high_low_ratio", "close_high_ratio", "close_low_ratio",
            "returns_1", "returns_5", "returns_10", "volume_ratio",
        ]

        indicator_features = []
        for flag, cols in FeatureEngineer.FLAG_TO_FEATURES.items():
            if self.indicator_flags.get(flag, False):
                indicator_features.extend(cols)

        candidate_features = list(dict.fromkeys(requested_features + base_engineered + indicator_features))
        available_features = [c for c in candidate_features if c in df.columns]

        required_columns = ["open", "high", "low", "close", "volume"]
        for column in required_columns:
            if column in df.columns and column not in available_features:
                available_features.append(column)

        if "close" not in df.columns:
            self._log(f"⚠️ Skipping {symbol}: close column unavailable")
            return None
        if not available_features:
            self._log(f"⚠️ Skipping {symbol}: no usable features")
            return None

        df[available_features] = (
            df[available_features]
            .replace([np.inf, -np.inf], np.nan)
            .ffill()
            .bfill()
            .fillna(0.0)
        )

        df[self.target_column] = df["close"].shift(-horizon)
        df = df.dropna(subset=[self.target_column]).copy()
        if len(df) < 50:
            return None

        df["group_id"] = str(symbol)
        df["time_idx"] = np.arange(len(df), dtype=np.int64)

        keep_columns = ["time", "time_idx", "group_id", self.target_column] + available_features
        return df[keep_columns], available_features

    def build_dataframe_for_timeframe(self, csv_folder, feature_cols, horizon, timeframe):
        csv_folder = Path(csv_folder)
        symbols, _ = self.scan_folder(csv_folder)
        symbols_with_tf = [s for s, tfs in symbols if timeframe in tfs]
        if not symbols_with_tf:
            raise ValueError(f"No symbols found with timeframe {timeframe} in {csv_folder}")

        frames = []
        all_features = []
        for symbol in symbols_with_tf:
            df = self.load_csv_data(symbol, timeframe, csv_folder)
            prepared = self._prepare_single_series(df, symbol, horizon, feature_cols)
            if prepared is None:
                continue
            prepared_df, available_features = prepared
            frames.append(prepared_df)
            all_features.extend(available_features)
            self._log(f"   📊 {symbol}_{timeframe}: {len(prepared_df)} rows")

        if not frames:
            raise ValueError(f"No usable series produced for timeframe {timeframe}")

        all_features = list(dict.fromkeys(all_features))
        result = pd.concat(frames, ignore_index=True, sort=False)

        for column in all_features:
            if column not in result.columns:
                result[column] = 0.0

        result[all_features] = (
            result[all_features]
            .replace([np.inf, -np.inf], np.nan)
            .ffill()
            .bfill()
            .fillna(0.0)
            .astype(np.float32)
        )

        result[self.target_column] = (
            result[self.target_column]
            .replace([np.inf, -np.inf], np.nan)
            .ffill()
            .bfill()
            .astype(np.float32)
        )

        result["group_id"] = result["group_id"].astype("category").cat.codes.astype(np.int32)
        result = result.sort_values(["group_id", "time_idx"]).reset_index(drop=True)

        self.feature_cols = all_features
        self._log(f"✅ [{timeframe}] Prepared {len(result):,} rows, {result['group_id'].nunique()} symbols")
        self._log(f"✅ [{timeframe}] Numeric features: {len(self.feature_cols)}")
        return result


class StreamingLightGBMTrainer:
    def __init__(self, log_callback=None, timeframe=None):
        self.log_callback = log_callback or print
        self.timeframe = timeframe
        self.gpu_manager = GPUManager(verbose=False)
        self.model = None
        self.is_trained = False
        self.stop_flag = False
        self.device_used = "cpu"

        tf_params = Config.TIMEFRAME_HYPERPARAMS.get(timeframe, {})
        self.iterations = tf_params.get("iterations", Config.DEFAULT_ITERATIONS)
        self.learning_rate = Config.DEFAULT_LEARNING_RATE
        self.max_depth = tf_params.get("max_depth", Config.DEFAULT_MAX_DEPTH)
        self.num_leaves = Config.DEFAULT_NUM_LEAVES
        self.reg_lambda = tf_params.get("reg_lambda", Config.DEFAULT_L2_REG)
        self.subsample = Config.DEFAULT_SUBSAMPLE
        self.colsample_bytree = Config.DEFAULT_COLSAMPLE
        self.bagging_freq = Config.DEFAULT_BAGGING_FREQ
        self.training_history = {}

        preferred = " -> ".join(self.gpu_manager.preferred_devices())
        tf_label = f" [{timeframe}]" if timeframe else ""
        self._log(f"🚀 LightGBM trainer{tf_label} initialized. Device attempt order: {preferred.upper()}")

    def _log(self, message):
        if self.log_callback:
            self.log_callback(message)

    def stop_training(self):
        self.stop_flag = True
        self._log("⏹️ Training stop requested")

    def _chronological_split(self, data):
        train_parts, val_parts = [], []
        for _, group in data.groupby("group_id"):
            group = group.sort_values("time_idx")
            n_val = max(1, int(len(group) * Config.VALIDATION_FRACTION))
            train_parts.append(group.iloc[:-n_val])
            val_parts.append(group.iloc[-n_val:])
        return pd.concat(train_parts, ignore_index=True), pd.concat(val_parts, ignore_index=True)

    def _compute_directional_accuracy(self, model, X_val, val_df):
        try:
            if "close" not in val_df.columns:
                return None
            predictions = model.predict(X_val)
            last_close = val_df["close"].to_numpy()
            actual_target = val_df["target"].to_numpy()
            actual_direction = np.sign(actual_target - last_close)
            predicted_direction = np.sign(predictions - last_close)
            valid_mask = actual_direction != 0
            if valid_mask.sum() == 0:
                return None
            matches = (actual_direction == predicted_direction) & valid_mask
            return float(matches.sum() / valid_mask.sum())
        except Exception as exc:
            self._log(f"⚠️ Could not compute directional accuracy: {exc}")
            return None

    def _build_model(self, device_type, iterations, learning_rate, max_depth):
        params = dict(
            n_estimators=iterations,
            learning_rate=learning_rate,
            max_depth=max_depth,
            num_leaves=self.num_leaves,
            reg_lambda=self.reg_lambda,
            subsample=self.subsample,
            subsample_freq=self.bagging_freq,
            colsample_bytree=self.colsample_bytree,
            objective="regression",
            metric="rmse",
            random_state=42,
            n_jobs=-1,
            verbosity=-1,
            device_type=device_type,
        )
        if device_type == "gpu":
            params["gpu_platform_id"] = 0
            params["gpu_device_id"] = 0
        return lgb.LGBMRegressor(**params)

    def train_single_timeframe(
        self,
        data,
        feature_cols,
        timeframe,
        iterations=None,
        learning_rate=0.05,
        max_depth=None,
        **kwargs,
    ):
        try:
            self.stop_flag = False
            self.timeframe = timeframe

            tf_params = Config.TIMEFRAME_HYPERPARAMS.get(timeframe, {})
            if iterations is None:
                iterations = tf_params.get("iterations", self.iterations)
            if max_depth is None:
                max_depth = tf_params.get("max_depth", self.max_depth)
            self.reg_lambda = tf_params.get("reg_lambda", self.reg_lambda)

            if len(data) == 0:
                raise ValueError("Prepared dataframe is empty")

            train_df, val_df = self._chronological_split(data)
            if len(train_df) == 0 or len(val_df) == 0:
                raise ValueError("Train/validation split produced an empty set")

            x_columns = feature_cols + ["group_id"]
            X_train = train_df[x_columns].copy()
            y_train = train_df["target"]
            X_val = val_df[x_columns].copy()
            y_val = val_df["target"]

            X_train["group_id"] = X_train["group_id"].astype("category")
            X_val["group_id"] = X_val["group_id"].astype("category")

            self._log(f"📊 [{timeframe}] Training rows: {len(train_df):,}")
            self._log(f"📊 [{timeframe}] Validation rows: {len(val_df):,}")
            self._log(f"⚙️ [{timeframe}] depth={max_depth}, iterations={iterations}, l2={self.reg_lambda}")

            last_error = None
            fitted_model = None
            used_device = None

            for device_type in self.gpu_manager.preferred_devices():
                if self.stop_flag:
                    break
                model = self._build_model(device_type, iterations, learning_rate, max_depth)
                self._log(f"🚀 [{timeframe}] Attempting training on device_type='{device_type}'...")
                try:
                    fitted_model = model.fit(
                        X_train, y_train,
                        eval_set=[(X_val, y_val)],
                        eval_metric="rmse",
                        categorical_feature=["group_id"],
                        callbacks=[
                            lgb.early_stopping(Config.EARLY_STOPPING_ROUNDS, verbose=False),
                            lgb.log_evaluation(period=0),
                        ],
                    )
                    used_device = device_type
                    break
                except Exception as exc:
                    last_error = exc
                    self._log(f"⚠️ [{timeframe}] device_type='{device_type}' failed ({exc}); falling back...")
                    continue

            if fitted_model is None:
                raise last_error or RuntimeError("No device succeeded in training")

            self.model = fitted_model
            self.device_used = used_device
            self._log(f"✅ [{timeframe}] Trained using device_type='{used_device}'")

            best_iteration = getattr(self.model, "best_iteration_", iterations) or iterations
            best_score_dict = getattr(self.model, "best_score_", {})
            best_score = None
            if best_score_dict:
                valid_scores = best_score_dict.get("valid_0", {})
                best_score = valid_scores.get("rmse")

            directional_accuracy = self._compute_directional_accuracy(self.model, X_val, val_df)
            if directional_accuracy is not None:
                self._log(f"🎯 [{timeframe}] Directional accuracy on validation set: {directional_accuracy * 100:.1f}% (50% = coin flip)")

            self.training_history = {
                "timeframe": timeframe,
                "device_used": used_device,
                "best_iteration": best_iteration,
                "best_score": best_score,
                "train_rows": len(train_df),
                "val_rows": len(val_df),
                "directional_accuracy": directional_accuracy,
            }
            self.is_trained = True
            self._log(f"🏁 [{timeframe}] Training complete (device={used_device}, best_iter={best_iteration}, val_rmse={best_score})")
            return True

        except Exception as exc:
            self._log(f"❌ [{timeframe}] Training error: {exc}")
            traceback.print_exc()
            return False

    def save_model(self, model_dir, filename_stem, feature_cols, horizon, timeframe):
        if not self.is_trained or self.model is None:
            self._log("⚠️ No trained model available")
            return False
        try:
            output_dir = Path(model_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            model_path = output_dir / f"{filename_stem}{Config.MODEL_EXT}"
            metadata_path = output_dir / f"{filename_stem}{Config.METADATA_EXT}"

            self.model.booster_.save_model(str(model_path))

            metadata = {
                "model_type": "lightgbm.LGBMRegressor",
                "timeframe": timeframe,
                "model_path": str(model_path),
                "feature_columns": feature_cols,
                "categorical_features": ["group_id"],
                "device_used": self.device_used,
                "horizon": horizon,
                "directional_accuracy": self.training_history.get("directional_accuracy"),
                "training_history": self.training_history,
                "training_time": datetime.now().isoformat(),
            }
            with open(metadata_path, "wb") as file:
                pickle.dump(metadata, file)

            self._log(f"✅ LightGBM model saved: {model_path}")
            self._log(f"✅ Metadata saved: {metadata_path}")
            return True
        except Exception as exc:
            self._log(f"❌ Save error: {exc}")
            traceback.print_exc()
            return False


class LightGBMTrainerGUI(CTk):
    INDICATOR_CONFIG = [
        ("ADX", "enable_adx", True),
        ("ATR", "enable_atr", True),
        ("BB", "enable_bb", True),
        ("CCI", "enable_cci", True),
        ("EMA", "enable_ema", True),
        ("KC", "enable_kc", True),
        ("MACD", "enable_macd", True),
        ("MFI", "enable_mfi", True),
        ("OBV", "enable_obv", True),
        ("ROC", "enable_roc", True),
        ("RSI", "enable_rsi", True),
        ("SMA", "enable_sma", True),
        ("STOCH", "enable_stochastic", True),
        ("VOL_MA", "enable_vol_ma", True),
        ("VWAP", "enable_vwap", True),
        ("WILLR", "enable_willr", True),
    ]

    def __init__(self):
        super().__init__()
        self.title(Config.APP_TITLE)
        self.geometry(Config.APP_GEOMETRY)
        self.resizable(False, False)

        self.gui_logger = GUILogger()
        self.gui_logger.enable_console_redirect()

        self.csv_folder = Config.DATASET_DIR
        self.model_folder = Config.MODEL_DIR
        self.available_symbols = []
        self.available_timeframes = []
        self.is_training = False
        self.stop_flag = False

        self.data_preparer = StreamingDataPreparer(self.gui_logger.log)
        self.current_trainer = None

        self._create_ui()
        self.gui_logger.set_log_widget(self.log_text)
        self.after(500, self.scan_data_source)
        self.protocol("WM_DELETE_WINDOW", self._on_closing)

    def _create_ui(self):
        main_frame = CTkFrame(self, fg_color="transparent")
        main_frame.pack(fill="both", expand=True, padx=10, pady=8)

        title_frame = CTkFrame(main_frame, corner_radius=8)
        title_frame.pack(fill="x", pady=(0, 5))

        CTkLabel(title_frame, text="🎯 LightGBM Model Trainer (per timeframe)",
                 font=("Ubuntu", 17, "bold")).pack(pady=6)
        CTkLabel(title_frame, text="BSD-C-3 License", font=("Ubuntu", 18), text_color="yellow").pack(pady=(0, 4))

        self.usage_row = UsageRow(main_frame)

        self._create_csv_section(main_frame)
        self._create_model_folder_section(main_frame)
        self._create_parameters_section(main_frame)
        self._create_features_section(main_frame)
        self._create_timeframes_section(main_frame)
        self._create_indicators_section(main_frame)
        self._create_controls_section(main_frame)
        self._create_status_section(main_frame)
        self._create_log_section(main_frame)

    def _create_csv_section(self, parent):
        frame = CTkFrame(parent, corner_radius=8)
        frame.pack(fill="x", pady=(0, 5))
        folder_frame = CTkFrame(frame, fg_color="transparent")
        folder_frame.pack(fill="x", padx=10, pady=4)
        self.select_csv_btn = CTkButton(folder_frame, text="📁 CSV Folder",
                                         command=self.select_csv_folder, width=130, height=30)
        self.select_csv_btn.pack(side="left", padx=(0, 8))
        self.csv_path_var = tk.StringVar(value=str(self.csv_folder))
        csv_entry = CTkEntry(folder_frame, textvariable=self.csv_path_var, state="readonly", height=30)
        csv_entry.pack(side="left", fill="x", expand=True)

    def _create_model_folder_section(self, parent):
        frame = CTkFrame(parent, corner_radius=8)
        frame.pack(fill="x", pady=(0, 5))
        folder_frame = CTkFrame(frame, fg_color="transparent")
        folder_frame.pack(fill="x", padx=10, pady=4)
        self.select_model_btn = CTkButton(folder_frame, text="📁 Model Folder",
                                           command=self.select_model_folder, width=130, height=30)
        self.select_model_btn.pack(side="left", padx=(0, 8))
        self.model_path_var = tk.StringVar(value=str(self.model_folder))
        model_entry = CTkEntry(folder_frame, textvariable=self.model_path_var, state="readonly", height=30)
        model_entry.pack(side="left", fill="x", expand=True)
        self.status_label = CTkLabel(frame, text="📊 Status: No folder selected", font=("Ubuntu", 11))
        self.status_label.pack(anchor="w", padx=10, pady=2)

    def _create_parameters_section(self, parent):
        frame = CTkFrame(parent, corner_radius=8)
        frame.pack(fill="x", pady=(0, 5))
        param_frame = CTkFrame(frame, fg_color="transparent")
        param_frame.pack(fill="x", padx=10, pady=4)

        self.horizon = tk.StringVar(value=str(Config.DEFAULT_HORIZON))
        self.iterations = tk.StringVar(value=str(Config.DEFAULT_ITERATIONS))
        self.learning_rate = tk.StringVar(value=str(Config.DEFAULT_LEARNING_RATE))
        self.max_depth = tk.StringVar(value=str(Config.DEFAULT_MAX_DEPTH))

        row = CTkFrame(param_frame, fg_color="transparent")
        row.pack(fill="x", pady=2)
        for col in range(8):
            row.grid_columnconfigure(col, weight=1 if col % 2 else 0)

        CTkLabel(row, text="Horizon:", width=80).grid(row=0, column=0, padx=(0, 4), sticky="w")
        CTkEntry(row, textvariable=self.horizon, width=70).grid(row=0, column=1, padx=(0, 12), sticky="w")
        CTkLabel(row, text="Iterations:", width=80).grid(row=0, column=2, padx=(0, 4), sticky="w")
        CTkEntry(row, textvariable=self.iterations, width=70).grid(row=0, column=3, padx=(0, 12), sticky="w")
        CTkLabel(row, text="LR:", width=80).grid(row=0, column=4, padx=(0, 4), sticky="w")
        CTkEntry(row, textvariable=self.learning_rate, width=70).grid(row=0, column=5, padx=(0, 12), sticky="w")
        CTkLabel(row, text="Depth:", width=80).grid(row=0, column=6, padx=(0, 4), sticky="w")
        CTkEntry(row, textvariable=self.max_depth, width=70).grid(row=0, column=7, sticky="w")

        self.auto_tune_var = tk.BooleanVar(value=True)
        ctk.CTkCheckBox(
            param_frame,
            text="Auto-tune depth / iterations / L2 per timeframe (recommended)",
            variable=self.auto_tune_var,
        ).pack(anchor="w", pady=(6, 0))

    def _create_features_section(self, parent):
        frame = CTkFrame(parent, corner_radius=8)
        frame.pack(fill="x", pady=(0, 5))
        feature_frame = CTkFrame(frame, fg_color="transparent")
        feature_frame.pack(fill="x", padx=10, pady=4)
        default_features = "time,open,high,low,close,volume"
        self.feature_cols_var = tk.StringVar(value=default_features)
        CTkLabel(feature_frame, text="Base Features (comma-separated):", font=("Ubuntu", 11)).pack(anchor="w", pady=(0, 3))
        feature_entry = CTkEntry(feature_frame, textvariable=self.feature_cols_var, height=30)
        feature_entry.pack(fill="x", pady=(0, 3))

    def _create_timeframes_section(self, parent):
        frame = CTkFrame(parent, corner_radius=8)
        frame.pack(fill="x", pady=(0, 5))
        tf_frame = CTkFrame(frame, fg_color="transparent")
        tf_frame.pack(fill="x", padx=10, pady=4)
        CTkLabel(tf_frame, text="Timeframes (one model each):", font=("Ubuntu", 11)).pack(anchor="w", pady=(0, 3))
        self.tf_vars = {}
        checkboxes_frame = CTkFrame(tf_frame, fg_color="transparent")
        checkboxes_frame.pack(fill="x")
        for i, tf in enumerate(Config.TIMEFRAMES):
            var = tk.BooleanVar(value=True)
            self.tf_vars[tf] = var
            ctk.CTkCheckBox(checkboxes_frame, text=tf, variable=var, width=80).grid(
                row=0, column=i, padx=4, sticky="w"
            )

    def _create_indicators_section(self, parent):
        frame = CTkFrame(parent, corner_radius=8)
        frame.pack(fill="x", pady=(0, 5))
        indicator_frame = CTkFrame(frame, fg_color="transparent")
        indicator_frame.pack(fill="x", padx=10, pady=4)
        CTkLabel(indicator_frame, text="Technical Indicators (A-Z, 6 per row):", font=("Ubuntu", 11)).pack(anchor="w", pady=(0, 3))

        self.indicator_vars = {}
        rows = [self.INDICATOR_CONFIG[i:i+6] for i in range(0, len(self.INDICATOR_CONFIG), 6)]
        for r_idx, row_items in enumerate(rows):
            row_frame = CTkFrame(indicator_frame, fg_color="transparent")
            row_frame.pack(fill="x", pady=1)
            for c_idx, (label, var_name, default) in enumerate(row_items):
                var = tk.BooleanVar(value=default)
                self.indicator_vars[var_name] = var
                ctk.CTkCheckBox(
                    row_frame,
                    text=label,
                    variable=var,
                    width=90,
                    font=("Ubuntu", 11),
                ).grid(row=0, column=c_idx, padx=6, sticky="w")

        CTkLabel(
            indicator_frame,
            text="Checked = computed on-the-fly and fed into model. Unchecked = skipped.",
            font=("Ubuntu", 9),
            text_color="gray"
        ).pack(anchor="w", pady=(2, 0))

    def _create_controls_section(self, parent):
        frame = CTkFrame(parent, corner_radius=8)
        frame.pack(fill="x", pady=(0, 5))
        button_frame = CTkFrame(frame, fg_color="transparent")
        button_frame.pack(pady=6)
        self.train_btn = CTkButton(button_frame, text="🚀 Start Training",
                                    command=self.start_training, width=140, height=38,
                                    fg_color="#2e7d32", hover_color="#1b5e20",
                                    font=("Ubuntu", 13, "bold"))
        self.train_btn.pack(side="left", padx=5)
        self.stop_btn = CTkButton(button_frame, text="⏹️ Stop Training",
                                   command=self.stop_training, width=140, height=38,
                                   fg_color="#c62828", hover_color="#b71c1c",
                                   font=("Ubuntu", 13, "bold"), state="disabled")
        self.stop_btn.pack(side="left", padx=5)

    def _create_status_section(self, parent):
        frame = CTkFrame(parent, corner_radius=8)
        frame.pack(fill="x", pady=(0, 5))
        self.status_text = CTkTextbox(frame, height=60, font=("Ubuntu Mono", 11))
        self.status_text.pack(fill="x", padx=10, pady=4)
        self.status_text.insert("1.0", "Ready")
        self.status_text.configure(state="disabled")

    def _create_log_section(self, parent):
        frame = CTkFrame(parent, corner_radius=8)
        frame.pack(fill="both", expand=True)
        self.log_text = CTkTextbox(frame, font=("Ubuntu Mono", 11))
        self.log_text.pack(fill="both", expand=True, padx=10, pady=4)

    def update_status(self, message):
        self.status_text.configure(state="normal")
        self.status_text.delete("1.0", "end")
        self.status_text.insert("1.0", message)
        self.status_text.configure(state="disabled")
        self.status_text.update_idletasks()

    def select_csv_folder(self):
        folder = filedialog.askdirectory(title="Select CSV Data Folder", initialdir=str(self.csv_folder))
        if folder:
            self.csv_path_var.set(folder)
            self.scan_data_source()

    def select_model_folder(self):
        folder = filedialog.askdirectory(title="Select Model Save Folder", initialdir=str(self.model_folder))
        if folder:
            self.model_path_var.set(folder)
            global MODEL_DIR
            MODEL_DIR = Path(folder)
            MODEL_DIR.mkdir(parents=True, exist_ok=True)
            self.gui_logger.log(f"💾 Model folder set to: {MODEL_DIR}")

    def scan_data_source(self):
        csv_folder = Path(self.csv_path_var.get())
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
        self.available_symbols = [s[0] for s in symbols]
        self.available_timeframes = timeframes
        self.status_label.configure(text=f"✅ Found {len(symbols)} symbols, {len(timeframes)} timeframes")
        self.gui_logger.log(f"✅ Loaded {len(symbols)} symbols", "success")
        self.train_btn.configure(state="normal")

    def start_training(self):
        if self.is_training:
            self.gui_logger.log("⏳ Training already in progress", "warning")
            return

        csv_folder = self.csv_path_var.get()
        if not Path(csv_folder).exists():
            self.gui_logger.log("❌ Invalid CSV folder", "error")
            return

        selected_timeframes = [tf for tf, var in self.tf_vars.items() if var.get()]
        if not selected_timeframes:
            self.gui_logger.log("❌ Select at least one timeframe", "error")
            return

        self.is_training = True
        self.stop_flag = False
        self.train_btn.configure(state="disabled")
        self.stop_btn.configure(state="normal")

        def training_task():
            try:
                self.gui_logger.log("=" * 60)
                self.gui_logger.log("🚀 Starting Per-Timeframe LightGBM Training")
                self.gui_logger.log("=" * 60)
                self.gui_logger.log(f"📂 CSV Folder: {csv_folder}")
                self.gui_logger.log(f"💾 System RAM: {MemoryOptimizer.get_system_memory_available_gb():.1f} GB available")

                flags = {name: var.get() for name, var in self.indicator_vars.items()}
                enabled = [name.replace("enable_", "").upper() for name, val in flags.items() if val]
                self.gui_logger.log(f"📊 Indicators enabled ({len(enabled)}): {', '.join(enabled)}")

                self.data_preparer.set_indicator_flags(**flags)

                feature_input = self.feature_cols_var.get().strip()
                if feature_input:
                    feature_cols = [f.strip() for f in feature_input.split(',') if f.strip()]
                else:
                    feature_cols = Config.DEFAULT_FEATURES

                horizon = int(self.horizon.get())
                learning_rate = float(self.learning_rate.get())
                gui_iterations = int(self.iterations.get())
                gui_max_depth = int(self.max_depth.get())
                auto_tune = self.auto_tune_var.get()
                model_folder = self.model_path_var.get()

                results = {}

                for timeframe in selected_timeframes:
                    if self.stop_flag:
                        self.gui_logger.log("⏹️ Training stopped by user", "warning")
                        break

                    self.gui_logger.log("-" * 60)
                    self.gui_logger.log(f"▶️ Timeframe: {timeframe}")
                    self.update_status(f"[{timeframe}] Building dataset...")

                    try:
                        data = self.data_preparer.build_dataframe_for_timeframe(
                            csv_folder=csv_folder,
                            feature_cols=feature_cols,
                            horizon=horizon,
                            timeframe=timeframe,
                        )
                    except ValueError as exc:
                        self.gui_logger.log(f"⚠️ [{timeframe}] Skipped: {exc}", "warning")
                        results[timeframe] = "skipped (no data)"
                        continue

                    feature_columns = self.data_preparer.feature_cols
                    self.update_status(f"[{timeframe}] Training LightGBM model...")

                    trainer = StreamingLightGBMTrainer(self.gui_logger.log, timeframe=timeframe)
                    self.current_trainer = trainer

                    success = trainer.train_single_timeframe(
                        data=data,
                        feature_cols=feature_columns,
                        timeframe=timeframe,
                        iterations=None if auto_tune else gui_iterations,
                        learning_rate=learning_rate,
                        max_depth=None if auto_tune else gui_max_depth,
                    )

                    if success and not trainer.stop_flag:
                        filename_stem = f"{Config.MODEL_PREFIX}_{timeframe}"
                        if trainer.save_model(
                            model_dir=model_folder,
                            filename_stem=filename_stem,
                            feature_cols=feature_columns,
                            horizon=horizon,
                            timeframe=timeframe,
                        ):
                            self.gui_logger.log(f"✅ [{timeframe}] Model saved: {filename_stem}{Config.MODEL_EXT}", "success")
                            results[timeframe] = f"trained ({trainer.device_used})"
                        else:
                            results[timeframe] = "failed to save"
                    elif trainer.stop_flag:
                        results[timeframe] = "stopped"
                        self.stop_flag = True
                    else:
                        results[timeframe] = "training failed"

                    del trainer
                    self.current_trainer = None
                    MemoryOptimizer.clear_cache()

                summary = ", ".join(f"{tf}: {status}" for tf, status in results.items())
                self.gui_logger.log("=" * 60)
                self.gui_logger.log(f"🏁 Summary: {summary}")
                self.update_status(f"Done. {summary}")

            except Exception as e:
                self.gui_logger.log(f"❌ Training error: {e}", "error")
                traceback.print_exc()
                self.update_status(f"Error: {e}")
            finally:
                self.is_training = False
                self.current_trainer = None
                self.train_btn.configure(state="normal")
                self.stop_btn.configure(state="disabled")

        threading.Thread(target=training_task, daemon=True).start()

    def stop_training(self):
        self.stop_flag = True
        if self.current_trainer is not None:
            self.current_trainer.stop_training()
        self.gui_logger.log("⏹️ Stopping training...", "warning")

    def _on_closing(self):
        self.gui_logger.disable_console_redirect()
        if self.is_training:
            result = messagebox.askyesno(
                "Training in Progress",
                "Training is currently running.\n\nDo you want to stop training and quit?",
                icon='warning',
            )
            if result:
                self.stop_flag = True
                if self.current_trainer is not None:
                    self.current_trainer.stop_training()
                self.after(100, self.destroy)
        else:
            result = messagebox.askyesno("Quit", "Do you want to quit AI Predict Train (LightGBM)?")
            if result:
                self.destroy()


def main():
    app = LightGBMTrainerGUI()
    app.mainloop()


if __name__ == "__main__":
    main()