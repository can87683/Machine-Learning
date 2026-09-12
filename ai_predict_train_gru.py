#!/usr/bin/env python3
# ai_predict_train_gru.py
# Copyright Su Nie | BSD-3C License | https://github.com/can87683

import setproctitle
setproctitle.setproctitle("ai_predict_train_gru_ctk")

import customtkinter as ctk
from customtkinter import CTk, CTkFrame, CTkLabel, CTkButton, CTkEntry, CTkTextbox
import tkinter as tk
from tkinter import filedialog, messagebox
import threading
import subprocess
import copy
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

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Suppress FutureWarnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


# ===== Configuration Class =====
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

    # Model settings (GRU)
    DEFAULT_HORIZON = 10
    DEFAULT_SEQUENCE_LENGTH = 30        # lookback window fed to the GRU
    DEFAULT_EPOCHS = 100
    DEFAULT_LEARNING_RATE = 0.001
    DEFAULT_HIDDEN_SIZE = 64
    DEFAULT_NUM_LAYERS = 2
    DEFAULT_DROPOUT = 0.2
    DEFAULT_EMBEDDING_DIM = 8           # per-symbol embedding size
    DEFAULT_BATCH_SIZE = 128
    GRADIENT_CLIP_NORM = 5.0

    # Training settings
    EARLY_STOPPING_PATIENCE = 10        # epochs without val improvement
    VALIDATION_FRACTION = 0.15          # last X% of each symbol's series, chronologically

    # Memory settings
    MEMORY_WARNING_THRESHOLD_GB = 0.5

    # Timeframes -> one independent model is trained per timeframe
    TIMEFRAMES = ['M5', 'M15', 'H1', 'H4', 'D1', 'W1']

    # Output naming
    MODEL_PREFIX = "ai_predict_gru"
    MODEL_EXT = ".pt"            # PyTorch state dict
    METADATA_EXT = ".pkl"

    # UI settings
    APP_TITLE = "🚀 GRU Model Trainer"
    APP_GEOMETRY = "640x1050"
    THEME = "dark"
    COLOR_THEME = "dark-blue"

    @classmethod
    def ensure_directories(cls):
        cls.MODEL_DIR.mkdir(parents=True, exist_ok=True)
        cls.DATASET_DIR.mkdir(parents=True, exist_ok=True)


Config.ensure_directories()

# ===== CTK Theme Settings =====
ctk.set_appearance_mode(Config.THEME)
ctk.set_default_color_theme(Config.COLOR_THEME)

# ===== Constants =====
HOME = Config.HOME
MODEL_DIR = Config.MODEL_DIR
DATASET_DIR = Config.DATASET_DIR

torch.manual_seed(42)


class GPUManager:
    def __init__(self, verbose=True):
        self.verbose = verbose
        self.gpu_available = False
        self.gpu_name = None
        self._detect_gpu()

        self.torch_cuda_available = torch.cuda.is_available()
        if self.torch_cuda_available:
            torch.backends.cudnn.benchmark = True

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
        if self.gpu_available and self.torch_cuda_available:
            return [torch.device("cuda"), torch.device("cpu")]
        return [torch.device("cpu")]


class UsageRow:
    def __init__(self, parent):
        self.parent = parent
        self.usage_frame = None
        self.usage_label = None
        self._create_usage_row()

    def _create_usage_row(self):
        self.usage_frame = CTkFrame(self.parent, height=40, corner_radius=5)
        self.usage_frame.pack(fill="x", pady=(0, 10))

        self.usage_label = CTkLabel(
            self.usage_frame,
            text="Loading system usage...",
            font=("Ubuntu Mono", 20),
            text_color="Lime"
        )
        self.usage_label.pack(expand=True, fill="x", padx=10, pady=5)

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
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @staticmethod
    def monitor_memory(threshold_gb=0.5):
        return MemoryOptimizer.get_system_memory_available_gb() < threshold_gb


class FeatureEngineer:
    """Extract time-based and comprehensive technical features"""

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

        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

        return df

    @staticmethod
    def _add_atr(df, period=14):
        prev_close = df['close'].shift(1)
        true_range = pd.concat([
            df['high'] - df['low'],
            (df['high'] - prev_close).abs(),
            (df['low'] - prev_close).abs(),
        ], axis=1).max(axis=1)
        df['atr'] = true_range.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
        return df

    @staticmethod
    def _add_bollinger(df, period=20, std_dev=2):
        mid = df['close'].rolling(period).mean()
        std = df['close'].rolling(period).std()
        upper = mid + std_dev * std
        lower = mid - std_dev * std
        df['bb_mid'] = mid
        band_range = (upper - lower).replace(0, np.nan)
        df['bb_percent_b'] = (df['close'] - lower) / band_range
        df['bb_bandwidth'] = band_range / mid
        return df

    @staticmethod
    def _add_cci(df, period=20):
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        ma_tp = typical_price.rolling(period).mean()
        md = typical_price.rolling(period).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True)
        df['cci'] = (typical_price - ma_tp) / (0.015 * md.replace(0, np.nan))
        return df

    @staticmethod
    def _add_cmf(df, period=20):
        mfv = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low']).replace(0, np.nan) * df['volume']
        df['cmf'] = mfv.rolling(period).sum() / df['volume'].rolling(period).sum().replace(0, np.nan)
        return df

    @staticmethod
    def _add_donchian(df, period=20):
        high_20 = df['high'].rolling(period).max()
        low_20 = df['low'].rolling(period).min()
        range_20 = (high_20 - low_20).replace(0, np.nan)
        df['donchian_position'] = (df['close'] - low_20) / range_20
        df['donchian_width'] = range_20 / df['close']
        return df

    @staticmethod
    def _add_ema(df):
        for n in [9, 21, 50, 200]:
            ema = df['close'].ewm(span=n, adjust=False).mean()
            df[f'close_vs_ema_{n}'] = df['close'] / ema - 1
        df['ema_9_21_spread'] = df['close_vs_ema_9'] - df['close_vs_ema_21']
        df['ema_21_50_spread'] = df['close_vs_ema_21'] - df['close_vs_ema_50']
        df['ema_21_slope_5'] = df['close'].ewm(span=21, adjust=False).mean().diff(5) / df['close']
        return df

    @staticmethod
    def _add_force_index(df, period=13):
        fi = df['close'].diff() * df['volume']
        df['force_index'] = fi.ewm(span=period, adjust=False).mean()
        return df

    @staticmethod
    def _add_macd(df, fast=12, slow=26, signal=9):
        ema_fast = df['close'].ewm(span=fast, adjust=False).mean()
        ema_slow = df['close'].ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        df['macd'] = macd_line
        df['macd_signal'] = signal_line
        df['macd_hist'] = macd_line - signal_line
        return df

    @staticmethod
    def _add_mfi(df, period=14):
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        raw_money_flow = typical_price * df['volume']
        positive_flow = raw_money_flow.where(typical_price > typical_price.shift(1), 0.0)
        negative_flow = raw_money_flow.where(typical_price < typical_price.shift(1), 0.0)
        positive_sum = positive_flow.rolling(period).sum()
        negative_sum = negative_flow.rolling(period).sum()
        money_flow_ratio = positive_sum / negative_sum.replace(0, np.nan)
        df['mfi'] = 100 - (100 / (1 + money_flow_ratio))
        return df

    @staticmethod
    def _add_obv(df):
        price_direction = np.sign(df['close'].diff()).fillna(0)
        df['obv'] = (price_direction * df['volume']).cumsum()
        rolling_vol = df['volume'].rolling(20).sum().replace(0, np.nan)
        df['obv_slope_5'] = df['obv'].diff(5) / rolling_vol
        return df

    @staticmethod
    def _add_price_structure(df):
        high = df['high']
        low = df['low']
        open_ = df['open']
        close = df['close']
        range_hl = (high - low).replace(0, np.nan)
        df['hl_range_pct'] = range_hl / close
        df['body_pct'] = (close - open_).abs() / close
        df['body_range_ratio'] = (close - open_).abs() / range_hl
        df['close_location_value'] = (close - low) / range_hl
        open_close_max = df[['open', 'close']].max(axis=1)
        open_close_min = df[['open', 'close']].min(axis=1)
        df['upper_shadow_pct'] = (high - open_close_max) / close
        df['lower_shadow_pct'] = (open_close_min - low) / close
        return df

    @staticmethod
    def _add_realized_vol(df):
        ret_1 = np.log(df['close'] / df['close'].shift(1))
        df['realized_vol_20'] = ret_1.rolling(20).std()
        df['realized_vol_100'] = ret_1.rolling(100).std()
        return df

    @staticmethod
    def _add_roc(df):
        for n in [5, 10, 20]:
            df[f'roc_{n}'] = df['close'].pct_change(n) * 100
        return df

    @staticmethod
    def _add_rsi(df, period=14):
        delta = df['close'].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        df['rsi'] = 100 - (100 / (1 + rs))
        return df

    @staticmethod
    def _add_rvwap(df, period=20):
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        rolling_tp_vol = (typical_price * df['volume']).rolling(period).sum()
        rolling_vol = df['volume'].rolling(period).sum().replace(0, np.nan)
        df['rvwap'] = rolling_tp_vol / rolling_vol
        df['close_vs_rvwap'] = df['close'] / df['rvwap'] - 1
        return df

    @staticmethod
    def _add_stochastic(df, period=14, smooth=3):
        lowest_low = df['low'].rolling(period).min()
        highest_high = df['high'].rolling(period).max()
        denom = (highest_high - lowest_low).replace(0, np.nan)
        percent_k = 100 * (df['close'] - lowest_low) / denom
        df['stoch_k'] = percent_k
        df['stoch_d'] = percent_k.rolling(smooth).mean()
        return df

    @staticmethod
    def _add_volume_metrics(df):
        volume_ma_20 = df['volume'].rolling(20).mean()
        volume_std_20 = df['volume'].rolling(20).std()
        df['volume_ratio_20'] = df['volume'] / volume_ma_20.replace(0, np.nan)
        df['volume_zscore_20'] = (df['volume'] - volume_ma_20) / volume_std_20.replace(0, np.nan)
        df['return_x_volume_1'] = df['close'].pct_change() * df['volume']
        dollar_volume = (df['close'] * df['volume']).replace(0, np.nan)
        df['amihud_illiquidity_20'] = (df['close'].pct_change().abs() / dollar_volume).rolling(20).mean()
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

        self.enable_atr = True
        self.enable_bollinger = True
        self.enable_cci = True
        self.enable_cmf = True
        self.enable_donchian = True
        self.enable_ema = True
        self.enable_force_index = True
        self.enable_macd = True
        self.enable_mfi = True
        self.enable_obv = True
        self.enable_price_structure = True
        self.enable_realized_vol = True
        self.enable_roc = True
        self.enable_rsi = True
        self.enable_rvwap = True
        self.enable_stochastic = True
        self.enable_volume_metrics = True

    def _log(self, message):
        if self.log_callback:
            self.log_callback(message)

    def set_indicator_flags(self, **kwargs):
        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, v)

    def load_csv_data(self, symbol, timeframe, csv_folder):
        csv_file = Path(csv_folder) / symbol / f"{symbol}_{timeframe}.csv"
        if not csv_file.exists(): return None
        try:
            df = pd.read_csv(csv_file)
            if "time" not in df.columns: return None
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
        if not csv_folder.exists(): return [], []
        symbols = []
        all_timeframes = set()
        for symbol_dir in sorted(csv_folder.iterdir()):
            if not symbol_dir.is_dir(): continue
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
        if df is None or len(df) < 100: return None
        df = df.copy()
        if len(df) > Config.MAX_ROWS_PER_SYMBOL:
            df = df.iloc[-Config.MAX_ROWS_PER_SYMBOL:].copy()

        df = FeatureEngineer.add_time_features(df)

        engineered_features = [
            "hour_sin", "hour_cos", "day_sin", "day_cos", "month_sin", "month_cos",
            "returns_1", "returns_5", "returns_10"
        ]

        df['returns_1'] = df['close'].pct_change()
        df['returns_5'] = df['close'].pct_change(periods=5)
        df['returns_10'] = df['close'].pct_change(periods=10)

        if self.enable_atr:
            df = FeatureEngineer._add_atr(df)
            engineered_features.append("atr")
        if self.enable_bollinger:
            df = FeatureEngineer._add_bollinger(df)
            engineered_features.extend(["bb_percent_b", "bb_bandwidth"])
        if self.enable_cci:
            df = FeatureEngineer._add_cci(df)
            engineered_features.append("cci")
        if self.enable_cmf:
            df = FeatureEngineer._add_cmf(df)
            engineered_features.append("cmf")
        if self.enable_donchian:
            df = FeatureEngineer._add_donchian(df)
            engineered_features.extend(["donchian_position", "donchian_width"])
        if self.enable_ema:
            df = FeatureEngineer._add_ema(df)
            engineered_features.extend(["close_vs_ema_9", "close_vs_ema_21", "close_vs_ema_50", "close_vs_ema_200", "ema_9_21_spread", "ema_21_50_spread", "ema_21_slope_5"])
        if self.enable_force_index:
            df = FeatureEngineer._add_force_index(df)
            engineered_features.append("force_index")
        if self.enable_macd:
            df = FeatureEngineer._add_macd(df)
            engineered_features.extend(["macd", "macd_signal", "macd_hist"])
        if self.enable_mfi:
            df = FeatureEngineer._add_mfi(df)
            engineered_features.append("mfi")
        if self.enable_obv:
            df = FeatureEngineer._add_obv(df)
            engineered_features.append("obv_slope_5")
        if self.enable_price_structure:
            df = FeatureEngineer._add_price_structure(df)
            engineered_features.extend(["hl_range_pct", "body_pct", "body_range_ratio", "close_location_value", "upper_shadow_pct", "lower_shadow_pct"])
        if self.enable_realized_vol:
            df = FeatureEngineer._add_realized_vol(df)
            engineered_features.extend(["realized_vol_20", "realized_vol_100"])
        if self.enable_roc:
            df = FeatureEngineer._add_roc(df)
            engineered_features.extend(["roc_5", "roc_10", "roc_20"])
        if self.enable_rsi:
            df = FeatureEngineer._add_rsi(df)
            engineered_features.append("rsi")
        if self.enable_rvwap:
            df = FeatureEngineer._add_rvwap(df)
            engineered_features.append("close_vs_rvwap")
        if self.enable_stochastic:
            df = FeatureEngineer._add_stochastic(df)
            engineered_features.extend(["stoch_k", "stoch_d"])
        if self.enable_volume_metrics:
            df = FeatureEngineer._add_volume_metrics(df)
            engineered_features.extend(["volume_ratio_20", "volume_zscore_20", "return_x_volume_1", "amihud_illiquidity_20"])

        requested_features = [c for c in requested_features if c != "time"]
        candidate_features = list(dict.fromkeys(requested_features + engineered_features))
        available_features = [c for c in candidate_features if c in df.columns]

        required_columns = ["open", "high", "low", "close", "volume"]
        for column in required_columns:
            if column in df.columns and column not in available_features:
                available_features.append(column)

        if "close" not in df.columns: return None
        if not available_features: return None

        df[available_features] = df[available_features].replace([np.inf, -np.inf], np.nan).ffill().bfill().fillna(0.0)
        df[self.target_column] = df["close"].shift(-horizon)
        df = df.dropna(subset=[self.target_column]).copy()

        if len(df) < 50: return None

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
            if prepared is None: continue
            prepared_df, available_features = prepared
            frames.append(prepared_df)
            all_features.extend(available_features)
            self._log(f"   📊 {symbol}_{timeframe}: {len(prepared_df)} rows")

        if not frames:
            raise ValueError(f"No usable series produced for timeframe {timeframe}")

        all_features = list(dict.fromkeys(all_features))
        result = pd.concat(frames, ignore_index=True, sort=False)
        for column in all_features:
            if column not in result.columns: result[column] = 0.0

        result[all_features] = result[all_features].replace([np.inf, -np.inf], np.nan).ffill().bfill().fillna(0.0).astype(np.float32)
        result[self.target_column] = result[self.target_column].replace([np.inf, -np.inf], np.nan).ffill().bfill().astype(np.float32)

        result["group_id"] = result["group_id"].astype("category")
        self.group_id_categories = list(result["group_id"].cat.categories)
        result["group_id"] = result["group_id"].cat.codes.astype(np.int64)
        result = result.sort_values(["group_id", "time_idx"]).reset_index(drop=True)

        self.feature_cols = all_features
        self._log(f"✅ [{timeframe}] Prepared {len(result):,} rows, {result['group_id'].nunique()} symbols")
        self._log(f"✅ [{timeframe}] Numeric features: {len(self.feature_cols)}")
        return result


class SequenceDataset(Dataset):
    def __init__(self, X, group_ids, y):
        self.X = torch.from_numpy(X)
        self.group_ids = torch.from_numpy(group_ids).long()
        self.y = torch.from_numpy(y)

    def __len__(self): return len(self.y)
    def __getitem__(self, idx): return self.X[idx], self.group_ids[idx], self.y[idx]


class GRURegressor(nn.Module):
    def __init__(self, num_features, hidden_size, num_layers, dropout, num_groups, embedding_dim):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=max(num_groups, 1), embedding_dim=embedding_dim)
        self.gru = nn.GRU(
            input_size=num_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size + embedding_dim, max(hidden_size // 2, 8)),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(max(hidden_size // 2, 8), 1),
        )

    def forward(self, x_seq, group_ids):
        _, h_n = self.gru(x_seq)
        last_hidden = h_n[-1]
        emb = self.embedding(group_ids)
        combined = torch.cat([last_hidden, emb], dim=1)
        return self.head(combined).squeeze(-1)


class StreamingGRUTrainer:
    def __init__(self, log_callback=None, timeframe=None):
        self.log_callback = log_callback or print
        self.timeframe = timeframe
        self.gpu_manager = GPUManager(verbose=False)
        self.model = None
        self.is_trained = False
        self.stop_flag = False
        self.device_used = "cpu"

        self.sequence_length = Config.DEFAULT_SEQUENCE_LENGTH
        self.epochs = Config.DEFAULT_EPOCHS
        self.learning_rate = Config.DEFAULT_LEARNING_RATE
        self.hidden_size = Config.DEFAULT_HIDDEN_SIZE
        self.num_layers = Config.DEFAULT_NUM_LAYERS
        self.dropout = Config.DEFAULT_DROPOUT
        self.embedding_dim = Config.DEFAULT_EMBEDDING_DIM
        self.batch_size = Config.DEFAULT_BATCH_SIZE

        self.training_history = {}
        self.scaler_state = {}

        preferred = " -> ".join(str(d) for d in self.gpu_manager.preferred_devices())
        tf_label = f" [{timeframe}]" if timeframe else ""
        self._log(f"🚀 GRU trainer{tf_label} initialized. Device attempt order: {preferred.upper()}")

    def _log(self, message):
        if self.log_callback: self.log_callback(message)

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

    def _fit_scalers(self, train_df, feature_cols):
        feature_mean = train_df[feature_cols].mean().to_numpy(dtype=np.float64)
        feature_std = train_df[feature_cols].std().replace(0, 1.0).to_numpy(dtype=np.float64)
        target_mean = float(train_df["target"].mean())
        target_std = float(train_df["target"].std()) or 1.0
        return {"feature_mean": feature_mean, "feature_std": feature_std, "target_mean": target_mean, "target_std": target_std}

    def _apply_scalers(self, df, feature_cols, scaler_state):
        df = df.copy()
        df["raw_close"] = df["close"]
        df["raw_target"] = df["target"]
        df[feature_cols] = (df[feature_cols].to_numpy(dtype=np.float64) - scaler_state["feature_mean"]) / scaler_state["feature_std"]
        df["target"] = (df["target"].to_numpy(dtype=np.float64) - scaler_state["target_mean"]) / scaler_state["target_std"]
        return df

    def _build_sequences_for_group(self, group_df, feature_cols, seq_len, context_df=None):
        if context_df is not None and len(context_df) > 0:
            combined = pd.concat([context_df, group_df], ignore_index=True)
            start_offset = len(context_df)
        else:
            combined = group_df
            start_offset = 0

        features = combined[feature_cols].to_numpy(dtype=np.float32)
        targets = combined["target"].to_numpy(dtype=np.float32)
        has_raw = "raw_close" in combined.columns and "raw_target" in combined.columns
        raw_close = combined["raw_close"].to_numpy(dtype=np.float32) if has_raw else None
        raw_target = combined["raw_target"].to_numpy(dtype=np.float32) if has_raw else None
        n = len(combined)
        if n < seq_len: return None

        first_target_idx = max(seq_len - 1, start_offset)
        X_list, y_list, raw_close_list, raw_target_list = [], [], [], []
        for i in range(first_target_idx, n):
            X_list.append(features[i - seq_len + 1: i + 1])
            y_list.append(targets[i])
            if has_raw:
                raw_close_list.append(raw_close[i])
                raw_target_list.append(raw_target[i])

        if not X_list: return None
        X = np.stack(X_list).astype(np.float32)
        y = np.array(y_list, dtype=np.float32)
        raw_close_arr = np.array(raw_close_list, dtype=np.float32) if has_raw else None
        raw_target_arr = np.array(raw_target_list, dtype=np.float32) if has_raw else None
        return X, y, raw_close_arr, raw_target_arr

    def _build_sequence_arrays(self, train_df, val_df, feature_cols, seq_len):
        X_train_list, y_train_list, g_train_list = [], [], []
        X_val_list, y_val_list, g_val_list = [], [], []
        raw_close_val_list, raw_target_val_list = [], []

        train_groups = dict(tuple(train_df.groupby("group_id")))
        val_groups = dict(tuple(val_df.groupby("group_id")))

        for group_id, group_df in train_groups.items():
            built = self._build_sequences_for_group(group_df.sort_values("time_idx"), feature_cols, seq_len)
            if built is None: continue
            X, y, _, _ = built
            X_train_list.append(X); y_train_list.append(y); g_train_list.append(np.full(len(y), group_id, dtype=np.int64))

        for group_id, group_df in val_groups.items():
            context = train_groups.get(group_id)
            context_tail = context.sort_values("time_idx").iloc[-(seq_len - 1):] if context is not None else None
            built = self._build_sequences_for_group(group_df.sort_values("time_idx"), feature_cols, seq_len, context_df=context_tail)
            if built is None: continue
            X, y, raw_close, raw_target = built
            X_val_list.append(X); y_val_list.append(y); g_val_list.append(np.full(len(y), group_id, dtype=np.int64))
            if raw_close is not None: raw_close_val_list.append(raw_close); raw_target_val_list.append(raw_target)

        if not X_train_list: raise ValueError("No training sequences could be built")
        X_train, y_train, g_train = np.concatenate(X_train_list), np.concatenate(y_train_list), np.concatenate(g_train_list)

        if X_val_list:
            X_val, y_val, g_val = np.concatenate(X_val_list), np.concatenate(y_val_list), np.concatenate(g_val_list)
            raw_close_val = np.concatenate(raw_close_val_list) if raw_close_val_list else None
            raw_target_val = np.concatenate(raw_target_val_list) if raw_target_val_list else None
        else:
            X_val, y_val, g_val = X_train[:0], y_train[:0], g_train[:0]
            raw_close_val, raw_target_val = None, None
        return X_train, y_train, g_train, X_val, y_val, g_val, raw_close_val, raw_target_val

    def _compute_directional_accuracy(self, model, X_val, g_val, raw_close_val, raw_target_val, device, scaler_state):
        try:
            if raw_close_val is None or raw_target_val is None or len(X_val) == 0: return None
            model.eval()
            preds_list = []
            with torch.no_grad():
                for start in range(0, len(X_val), self.batch_size):
                    X_batch = torch.from_numpy(X_val[start:start+self.batch_size]).to(device)
                    g_batch = torch.from_numpy(g_val[start:start+self.batch_size]).long().to(device)
                    preds_list.append(model(X_batch, g_batch).cpu().numpy())

            median_price = np.concatenate(preds_list, axis=0) * scaler_state["target_std"] + scaler_state["target_mean"]
            actual_direction = np.sign(raw_target_val - raw_close_val)
            predicted_direction = np.sign(median_price - raw_close_val)
            valid_mask = actual_direction != 0
            if valid_mask.sum() == 0: return None
            return float(((actual_direction == predicted_direction) & valid_mask).sum() / valid_mask.sum())
        except Exception: return None
        finally: model.train()

    def train_single_timeframe(self, data, feature_cols, timeframe, num_groups, epochs=100, learning_rate=0.001,
                               sequence_length=30, hidden_size=64, num_layers=2, dropout=0.2, batch_size=128,
                               status_callback=None, **kwargs):
        try:
            self.stop_flag = False
            self.timeframe = timeframe
            self.batch_size = batch_size
            self.hidden_size = hidden_size
            self.num_layers = num_layers
            self.dropout = dropout
            self.sequence_length = sequence_length

            if len(data) == 0: raise ValueError("Prepared dataframe is empty")
            train_df, val_df = self._chronological_split(data)
            if len(train_df) == 0: raise ValueError("Train split produced an empty set")

            scaler_state = self._fit_scalers(train_df, feature_cols)
            train_df = self._apply_scalers(train_df, feature_cols, scaler_state)
            val_df = self._apply_scalers(val_df, feature_cols, scaler_state) if len(val_df) else val_df

            X_train, y_train, g_train, X_val, y_val, g_val, raw_close_val, raw_target_val = self._build_sequence_arrays(
                train_df, val_df, feature_cols, sequence_length)

            self._log(f"📊 [{timeframe}] Training sequences: {len(y_train):,}, Val: {len(y_val):,}")
            train_loader = DataLoader(SequenceDataset(X_train, g_train, y_train), batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(SequenceDataset(X_val, g_val, y_val), batch_size=batch_size, shuffle=False) if len(y_val) > 0 else None

            last_error = None
            trained_ok = False
            for device in self.gpu_manager.preferred_devices():
                if self.stop_flag: break
                try:
                    trained_ok = self._run_training_loop(train_loader, val_loader, len(feature_cols), num_groups, hidden_size, num_layers, dropout, self.embedding_dim, epochs, learning_rate, device, timeframe, status_callback)
                    self.device_used = str(device)
                    break
                except RuntimeError as exc:
                    last_error = exc
                    MemoryOptimizer.clear_cache()

            if not trained_ok: raise last_error or RuntimeError("No device succeeded")
            self.scaler_state = scaler_state
            self.is_trained = True
            self.training_history["directional_accuracy"] = self._compute_directional_accuracy(self.model, X_val, g_val, raw_close_val, raw_target_val, torch.device(self.device_used), scaler_state)
            if self.training_history["directional_accuracy"] is not None:
                self._log(f"🎯 [{timeframe}] Directional accuracy: {self.training_history['directional_accuracy']*100:.1f}%")
            return True
        except Exception as exc:
            self._log(f"❌ [{timeframe}] Training error: {exc}")
            traceback.print_exc()
            return False

    def _run_training_loop(self, train_loader, val_loader, num_features, num_groups, hidden_size, num_layers, dropout, embedding_dim, epochs, learning_rate, device, timeframe, status_callback):
        model = GRURegressor(num_features, hidden_size, num_layers, dropout, num_groups, embedding_dim).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        loss_fn = nn.MSELoss()
        best_val_loss, best_state, best_epoch, patience_counter = float("inf"), copy.deepcopy(model.state_dict()), 0, 0

        for epoch in range(1, epochs + 1):
            if self.stop_flag: break
            model.train()
            train_loss_sum, train_count = 0.0, 0
            for X_batch, g_batch, y_batch in train_loader:
                X_batch, g_batch, y_batch = X_batch.to(device), g_batch.to(device), y_batch.to(device)
                optimizer.zero_grad()
                loss = loss_fn(model(X_batch, g_batch), y_batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), Config.GRADIENT_CLIP_NORM)
                optimizer.step()
                train_loss_sum += loss.item() * len(y_batch); train_count += len(y_batch)
            train_loss = train_loss_sum / max(train_count, 1)

            val_loss = train_loss
            if val_loader:
                model.eval()
                val_loss_sum, val_count = 0.0, 0
                with torch.no_grad():
                    for X_batch, g_batch, y_batch in val_loader:
                        loss = loss_fn(model(X_batch.to(device), g_batch.to(device)), y_batch.to(device))
                        val_loss_sum += loss.item() * len(y_batch); val_count += len(y_batch)
                val_loss = val_loss_sum / max(val_count, 1)

            if status_callback: status_callback(f"[{timeframe}] Epoch {epoch}/{epochs}  train={train_loss:.5f}  val={val_loss:.5f}")
            if val_loss < best_val_loss - 1e-6:
                best_val_loss, best_state, best_epoch, patience_counter = val_loss, copy.deepcopy(model.state_dict()), epoch, 0
            else:
                patience_counter += 1
                if patience_counter >= Config.EARLY_STOPPING_PATIENCE: break

        model.load_state_dict(best_state)
        model.eval()
        self.model = model
        self.training_history.update({"timeframe": timeframe, "best_epoch": best_epoch, "best_val_loss": best_val_loss, "epochs_run": epoch})
        return True

    def save_model(self, model_dir, filename_stem, feature_cols, horizon, timeframe, num_groups, group_id_categories, sequence_length):
        if not self.is_trained or self.model is None: return False
        try:
            output_dir = Path(model_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            model_path = output_dir / f"{filename_stem}{Config.MODEL_EXT}"
            metadata_path = output_dir / f"{filename_stem}{Config.METADATA_EXT}"
            torch.save(self.model.state_dict(), str(model_path))
            metadata = {
                "model_type": "torch.GRURegressor", "timeframe": timeframe, "model_path": str(model_path),
                "feature_columns": feature_cols, "categorical_features": ["group_id"], "group_id_categories": group_id_categories,
                "num_groups": num_groups, "sequence_length": sequence_length,
                "architecture": {"num_features": len(feature_cols), "hidden_size": self.hidden_size, "num_layers": self.num_layers, "dropout": self.dropout, "embedding_dim": self.embedding_dim},
                "directional_accuracy": self.training_history.get("directional_accuracy"), "scaler_state": self.scaler_state,
                "device_used": self.device_used, "horizon": horizon, "training_history": self.training_history, "training_time": datetime.now().isoformat(),
            }
            with open(metadata_path, "wb") as file: pickle.dump(metadata, file)
            self._log(f"✅ GRU model saved: {model_path}")
            return True
        except Exception as exc:
            self._log(f"❌ Save error: {exc}")
            return False


class GRUTrainerGUI(CTk):
    def __init__(self):
        super().__init__()
        self.title(Config.APP_TITLE)
        self.geometry(Config.APP_GEOMETRY)
        self.resizable(False, False)

        self.gui_logger = GUILogger()
        self.gui_logger.enable_console_redirect()

        self.csv_folder = Config.DATASET_DIR
        self.model_folder = Config.MODEL_DIR
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
        main_frame.pack(fill="both", expand=True, padx=10, pady=5)

        title_frame = CTkFrame(main_frame, corner_radius=6)
        title_frame.pack(fill="x", pady=(0, 4))

        CTkLabel(title_frame, text="Copyright Su Nie | BSD-3C License | https://github.com/can87683", font=("Ubuntu", 16), text_color="yellow").pack(pady=(0, 4))

        CTkLabel(title_frame, text="🎯 GRU Model Trainer (per timeframe)", font=("Ubuntu", 20, "bold")).pack(pady=4)

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
        frame = CTkFrame(parent, corner_radius=6)
        frame.pack(fill="x", pady=(0, 4))
        folder_frame = CTkFrame(frame, fg_color="transparent")
        folder_frame.pack(fill="x", padx=8, pady=3)
        self.select_csv_btn = CTkButton(folder_frame, text="📁 CSV Folder", command=self.select_csv_folder, width=120, height=28)
        self.select_csv_btn.pack(side="left", padx=(0, 6))
        self.csv_path_var = tk.StringVar(value=str(self.csv_folder))
        CTkEntry(folder_frame, textvariable=self.csv_path_var, state="readonly", height=28).pack(side="left", fill="x", expand=True)

    def _create_model_folder_section(self, parent):
        frame = CTkFrame(parent, corner_radius=6)
        frame.pack(fill="x", pady=(0, 4))
        folder_frame = CTkFrame(frame, fg_color="transparent")
        folder_frame.pack(fill="x", padx=8, pady=3)
        self.select_model_btn = CTkButton(folder_frame, text="📁 Model Folder", command=self.select_model_folder, width=120, height=28)
        self.select_model_btn.pack(side="left", padx=(0, 6))
        self.model_path_var = tk.StringVar(value=str(self.model_folder))
        CTkEntry(folder_frame, textvariable=self.model_path_var, state="readonly", height=28).pack(side="left", fill="x", expand=True)
        self.status_label = CTkLabel(frame, text="📊 Status: No folder selected", font=("Ubuntu", 10))
        self.status_label.pack(anchor="w", padx=8, pady=1)

    def _create_parameters_section(self, parent):
        frame = CTkFrame(parent, corner_radius=6)
        frame.pack(fill="x", pady=(0, 4))
        param_frame = CTkFrame(frame, fg_color="transparent")
        param_frame.pack(fill="x", padx=8, pady=3)

        self.horizon = tk.StringVar(value=str(Config.DEFAULT_HORIZON))
        self.epochs = tk.StringVar(value=str(Config.DEFAULT_EPOCHS))
        self.learning_rate = tk.StringVar(value=str(Config.DEFAULT_LEARNING_RATE))
        self.sequence_length = tk.StringVar(value=str(Config.DEFAULT_SEQUENCE_LENGTH))
        self.hidden_size = tk.StringVar(value=str(Config.DEFAULT_HIDDEN_SIZE))
        self.num_layers = tk.StringVar(value=str(Config.DEFAULT_NUM_LAYERS))
        self.batch_size = tk.StringVar(value=str(Config.DEFAULT_BATCH_SIZE))
        self.dropout = tk.StringVar(value=str(Config.DEFAULT_DROPOUT))

        row1 = CTkFrame(param_frame, fg_color="transparent")
        row1.pack(fill="x", pady=1)
        for col in range(8): row1.grid_columnconfigure(col, weight=1 if col % 2 else 0)
        CTkLabel(row1, text="Horizon:", width=70).grid(row=0, column=0, padx=(0, 3), sticky="w")
        CTkEntry(row1, textvariable=self.horizon, width=60).grid(row=0, column=1, padx=(0, 10), sticky="w")
        CTkLabel(row1, text="Epochs:", width=70).grid(row=0, column=2, padx=(0, 3), sticky="w")
        CTkEntry(row1, textvariable=self.epochs, width=60).grid(row=0, column=3, padx=(0, 10), sticky="w")
        CTkLabel(row1, text="LR:", width=70).grid(row=0, column=4, padx=(0, 3), sticky="w")
        CTkEntry(row1, textvariable=self.learning_rate, width=60).grid(row=0, column=5, padx=(0, 10), sticky="w")
        CTkLabel(row1, text="Seq Len:", width=70).grid(row=0, column=6, padx=(0, 3), sticky="w")
        CTkEntry(row1, textvariable=self.sequence_length, width=60).grid(row=0, column=7, sticky="w")

        row2 = CTkFrame(param_frame, fg_color="transparent")
        row2.pack(fill="x", pady=1)
        for col in range(8): row2.grid_columnconfigure(col, weight=1 if col % 2 else 0)
        CTkLabel(row2, text="Hidden:", width=70).grid(row=0, column=0, padx=(0, 3), sticky="w")
        CTkEntry(row2, textvariable=self.hidden_size, width=60).grid(row=0, column=1, padx=(0, 10), sticky="w")
        CTkLabel(row2, text="Layers:", width=70).grid(row=0, column=2, padx=(0, 3), sticky="w")
        CTkEntry(row2, textvariable=self.num_layers, width=60).grid(row=0, column=3, padx=(0, 10), sticky="w")
        CTkLabel(row2, text="Batch:", width=70).grid(row=0, column=4, padx=(0, 3), sticky="w")
        CTkEntry(row2, textvariable=self.batch_size, width=60).grid(row=0, column=5, padx=(0, 10), sticky="w")
        CTkLabel(row2, text="Dropout:", width=70).grid(row=0, column=6, padx=(0, 3), sticky="w")
        CTkEntry(row2, textvariable=self.dropout, width=60).grid(row=0, column=7, sticky="w")

    def _create_features_section(self, parent):
        frame = CTkFrame(parent, corner_radius=6)
        frame.pack(fill="x", pady=(0, 4))
        feature_frame = CTkFrame(frame, fg_color="transparent")
        feature_frame.pack(fill="x", padx=8, pady=3)
        self.feature_cols_var = tk.StringVar(value="time,open,high,low,close,volume")
        CTkLabel(feature_frame, text="Base Features (comma-separated):", font=("Ubuntu", 10)).pack(anchor="w", pady=(0, 2))
        CTkEntry(feature_frame, textvariable=self.feature_cols_var, height=28).pack(fill="x", pady=(0, 2))

    def _create_timeframes_section(self, parent):
        frame = CTkFrame(parent, corner_radius=6)
        frame.pack(fill="x", pady=(0, 4))
        tf_frame = CTkFrame(frame, fg_color="transparent")
        tf_frame.pack(fill="x", padx=8, pady=3)
        CTkLabel(tf_frame, text="Timeframes (one model each):", font=("Ubuntu", 10)).pack(anchor="w", pady=(0, 2))
        self.tf_vars = {}
        checkboxes_frame = CTkFrame(tf_frame, fg_color="transparent")
        checkboxes_frame.pack(fill="x")
        for i, tf in enumerate(Config.TIMEFRAMES):
            var = tk.BooleanVar(value=True)
            self.tf_vars[tf] = var
            ctk.CTkCheckBox(checkboxes_frame, text=tf, variable=var, width=70, font=("Ubuntu", 10)).grid(row=0, column=i, padx=3, sticky="w")

    def _create_indicators_section(self, parent):
        frame = CTkFrame(parent, corner_radius=6)
        frame.pack(fill="x", pady=(0, 4))
        indicator_frame = CTkFrame(frame, fg_color="transparent")
        indicator_frame.pack(fill="x", padx=8, pady=3)
        CTkLabel(indicator_frame, text="Technical Indicators (inputs):", font=("Ubuntu", 10)).pack(anchor="w", pady=(0, 2))

        checkboxes_frame = CTkFrame(indicator_frame, fg_color="transparent")
        checkboxes_frame.pack(fill="x")

        indicators = [
            ("ATR", "enable_atr"), ("Bollinger", "enable_bollinger"), ("CCI", "enable_cci"),
            ("CMF", "enable_cmf"), ("Donchian", "enable_donchian"), ("EMA", "enable_ema"),
            ("Force Index", "enable_force_index"), ("MACD", "enable_macd"), ("MFI", "enable_mfi"),
            ("OBV", "enable_obv"), ("Price Struct", "enable_price_structure"), ("Realized Vol", "enable_realized_vol"),
            ("ROC", "enable_roc"), ("RSI", "enable_rsi"), ("RVWAP", "enable_rvwap"),
            ("Stochastic", "enable_stochastic"), ("Volume", "enable_volume_metrics")
        ]

        self.indicator_vars = {}
        cols_per_row = 6
        for i, (label, var_name) in enumerate(indicators):
            row_idx = i // cols_per_row
            col_idx = i % cols_per_row
            var = tk.BooleanVar(value=True)
            self.indicator_vars[var_name] = var
            ctk.CTkCheckBox(checkboxes_frame, text=label, variable=var, width=95, font=("Ubuntu", 10)).grid(row=row_idx, column=col_idx, padx=4, pady=2, sticky="w")

        CTkLabel(indicator_frame, text="Uncheck indicators you don't want as model inputs", font=("Ubuntu", 8), text_color="gray").pack(anchor="w", pady=(4, 0))

    def _create_controls_section(self, parent):
        frame = CTkFrame(parent, corner_radius=6)
        frame.pack(fill="x", pady=(0, 4))
        button_frame = CTkFrame(frame, fg_color="transparent")
        button_frame.pack(pady=4)
        self.train_btn = CTkButton(button_frame, text="🚀 Start Training", command=self.start_training, width=130, height=34, fg_color="#2e7d32", hover_color="#1b5e20", font=("Ubuntu", 12, "bold"))
        self.train_btn.pack(side="left", padx=4)
        self.stop_btn = CTkButton(button_frame, text="⏹️ Stop Training", command=self.stop_training, width=130, height=34, fg_color="#c62828", hover_color="#b71c1c", font=("Ubuntu", 12, "bold"), state="disabled")
        self.stop_btn.pack(side="left", padx=4)

    def _create_status_section(self, parent):
        frame = CTkFrame(parent, corner_radius=6)
        frame.pack(fill="x", pady=(0, 4))
        self.status_text = CTkTextbox(frame, height=50, font=("Ubuntu Mono", 10))
        self.status_text.pack(fill="x", padx=8, pady=3)
        self.status_text.insert("1.0", "Ready")
        self.status_text.configure(state="disabled")

    def _create_log_section(self, parent):
        frame = CTkFrame(parent, corner_radius=6)
        frame.pack(fill="both", expand=True)
        self.log_text = CTkTextbox(frame, font=("Ubuntu Mono", 10))
        self.log_text.pack(fill="both", expand=True, padx=8, pady=3)

    def update_status(self, message):
        self.status_text.configure(state="normal")
        self.status_text.delete("1.0", "end")
        self.status_text.insert("1.0", message)
        self.status_text.configure(state="disabled")

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

    def scan_data_source(self):
        csv_folder = Path(self.csv_path_var.get())
        if not csv_folder.exists():
            self.status_label.configure(text="❌ Folder not found")
            self.train_btn.configure(state="disabled")
            return
        symbols, timeframes = self.data_preparer.scan_folder(csv_folder)
        if not symbols:
            self.status_label.configure(text="⚠️ No symbols found")
            self.train_btn.configure(state="disabled")
            return
        self.status_label.configure(text=f"✅ Found {len(symbols)} symbols, {len(timeframes)} timeframes")
        self.train_btn.configure(state="normal")

    def start_training(self):
        if self.is_training: return
        csv_folder = self.csv_path_var.get()
        if not Path(csv_folder).exists(): return
        selected_timeframes = [tf for tf, var in self.tf_vars.items() if var.get()]
        if not selected_timeframes: return

        self.is_training = True
        self.stop_flag = False
        self.train_btn.configure(state="disabled")
        self.stop_btn.configure(state="normal")

        def training_task():
            try:
                self.gui_logger.log("=" * 60)
                self.gui_logger.log("🚀 Starting Per-Timeframe GRU Training")

                flags = {k: v.get() for k, v in self.indicator_vars.items()}
                self.data_preparer.set_indicator_flags(**flags)

                enabled_list = [k.replace("enable_", "").upper() for k, v in self.indicator_vars.items() if v.get()]
                self.gui_logger.log(f"📊 Indicators enabled: {', '.join(enabled_list)}")

                feature_input = self.feature_cols_var.get().strip()
                feature_cols = [f.strip() for f in feature_input.split(',') if f.strip()] if feature_input else Config.DEFAULT_FEATURES
                horizon = int(self.horizon.get())
                epochs = int(self.epochs.get())
                learning_rate = float(self.learning_rate.get())
                sequence_length = int(self.sequence_length.get())
                hidden_size = int(self.hidden_size.get())
                num_layers = int(self.num_layers.get())
                batch_size = int(self.batch_size.get())
                dropout = float(self.dropout.get())
                model_folder = self.model_path_var.get()

                results = {}
                for timeframe in selected_timeframes:
                    if self.stop_flag: break
                    self.update_status(f"[{timeframe}] Building dataset...")
                    try:
                        data = self.data_preparer.build_dataframe_for_timeframe(csv_folder, feature_cols, horizon, timeframe)
                    except ValueError as exc:
                        results[timeframe] = "skipped"
                        continue

                    self.update_status(f"[{timeframe}] Training GRU model...")
                    trainer = StreamingGRUTrainer(self.gui_logger.log, timeframe=timeframe)
                    self.current_trainer = trainer
                    success = trainer.train_single_timeframe(
                        data=data, feature_cols=self.data_preparer.feature_cols, timeframe=timeframe,
                        num_groups=len(self.data_preparer.group_id_categories), epochs=epochs, learning_rate=learning_rate,
                        sequence_length=sequence_length, hidden_size=hidden_size, num_layers=num_layers,
                        dropout=dropout, batch_size=batch_size, status_callback=self.update_status
                    )
                    if success and not trainer.stop_flag:
                        filename_stem = f"{Config.MODEL_PREFIX}_{timeframe}"
                        trainer.save_model(model_folder, filename_stem, self.data_preparer.feature_cols, horizon, timeframe, len(self.data_preparer.group_id_categories), self.data_preparer.group_id_categories, sequence_length)
                        results[timeframe] = f"trained ({trainer.device_used})"
                    else:
                        results[timeframe] = "stopped" if trainer.stop_flag else "failed"

                    del trainer
                    self.current_trainer = None
                    MemoryOptimizer.clear_cache()

                self.gui_logger.log(f"🏁 Summary: {', '.join(f'{tf}: {status}' for tf, status in results.items())}")
                self.update_status("Done.")
            except Exception as e:
                self.gui_logger.log(f"❌ Training error: {e}", "error")
                traceback.print_exc()
            finally:
                self.is_training = False
                self.current_trainer = None
                self.train_btn.configure(state="normal")
                self.stop_btn.configure(state="disabled")

        threading.Thread(target=training_task, daemon=True).start()

    def stop_training(self):
        self.stop_flag = True
        if self.current_trainer: self.current_trainer.stop_training()

    def _on_closing(self):
        self.gui_logger.disable_console_redirect()
        if self.is_training:
            if messagebox.askyesno("Training in Progress", "Stop training and quit?"):
                self.stop_flag = True
                if self.current_trainer: self.current_trainer.stop_training()
                self.after(100, self.destroy)
        else:
            if messagebox.askyesno("Quit", "Do you want to quit?"):
                self.destroy()


def main():
    app = GRUTrainerGUI()
    app.mainloop()

if __name__ == "__main__":
    main()