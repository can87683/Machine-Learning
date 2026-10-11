"""
customtkinter lightning numpy pandas psutil pytorch-forecasting pytorch-lightning setproctitle torch
"""
#!/usr/bin/env python3
# ai_predict_train_tft.py
# Copyright Su Nie | BSD-3C License | https://github.com/can87683


import setproctitle
setproctitle.setproctitle("ai_predict_train_tft_ctk")

import customtkinter as ctk
from customtkinter import CTk, CTkFrame, CTkLabel, CTkButton, CTkEntry, CTkTextbox, CTkCheckBox
import tkinter as tk
from tkinter import filedialog, messagebox
import threading
import multiprocessing
import configparser
from pathlib import Path
import numpy as np
import pandas as pd
import pickle
import gc
import psutil
from datetime import datetime
import sys

import torch

multiprocessing.set_start_method('spawn', force=True)

import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer, QuantileLoss
from pytorch_forecasting.data.encoders import GroupNormalizer


class Config:
    HOME = Path.home()
    MODEL_DIR = HOME / "ai_model_ml"
    DATASET_DIR = HOME / "mt5_data"

    MAX_ROWS_PER_SYMBOL = 5000
    MAX_DATASET_ROWS = 10000

    DEFAULT_FEATURES = ['time', 'open', 'high', 'low', 'close', 'tick_volume']

    DEFAULT_LOOKBACK = 60
    DEFAULT_HORIZON = 10
    DEFAULT_EPOCHS = 30
    DEFAULT_BATCH_SIZE = 512
    DEFAULT_HIDDEN_SIZE = 64
    DEFAULT_NUM_HEADS = 4
    DROPOUT_RATE = 0.2

    TIMEFRAME_HYPERPARAMS = {
        "M5":  {"lookback": 60, "hidden_size": 64, "dropout": 0.2},
        "M15": {"lookback": 60, "hidden_size": 64, "dropout": 0.2},
        "H1":  {"lookback": 48, "hidden_size": 48, "dropout": 0.25},
        "H4":  {"lookback": 40, "hidden_size": 48, "dropout": 0.25},
        "D1":  {"lookback": 26, "hidden_size": 32, "dropout": 0.3},
        "W1":  {"lookback": 16, "hidden_size": 24, "dropout": 0.35},
    }

    QUANTILES = [0.02, 0.1, 0.25, 0.5, 0.75, 0.9, 0.98]
    EARLY_STOPPING_PATIENCE = 10
    LEARNING_RATE = 0.0005
    GRADIENT_CLIP_NORM = 1.0

    NUM_WORKERS = min(4, multiprocessing.cpu_count() or 4)
    TIMEFRAMES = ['M5', 'M15', 'H1', 'H4', 'D1', 'W1']

    MODEL_PREFIX = "ai_predict_tft"
    MODEL_EXT = ".ckpt"
    METADATA_EXT = ".pkl"

    APP_TITLE = "🚀 TFT Model Trainer"
    APP_GEOMETRY = "640x1050"
    THEME = "dark"
    COLOR_THEME = "dark-blue"

    SEED = 42

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

pl.seed_everything(Config.SEED, workers=True)


class GPUManager:
    def __init__(self, verbose=True, log_callback=None):
        self.verbose = verbose
        self.log_callback = log_callback
        self.use_cuda = torch.cuda.is_available()
        self.use_mps = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()

        if self.use_cuda:
            self.device = torch.device('cuda:0')
            self.gpu_name = torch.cuda.get_device_name(0)
            self.gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            self._log(f"✅ GPU detected: {self.gpu_name} ({self.gpu_memory_gb:.2f} GB)")
        elif self.use_mps:
            self.device = torch.device('mps')
            self.gpu_name = "Apple MPS"
            self.gpu_memory_gb = 0.0
            self._log("✅ MPS detected")
        else:
            self.device = torch.device('cpu')
            self.gpu_name = None
            self.gpu_memory_gb = 0.0
            self._log("ℹ️ No GPU detected, using CPU")

    def _log(self, message):
        if self.log_callback:
            self.log_callback(message)
        elif self.verbose:
            print(f"[GPUManager] {message}")

    def get_device(self, framework='pytorch'):
        return self.device if framework == 'pytorch' else None

    def optimize_for_gpu(self):
        if not (self.use_cuda or self.use_mps):
            return {'batch_size': 128, 'device': self.device}
        mem_gb = self.gpu_memory_gb if self.use_cuda else 8.0
        return {'batch_size': 1024 if mem_gb >= 8 else 512, 'device': self.device}

    def log_system_info(self):
        self._log("=" * 60)
        self._log("🖥️ System & Library Information")
        self._log("=" * 60)
        self._log(f"🐍 Python: {sys.version.split()[0]}")
        self._log(f"🔥 PyTorch: {torch.__version__}")
        self._log(f"🔌 CUDA Available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            self._log(f"🎮 GPU Device: {torch.cuda.get_device_name(0)}")
            self._log(f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        elif self.use_mps:
            self._log("🍎 Using Apple MPS backend")
        else:
            self._log("⚠️ No usable GPU for PyTorch. Will use CPU.")
        self._log("=" * 60)


class UsageRow:
    # --- Display constants ---
    FRAME_HEIGHT = 40
    FRAME_CORNER_RADIUS = 5
    FRAME_PADY = (0, 10)

    FONT = ("Ubuntu Mono", 20)
    TEXT_COLOR = "Lime"
    PLACEHOLDER_TEXT = "Loading system usage..."

    LABEL_PADX = 10
    LABEL_PADY = 5

    REFRESH_MS = 2000
    CPU_SAMPLE_INTERVAL = 0.5

    def __init__(self, parent):
        self.usage_frame = CTkFrame(
            parent,
            height=self.FRAME_HEIGHT,
            corner_radius=self.FRAME_CORNER_RADIUS,
        )
        self.usage_frame.pack(fill="x", pady=self.FRAME_PADY)

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
    def get_system_memory_available_gb():
        return psutil.virtual_memory().available / 1e9

    @staticmethod
    def clear_cache():
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class FeatureEngineer:
    @staticmethod
    def add_time_features(df):
        if 'time' not in df.columns:
            return df
        if not pd.api.types.is_datetime64_any_dtype(df['time']):
            df['time'] = pd.to_datetime(df['time'])
        df['hour'] = df['time'].dt.hour
        df['day_of_week'] = df['time'].dt.dayofweek
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
            (df['low'] - prev_close).abs()
        ], axis=1).max(axis=1)
        df['atr'] = true_range.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
        return df

    @staticmethod
    def _add_bollinger(df, period=20, std_dev=2):
        mid = df['close'].rolling(period).mean()
        std = df['close'].rolling(period).std()
        upper = mid + std_dev * std
        lower = mid - std_dev * std
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
        mfv = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low']).replace(0, np.nan) * df['tick_volume']
        df['cmf'] = mfv.rolling(period).sum() / df['tick_volume'].rolling(period).sum().replace(0, np.nan)
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
        df['force_index'] = (df['close'].diff() * df['tick_volume']).ewm(span=period, adjust=False).mean()
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
        raw_money_flow = typical_price * df['tick_volume']
        positive_flow = raw_money_flow.where(typical_price > typical_price.shift(1), 0.0)
        negative_flow = raw_money_flow.where(typical_price < typical_price.shift(1), 0.0)
        money_flow_ratio = positive_flow.rolling(period).sum() / negative_flow.rolling(period).sum().replace(0, np.nan)
        df['mfi'] = 100 - (100 / (1 + money_flow_ratio))
        return df

    @staticmethod
    def _add_obv(df):
        df['obv'] = (np.sign(df['close'].diff()).fillna(0) * df['tick_volume']).cumsum()
        rolling_vol = df['tick_volume'].rolling(20).sum().replace(0, np.nan)
        df['obv_slope_5'] = df['obv'].diff(5) / rolling_vol
        return df

    @staticmethod
    def _add_price_structure(df):
        range_hl = (df['high'] - df['low']).replace(0, np.nan)
        df['hl_range_pct'] = range_hl / df['close']
        df['body_pct'] = (df['close'] - df['open']).abs() / df['close']
        df['body_range_ratio'] = (df['close'] - df['open']).abs() / range_hl
        df['close_location_value'] = (df['close'] - df['low']) / range_hl
        open_close_max = df[['open', 'close']].max(axis=1)
        open_close_min = df[['open', 'close']].min(axis=1)
        df['upper_shadow_pct'] = (df['high'] - open_close_max) / df['close']
        df['lower_shadow_pct'] = (open_close_min - df['low']) / df['close']
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
        df['rsi'] = 100 - (100 / (1 + avg_gain / avg_loss.replace(0, np.nan)))
        return df

    @staticmethod
    def _add_rvwap(df, period=20):
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        rolling_tp_vol = (typical_price * df['tick_volume']).rolling(period).sum()
        rolling_vol = df['tick_volume'].rolling(period).sum().replace(0, np.nan)
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
        volume_ma_20 = df['tick_volume'].rolling(20).mean()
        volume_std_20 = df['tick_volume'].rolling(20).std()
        df['volume_ratio_20'] = df['tick_volume'] / volume_ma_20.replace(0, np.nan)
        df['volume_zscore_20'] = (df['tick_volume'] - volume_ma_20) / volume_std_20.replace(0, np.nan)
        df['return_x_volume_1'] = df['close'].pct_change() * df['tick_volume']
        dollar_volume = (df['close'] * df['tick_volume']).replace(0, np.nan)
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
        self.log_widget.configure(font=("Ubuntu Mono", 12), text_color="white")
        for msg, level in self.messages:
            self._write_to_widget(msg, level)
        self.messages.clear()

    def _write_to_widget(self, message, level):
        log_entry = f"[{datetime.now().strftime('%H:%M:%S')}] {message}\n"
        self.log_widget.insert("end", log_entry)
        self.log_widget.see("end")
        self.log_widget.update_idletasks()

    def log(self, message, level="info"):
        console_entry = f"[{datetime.now().strftime('%H:%M:%S')}] [{level.upper()}] {message}"
        self._original_stdout.write(console_entry + "\n")
        self._original_stdout.flush()
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
        if not csv_file.exists():
            return None
        df = pd.read_csv(csv_file)
        if "time" not in df.columns:
            self._log(f"⚠️ Missing time column: {csv_file}")
            return None
        if pd.api.types.is_numeric_dtype(df["time"]):
            df["time"] = pd.to_datetime(df["time"], unit="s")
        else:
            df["time"] = pd.to_datetime(df["time"])
        return df.sort_values("time").drop_duplicates("time").reset_index(drop=True)

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

    def _prepare_single_series(self, df, group_id, horizon, requested_features):
        if df is None or len(df) < 100:
            return None
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

        required_columns = ["open", "high", "low", "close", "tick_volume"]
        for col in required_columns:
            if col in df.columns and col not in available_features:
                available_features.append(col)

        if "close" not in df.columns:
            self._log(f"⚠️ Skipping {group_id}: close column unavailable")
            return None
        if not available_features:
            self._log(f"⚠️ Skipping {group_id}: no usable features")
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
        if len(df) < Config.DEFAULT_LOOKBACK + horizon + 10:
            return None

        df["group_id"] = str(group_id)
        df["time_idx"] = np.arange(len(df), dtype=np.int64)
        return df[["time", "time_idx", "group_id", self.target_column] + available_features], available_features

    def build_dataframe(self, csv_folder, feature_cols, horizon=10, timeframe=None):
        csv_folder = Path(csv_folder)
        symbols, _ = self.scan_folder(csv_folder)
        if not symbols or not timeframe:
            raise ValueError("Invalid symbols or timeframe")

        frames = []
        all_features = []
        for symbol, timeframes in symbols:
            if timeframe not in timeframes:
                continue
            df = self.load_csv_data(symbol, timeframe, csv_folder)
            prepared = self._prepare_single_series(df, symbol, horizon, feature_cols)
            if prepared is None:
                continue
            prepared_df, available_features = prepared
            frames.append(prepared_df)
            all_features.extend(available_features)
            self._log(f"   📊 {symbol} [{timeframe}]: {len(prepared_df)} rows")

        if not frames:
            raise ValueError(f"No usable time series for {timeframe}")

        all_features = list(dict.fromkeys(all_features))
        result = pd.concat(frames, ignore_index=True, sort=False)
        for col in all_features:
            if col not in result.columns:
                result[col] = 0.0

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
        result = result.sort_values(["group_id", "time_idx"]).reset_index(drop=True)

        if len(result) > Config.MAX_DATASET_ROWS:
            self._log(f"⚠️ Dataset exceeds {Config.MAX_DATASET_ROWS} rows. Truncating per symbol.")
            max_per_symbol = max(100, Config.MAX_DATASET_ROWS // max(1, result['group_id'].nunique()))
            result = result.groupby("group_id").tail(max_per_symbol).reset_index(drop=True)
            self._log(f"✅ [{timeframe}] Truncated to {len(result):,} rows")

        self.feature_cols = all_features
        self._log(f"✅ [{timeframe}] Prepared {len(result):,} rows, {result['group_id'].nunique()} symbols, {len(self.feature_cols)} features")
        return result


class StreamingTFTTrainer:
    def __init__(self, log_callback=None, timeframe=None):
        self.log_callback = log_callback or print
        self.timeframe = timeframe
        self.model = None
        self.training_dataset = None
        self.validation_dataset = None
        self.dataset_parameters = None
        self.checkpoint_path = None
        self.is_trained = False
        self.stop_flag = False
        self.gpu_manager = GPUManager(verbose=False)
        self.device = self.gpu_manager.get_device("pytorch")
        self.batch_size = self.gpu_manager.optimize_for_gpu().get("batch_size", Config.DEFAULT_BATCH_SIZE)

        tf_params = Config.TIMEFRAME_HYPERPARAMS.get(timeframe, {})
        self.hidden_size = tf_params.get("hidden_size", Config.DEFAULT_HIDDEN_SIZE)
        self.dropout = tf_params.get("dropout", Config.DROPOUT_RATE)
        self.hidden_continuous_size = max(8, self.hidden_size // 2)
        self.quantiles = list(Config.QUANTILES)
        self.training_history = {
            "best_val_loss": float("inf"),
            "epochs_completed": 0,
            "directional_accuracy": None,
        }
        self._log(f"🚀 TFT trainer [{timeframe}] initialized on device: {self.device}")

    def _log(self, message):
        if self.log_callback:
            self.log_callback(message)

    def stop_training(self):
        self.stop_flag = True
        self._log("⏹️ Training stop requested")

    def train_streaming(self, data_stream, total_samples, epochs=30, batch_size=512, lookback=60, horizon=10, **kwargs):
        self.stop_flag = False
        csv_folder = kwargs.get("csv_folder")
        feature_cols = kwargs.get("feature_cols")
        timeframe = kwargs.get("timeframe", self.timeframe)
        auto_tune = kwargs.get("auto_tune_lookback", True)
        preparer = kwargs.get("data_preparer")

        if auto_tune:
            lookback = Config.TIMEFRAME_HYPERPARAMS.get(timeframe, {}).get("lookback", lookback)

        self._log(f"📊 [{timeframe}] Building PyTorch Forecasting dataframe...")
        data = preparer.build_dataframe(csv_folder, feature_cols, horizon, timeframe)
        if len(data) == 0:
            raise ValueError("Empty dataframe")

        group_max = data.groupby("group_id")["time_idx"].transform("max")
        training_data = data[data["time_idx"] <= (group_max - horizon)].copy()
        if len(training_data) == 0:
            raise ValueError("Empty training data after horizon trim")

        known_reals = [c for c in preparer.feature_cols if c in {"hour_sin", "hour_cos", "day_sin", "day_cos", "month_sin", "month_cos"}]
        unknown_reals = [c for c in preparer.feature_cols if c not in known_reals]
        if "close" not in unknown_reals and "close" in data.columns:
            unknown_reals.append("close")

        self._log(f"🔧 [{timeframe}] Creating TimeSeriesDataSet (lookback={lookback}, horizon={horizon})...")
        self.training_dataset = TimeSeriesDataSet(
            training_data,
            time_idx="time_idx",
            target="target",
            group_ids=["group_id"],
            min_encoder_length=lookback,
            max_encoder_length=lookback,
            min_prediction_length=horizon,
            max_prediction_length=horizon,
            static_categoricals=["group_id"],
            time_varying_known_reals=["time_idx"] + known_reals,
            time_varying_unknown_reals=unknown_reals,
            target_normalizer=GroupNormalizer(groups=["group_id"]),
            add_relative_time_idx=True,
            add_encoder_length=True,
            allow_missing_timesteps=False,
        )
        self.dataset_parameters = self.training_dataset.get_parameters()
        self.validation_dataset = TimeSeriesDataSet.from_dataset(
            self.training_dataset, data, predict=False, stop_randomization=True
        )

        train_loader = self.training_dataset.to_dataloader(train=True, batch_size=batch_size, num_workers=Config.NUM_WORKERS)
        val_loader = self.validation_dataset.to_dataloader(train=False, batch_size=batch_size, num_workers=Config.NUM_WORKERS)

        self._log(f"📊 [{timeframe}] Train windows: {len(self.training_dataset):,} | Val windows: {len(self.validation_dataset):,}")

        self.model = TemporalFusionTransformer.from_dataset(
            self.training_dataset,
            learning_rate=Config.LEARNING_RATE,
            hidden_size=self.hidden_size,
            attention_head_size=Config.DEFAULT_NUM_HEADS,
            dropout=self.dropout,
            hidden_continuous_size=self.hidden_continuous_size,
            output_size=len(self.quantiles),
            loss=QuantileLoss(quantiles=self.quantiles),
            log_interval=-1,
            reduce_on_plateau_patience=4,
        )
        self._log(f"🧠 [{timeframe}] Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")

        accelerator = "gpu" if self.device.type == "cuda" else ("mps" if self.device.type == "mps" else "cpu")
        precision = "bf16-mixed" if accelerator == "gpu" and torch.cuda.is_bf16_supported() else "32-true"

        checkpoint_callback = ModelCheckpoint(
            dirpath=str(MODEL_DIR),
            filename=f"tft-{timeframe.lower()}-best-{{epoch:02d}}-{{val_loss:.5f}}",
            monitor="val_loss",
            mode="min",
            save_top_k=1,
            save_last=True,
        )
        early_stopping = EarlyStopping(
            monitor="val_loss",
            patience=Config.EARLY_STOPPING_PATIENCE,
            mode="min",
            verbose=True,
        )

        lightning_trainer = pl.Trainer(
            accelerator=accelerator,
            devices=1,
            max_epochs=epochs,
            min_epochs=1,
            precision=precision,
            gradient_clip_val=Config.GRADIENT_CLIP_NORM,
            callbacks=[checkpoint_callback, early_stopping],
            logger=False,
            enable_progress_bar=False,
            enable_model_summary=False,
            num_sanity_val_steps=0,
        )

        self._log(f"🚀 [{timeframe}] Starting TFT training (accelerator={accelerator}, precision={precision})...")
        lightning_trainer.fit(self.model, train_dataloaders=train_loader, val_dataloaders=val_loader)

        if checkpoint_callback.best_model_score is not None:
            self.training_history["best_val_loss"] = float(checkpoint_callback.best_model_score.detach().cpu().item())
        self.training_history["epochs_completed"] = lightning_trainer.current_epoch + 1
        self.is_trained = True
        self.checkpoint_path = checkpoint_callback.best_model_path

        self.training_history["directional_accuracy"] = self._compute_directional_accuracy(val_loader, accelerator)
        if self.training_history["directional_accuracy"] is not None:
            self._log(f"🎯 [{timeframe}] Directional accuracy: {self.training_history['directional_accuracy'] * 100:.1f}%")

        self._log(f"🏁 [{timeframe}] TFT training complete")
        return True

    def _compute_directional_accuracy(self, dataloader, accelerator):
        self.model.eval()
        pred_res = self.model.predict(
            dataloader,
            mode="prediction",
            return_x=True,
            trainer_kwargs=dict(
                accelerator=accelerator,
                devices=1,
                enable_progress_bar=False,
                logger=False,
            ),
        )
        preds = pred_res.output
        if isinstance(preds, (list, tuple)):
            preds = preds[0]

        encoder_target = pred_res.x["encoder_target"]
        decoder_target = pred_res.x["decoder_target"]
        if isinstance(encoder_target, (list, tuple)):
            encoder_target = encoder_target[0]
        if isinstance(decoder_target, (list, tuple)):
            decoder_target = decoder_target[0]

        median_idx = len(self.quantiles) // 2
        preds_median = preds[..., median_idx]

        last_actual = encoder_target[:, -1].unsqueeze(-1)
        actual_dir = torch.sign(decoder_target - last_actual)
        pred_dir = torch.sign(preds_median - last_actual)

        valid_mask = actual_dir != 0
        if valid_mask.sum().item() == 0:
            self.model.train()
            return None

        accuracy = ((actual_dir == pred_dir) & valid_mask).float().sum().item() / valid_mask.float().sum().item()
        self.model.train()
        return accuracy

    def save_model(self, filename):
        if not self.is_trained or self.model is None:
            self._log("⚠️ No trained model available")
            return False

        output_dir = Path(MODEL_DIR)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / filename
        if output_path.suffix != Config.MODEL_EXT:
            output_path = output_path.with_suffix(Config.MODEL_EXT)

        self.model.to("cpu")
        self.model.save_checkpoint(str(output_path))

        metadata_path = output_path.with_suffix(Config.METADATA_EXT)
        metadata = {
            "model_type": "pytorch_forecasting.TemporalFusionTransformer",
            "timeframe": self.timeframe,
            "checkpoint_path": str(output_path),
            "dataset_parameters": self.dataset_parameters,
            "feature_columns": self.training_dataset.reals if self.training_dataset else [],
            "quantiles": self.quantiles,
            "directional_accuracy": self.training_history.get("directional_accuracy"),
            "training_history": self.training_history,
            "training_time": datetime.now().isoformat(),
            "lookback": self.training_dataset.max_encoder_length if self.training_dataset else None,
            "horizon": self.training_dataset.max_prediction_length if self.training_dataset else None,
        }
        with open(metadata_path, "wb") as f:
            pickle.dump(metadata, f)

        self._log(f"✅ [{self.timeframe}] Checkpoint saved: {output_path}")
        self._log(f"✅ [{self.timeframe}] Metadata saved: {metadata_path}")
        return True

class TFTTrainerGUI(CTk):
    INI_PATH = Path(__file__).with_suffix(".ini")

    INDICATOR_CONFIG = [
        ("ATR", "enable_atr", True),
        ("Bollinger", "enable_bollinger", True),
        ("CCI", "enable_cci", True),
        ("CMF", "enable_cmf", True),
        ("Donchian", "enable_donchian", True),
        ("EMA", "enable_ema", True),
        ("Force Index", "enable_force_index", True),
        ("MACD", "enable_macd", True),
        ("MFI", "enable_mfi", True),
        ("OBV", "enable_obv", True),
        ("Price Struct", "enable_price_structure", True),
        ("Realized Vol", "enable_realized_vol", True),
        ("ROC", "enable_roc", True),
        ("RSI", "enable_rsi", True),
        ("RVWAP", "enable_rvwap", True),
        ("Stochastic", "enable_stochastic", True),
        ("Volume", "enable_volume_metrics", True),
    ]

    def __init__(self):
        super().__init__()
        self.title(Config.APP_TITLE)
        self.geometry(Config.APP_GEOMETRY)
        self.resizable(False, False)
        self._is_closing = False

        # Hide window while we build the UI and restore saved geometry,
        # so the user never sees the default position/size flash before
        # the saved position is applied.
        self.withdraw()

        self.gui_logger = GUILogger()
        self.gui_logger.enable_console_redirect()

        self.gpu_manager = GPUManager(verbose=False, log_callback=self.gui_logger.log)

        self.csv_folder = Config.DATASET_DIR
        self.model_folder = Config.MODEL_DIR
        self.timeframe_vars = {}
        self.is_training = False
        self.stop_flag = False

        self.data_preparer = StreamingDataPreparer(self.gui_logger.log)
        self.current_trainer = None

        self._create_ui()
        self._load_state()  # applied while the window is hidden
        self.gui_logger.set_log_widget(self.log_text)

        self.gpu_manager.log_system_info()

        self.after(500, self.scan_data_source)
        self.protocol("WM_DELETE_WINDOW", self._on_closing)

        # Flush pending geometry/layout work, then reveal the window.
        self.update_idletasks()
        self.deiconify()

    def _load_state(self):
        if not self.INI_PATH.exists():
            return
        cp = configparser.ConfigParser()
        cp.read(str(self.INI_PATH))

        if cp.has_section("window"):
            geom = cp.get("window", "geometry", fallback="")
            if geom:
                # wm_geometry() is frame-relative and round-trips exactly
                # through self.geometry() - no title-bar drift.
                self.geometry(geom)
            else:
                # Legacy ini stored only x/y from winfo_x/winfo_y (client coords).
                # On Windows this caused a ~37px downward creep per launch.
                # Fall back to size only to break the drift loop.
                x = cp.get("window", "x", fallback="")
                y = cp.get("window", "y", fallback="")
                if x and y:
                    try:
                        # winfo_x/y are client coords; geometry() expects
                        # frame coords. Since we can't reliably reconstruct
                        # the title-bar offset here, just restore size only
                        # and let the WM place the window.
                        self.geometry(Config.APP_GEOMETRY)
                    except Exception:
                        pass

        if cp.has_section("paths"):
            csv = cp.get("paths", "csv_folder", fallback="")
            model = cp.get("paths", "model_folder", fallback="")
            if csv:
                self.csv_path_var.set(csv)
                self.csv_folder = Path(csv)
            if model:
                self.model_path_var.set(model)
                self.model_folder = Path(model)

        if cp.has_section("params"):
            self.lookback.set(cp.get("params", "lookback", fallback=str(Config.DEFAULT_LOOKBACK)))
            self.horizon.set(cp.get("params", "horizon", fallback=str(Config.DEFAULT_HORIZON)))
            self.epochs.set(cp.get("params", "epochs", fallback=str(Config.DEFAULT_EPOCHS)))
            self.batch_size.set(cp.get("params", "batch_size", fallback=str(Config.DEFAULT_BATCH_SIZE)))
            self.auto_tune_var.set(cp.getboolean("params", "auto_tune", fallback=True))

        if cp.has_section("features"):
            self.feature_cols_var.set(cp.get("features", "base_features", fallback="time,open,high,low,close,tick_volume"))

        if cp.has_section("indicators"):
            for var_name in self.indicator_vars:
                self.indicator_vars[var_name].set(cp.getboolean("indicators", var_name, fallback=True))

    def _save_state(self):
        cp = configparser.ConfigParser()
        # Store wm_geometry() directly. It is frame-relative (same coordinate
        # system that geometry() consumes), so a save/restore round-trip is
        # exact. Using winfo_x()/winfo_y() instead caused the ~37px downward
        # creep per launch because those return client-area coords.
        cp["window"] = {"geometry": self.wm_geometry()}
        cp["paths"] = {"csv_folder": str(self.csv_path_var.get()), "model_folder": str(self.model_path_var.get())}
        cp["params"] = {
            "lookback": self.lookback.get(),
            "horizon": self.horizon.get(),
            "epochs": self.epochs.get(),
            "batch_size": self.batch_size.get(),
            "auto_tune": str(self.auto_tune_var.get()),
        }
        cp["features"] = {"base_features": self.feature_cols_var.get()}
        cp["indicators"] = {k: str(v.get()) for k, v in self.indicator_vars.items()}

        with open(self.INI_PATH, "w") as f:
            cp.write(f)

    def _create_ui(self):
        main_frame = CTkFrame(self, fg_color="transparent")
        main_frame.pack(fill="both", expand=True, padx=10, pady=5)
        title_frame = CTkFrame(main_frame, corner_radius=6)
        title_frame.pack(fill="x", pady=(0, 4))

        CTkLabel(title_frame, text="Copyright Su Nie | BSD-3C License | https://github.com/can87683", font=("Ubuntu", 16), text_color="yellow").pack(pady=(0, 4))
        CTkLabel(title_frame, text="🎯 TFT Model Trainer", font=("Ubuntu", 20, "bold")).pack(pady=4)

        UsageRow(main_frame)
        self._create_csv_section(main_frame)
        self._create_model_folder_section(main_frame)
        self._create_parameters_section(main_frame)
        self._create_features_section(main_frame)
        self._create_timeframe_section(main_frame)
        self._create_indicators_section(main_frame)
        self._create_controls_section(main_frame)
        self._create_status_section(main_frame)
        self._create_log_section(main_frame)

    def _create_csv_section(self, parent):
        frame = CTkFrame(parent, corner_radius=6)
        frame.pack(fill="x", pady=(0, 4))
        folder_frame = CTkFrame(frame, fg_color="transparent")
        folder_frame.pack(fill="x", padx=8, pady=3)
        CTkButton(folder_frame, text="📁 CSV Folder", command=self.select_csv_folder, width=120, height=28).pack(side="left", padx=(0, 6))
        self.csv_path_var = tk.StringVar(value=str(self.csv_folder))
        CTkEntry(folder_frame, textvariable=self.csv_path_var, state="readonly", height=28).pack(side="left", fill="x", expand=True)

    def _create_model_folder_section(self, parent):
        frame = CTkFrame(parent, corner_radius=6)
        frame.pack(fill="x", pady=(0, 4))
        folder_frame = CTkFrame(frame, fg_color="transparent")
        folder_frame.pack(fill="x", padx=8, pady=3)
        CTkButton(folder_frame, text="📁 Model Folder", command=self.select_model_folder, width=120, height=28).pack(side="left", padx=(0, 6))
        self.model_path_var = tk.StringVar(value=str(self.model_folder))
        CTkEntry(folder_frame, textvariable=self.model_path_var, state="readonly", height=28).pack(side="left", fill="x", expand=True)
        self.status_label = CTkLabel(frame, text="📊 Status: No folder selected", font=("Ubuntu", 10))
        self.status_label.pack(anchor="w", padx=8, pady=1)

    def _create_parameters_section(self, parent):
        frame = CTkFrame(parent, corner_radius=6)
        frame.pack(fill="x", pady=(0, 4))
        param_frame = CTkFrame(frame, fg_color="transparent")
        param_frame.pack(fill="x", padx=8, pady=3)

        self.lookback = tk.StringVar(value=str(Config.DEFAULT_LOOKBACK))
        self.horizon = tk.StringVar(value=str(Config.DEFAULT_HORIZON))
        self.epochs = tk.StringVar(value=str(Config.DEFAULT_EPOCHS))
        self.batch_size = tk.StringVar(value=str(Config.DEFAULT_BATCH_SIZE))

        row = CTkFrame(param_frame, fg_color="transparent")
        row.pack(fill="x", pady=1)
        for col in range(8):
            row.grid_columnconfigure(col, weight=1 if col % 2 else 0)
        CTkLabel(row, text="Lookback:", width=70).grid(row=0, column=0, padx=(0, 3), sticky="w")
        CTkEntry(row, textvariable=self.lookback, width=60).grid(row=0, column=1, padx=(0, 10), sticky="w")
        CTkLabel(row, text="Horizon:", width=70).grid(row=0, column=2, padx=(0, 3), sticky="w")
        CTkEntry(row, textvariable=self.horizon, width=60).grid(row=0, column=3, padx=(0, 10), sticky="w")
        CTkLabel(row, text="Epochs:", width=70).grid(row=0, column=4, padx=(0, 3), sticky="w")
        CTkEntry(row, textvariable=self.epochs, width=60).grid(row=0, column=5, padx=(0, 10), sticky="w")
        CTkLabel(row, text="Batch:", width=70).grid(row=0, column=6, padx=(0, 3), sticky="w")
        CTkEntry(row, textvariable=self.batch_size, width=60).grid(row=0, column=7, sticky="w")

        self.auto_tune_var = tk.BooleanVar(value=True)
        CTkCheckBox(
            param_frame,
            text="Auto-tune lookback / hidden size per timeframe (recommended)",
            variable=self.auto_tune_var,
            font=("Ubuntu", 10),
        ).pack(anchor="w", pady=(4, 0))

    def _create_features_section(self, parent):
        frame = CTkFrame(parent, corner_radius=6)
        frame.pack(fill="x", pady=(0, 4))
        feature_frame = CTkFrame(frame, fg_color="transparent")
        feature_frame.pack(fill="x", padx=8, pady=3)
        self.feature_cols_var = tk.StringVar(value="time,open,high,low,close,tick_volume")
        CTkLabel(feature_frame, text="Base Features (comma-separated):", font=("Ubuntu", 10)).pack(anchor="w", pady=(0, 2))
        CTkEntry(feature_frame, textvariable=self.feature_cols_var, height=28).pack(fill="x", pady=(0, 2))

    def _create_timeframe_section(self, parent):
        frame = CTkFrame(parent, corner_radius=6)
        frame.pack(fill="x", pady=(0, 4))
        tf_frame = CTkFrame(frame, fg_color="transparent")
        tf_frame.pack(fill="x", padx=8, pady=3)
        CTkLabel(tf_frame, text="Timeframes (one model each):", font=("Ubuntu", 10)).pack(anchor="w", pady=(0, 2))
        self.timeframe_checklist_frame = CTkFrame(tf_frame, fg_color="transparent")
        self.timeframe_checklist_frame.pack(fill="x")
        self.timeframe_hint_label = CTkLabel(
            self.timeframe_checklist_frame,
            text="Scan a CSV folder to list available timeframes.",
            font=("Ubuntu", 8),
            text_color="gray",
        )
        self.timeframe_hint_label.pack(anchor="w")

    def _rebuild_timeframe_checklist(self, timeframes):
        for widget in self.timeframe_checklist_frame.winfo_children():
            widget.destroy()
        self.timeframe_vars = {}
        if not timeframes:
            self.timeframe_hint_label = CTkLabel(
                self.timeframe_checklist_frame,
                text="No timeframes found.",
                font=("Ubuntu", 8),
                text_color="gray",
            )
            self.timeframe_hint_label.pack(anchor="w")
            return

        row = CTkFrame(self.timeframe_checklist_frame, fg_color="transparent")
        row.pack(fill="x")
        for i, tf in enumerate(timeframes):
            var = tk.BooleanVar(value=True)
            self.timeframe_vars[tf] = var
            CTkCheckBox(row, text=tf, variable=var, width=70, font=("Ubuntu", 10)).grid(row=0, column=i, padx=3, sticky="w")

    def _create_indicators_section(self, parent):
        frame = CTkFrame(parent, corner_radius=6)
        frame.pack(fill="x", pady=(0, 4))
        indicator_frame = CTkFrame(frame, fg_color="transparent")
        indicator_frame.pack(fill="x", padx=8, pady=3)
        CTkLabel(indicator_frame, text="Technical Indicators (inputs):", font=("Ubuntu", 10)).pack(anchor="w", pady=(0, 2))
        checkboxes_frame = CTkFrame(indicator_frame, fg_color="transparent")
        checkboxes_frame.pack(fill="x")

        self.indicator_vars = {}
        cols_per_row = 6
        for i, (label, var_name, default) in enumerate(self.INDICATOR_CONFIG):
            row_idx = i // cols_per_row
            col_idx = i % cols_per_row
            var = tk.BooleanVar(value=default)
            self.indicator_vars[var_name] = var
            CTkCheckBox(checkboxes_frame, text=label, variable=var, width=95, font=("Ubuntu", 10)).grid(
                row=row_idx, column=col_idx, padx=4, pady=2, sticky="w"
            )
        CTkLabel(
            indicator_frame,
            text="Uncheck indicators you don't want as model inputs",
            font=("Ubuntu", 8),
            text_color="gray",
        ).pack(anchor="w", pady=(4, 0))

    def _create_controls_section(self, parent):
        frame = CTkFrame(parent, corner_radius=6)
        frame.pack(fill="x", pady=(0, 4))
        button_frame = CTkFrame(frame, fg_color="transparent")
        button_frame.pack(pady=4)
        self.train_btn = CTkButton(
            button_frame, text="🚀 Start Training", command=self.start_training,
            width=130, height=34, fg_color="#2e7d32", hover_color="#1b5e20",
            font=("Ubuntu", 12, "bold"),
        )
        self.train_btn.pack(side="left", padx=4)
        self.stop_btn = CTkButton(
            button_frame, text="⏹️ Stop Training", command=self.stop_training,
            width=130, height=34, fg_color="#c62828", hover_color="#b71c1c",
            font=("Ubuntu", 12, "bold"), state="disabled",
        )
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
        self.log_text = CTkTextbox(frame, font=("Ubuntu Mono", 12))
        self.log_text.pack(fill="both", expand=True, padx=8, pady=3)

    def update_status(self, message):
        if self._is_closing:
            return
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
            self.model_folder = Path(folder)
            self.model_folder.mkdir(parents=True, exist_ok=True)
            self.gui_logger.log(f"💾 Model folder set to: {self.model_folder}")

    def scan_data_source(self):
        csv_folder = Path(self.csv_path_var.get())
        if not csv_folder.exists():
            self.gui_logger.log(f"❌ CSV folder not found: {csv_folder}", "error")
            self.status_label.configure(text="❌ Folder not found")
            self.train_btn.configure(state="disabled")
            self._rebuild_timeframe_checklist([])
            return

        self.gui_logger.log(f"📁 Scanning CSV folder: {csv_folder}")
        symbols, timeframes = self.data_preparer.scan_folder(csv_folder)
        if not symbols:
            self.gui_logger.log("⚠️ No symbols found", "warning")
            self.status_label.configure(text="⚠️ No symbols found")
            self.train_btn.configure(state="disabled")
            self._rebuild_timeframe_checklist([])
            return

        self._rebuild_timeframe_checklist(timeframes)
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

        selected_timeframes = [tf for tf, var in self.timeframe_vars.items() if var.get()]
        if not selected_timeframes:
            self.gui_logger.log("❌ Select at least one timeframe", "error")
            return

        self.is_training = True
        self.stop_flag = False
        self.train_btn.configure(state="disabled")
        self.stop_btn.configure(state="normal")

        def training_task():
            self.gui_logger.log("=" * 60)
            self.gui_logger.log("🚀 Starting Per-Timeframe TFT Training")
            self.gui_logger.log("=" * 60)
            self.gui_logger.log(f"📂 CSV Folder: {csv_folder}")
            self.gui_logger.log(f"💾 System RAM: {MemoryOptimizer.get_system_memory_available_gb():.1f} GB available")

            flags = {k: v.get() for k, v in self.indicator_vars.items()}
            self.data_preparer.set_indicator_flags(**flags)
            enabled_list = [k.replace("enable_", "").upper() for k, v in self.indicator_vars.items() if v.get()]
            self.gui_logger.log(f"📊 Indicators enabled ({len(enabled_list)}): {', '.join(enabled_list)}")

            feature_input = self.feature_cols_var.get().strip()
            if feature_input:
                feature_cols = [f.strip() for f in feature_input.split(',') if f.strip()]
            else:
                feature_cols = Config.DEFAULT_FEATURES

            lookback = int(self.lookback.get())
            horizon = int(self.horizon.get())
            batch_size = int(self.batch_size.get())
            epochs = int(self.epochs.get())

            results = {}

            for timeframe in selected_timeframes:
                if self.stop_flag or self._is_closing:
                    self.gui_logger.log("⏹️ Training stopped by user", "warning")
                    break

                self.gui_logger.log("-" * 60)
                self.gui_logger.log(f"▶️ Timeframe: {timeframe}")
                self.update_status(f"[{timeframe}] Building dataset and training...")

                trainer = StreamingTFTTrainer(self.gui_logger.log, timeframe=timeframe)
                self.current_trainer = trainer

                success = trainer.train_streaming(
                    None, 0, epochs, batch_size, lookback, horizon,
                    csv_folder=csv_folder,
                    feature_cols=feature_cols,
                    data_preparer=self.data_preparer,
                    timeframe=timeframe,
                    auto_tune_lookback=self.auto_tune_var.get(),
                )

                if success and not trainer.stop_flag:
                    filename = f"{Config.MODEL_PREFIX}_{timeframe.lower()}{Config.MODEL_EXT}"
                    if trainer.save_model(filename):
                        self.gui_logger.log(f"✅ [{timeframe}] Model saved: {filename}", "success")
                        results[timeframe] = "trained"
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

            self.is_training = False
            self.current_trainer = None
            if not self._is_closing:
                self.train_btn.configure(state="normal")
                self.stop_btn.configure(state="disabled")

        threading.Thread(target=training_task, daemon=True).start()

    def stop_training(self):
        self.stop_flag = True
        if self.current_trainer is not None:
            self.current_trainer.stop_training()
        self.gui_logger.log("⏹️ Stopping training...", "warning")

    def _on_closing(self):
        if self._is_closing:
            return
        self._is_closing = True
        self._save_state()

        def do_exit():
            self.gui_logger.disable_console_redirect()
            self.destroy()

        def confirm_exit():
            if self.is_training:
                if messagebox.askyesno("Training in Progress", "Stop training and quit?"):
                    self.stop_flag = True
                    if self.current_trainer is not None:
                        self.current_trainer.stop_training()
                    self.after(200, do_exit)
                else:
                    self._is_closing = False
            else:
                if messagebox.askyesno("Quit", "Do you want to quit AI Predict Train (TFT)?"):
                    do_exit()
                else:
                    self._is_closing = False

        self.after(0, confirm_exit)


if __name__ == "__main__":
    app = TFTTrainerGUI()
    app.mainloop()