"""
customtkinter numpy pandas psutil setproctitle xgboost
"""
#!/usr/bin/env python3
# ai_predict_train_xgboost.py
# Copyright Su Nie | BSD-3C License | https://github.com/can87683


import setproctitle
setproctitle.setproctitle("ai_predict_train_xgboost")

import customtkinter as ctk
from customtkinter import CTk, CTkFrame, CTkLabel, CTkButton, CTkEntry, CTkTextbox, CTkCheckBox
import tkinter as tk
from tkinter import filedialog, messagebox
import threading
import subprocess
import configparser
from pathlib import Path
import numpy as np
import pandas as pd
import pickle
import gc
import psutil
from datetime import datetime
import sys

import xgboost as xgb


class Config:
    HOME = Path.home()
    MODEL_DIR = HOME / "ai_model_ml"
    DATASET_DIR = HOME / "mt5_data"

    MAX_ROWS_PER_SYMBOL = 20000
    DEFAULT_FEATURES = ['time', 'open', 'high', 'low', 'close', 'tick_volume']

    DEFAULT_HORIZON = 10
    DEFAULT_ITERATIONS = 1000
    DEFAULT_LEARNING_RATE = 0.05
    DEFAULT_MAX_DEPTH = 6
    DEFAULT_L2_REG = 1.0
    DEFAULT_SUBSAMPLE = 0.8
    DEFAULT_COLSAMPLE = 0.8
    EARLY_STOPPING_ROUNDS = 50
    VALIDATION_FRACTION = 0.15

    TIMEFRAMES = ['M5', 'M15', 'H1', 'H4', 'D1', 'W1']
    MODEL_PREFIX = "ai_predict_xgboost"
    MODEL_EXT = ".ubj"
    METADATA_EXT = ".pkl"

    APP_TITLE = "🚀 XGBoost Model Trainer"
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


class GPUManager:
    def __init__(self, verbose=True, log_callback=None):
        self.verbose = verbose
        self.log_callback = log_callback
        self.gpu_available = False
        self.gpu_name = None
        self.gpu_memory_gb = 0.0
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

    def preferred_devices(self):
        return ["cuda", "cpu"] if self.gpu_available else ["cpu"]

    def log_system_info(self):
        self._log("=" * 60)
        self._log("🖥️ System & Library Information")
        self._log("=" * 60)
        self._log(f"🐍 Python: {sys.version.split()[0]}")
        self._log(f"🚀 XGBoost: {xgb.__version__}")
        self._log(f"🔌 CUDA Build: {xgb.build_info().get('CUDA_VERSION', 'N/A') if hasattr(xgb, 'build_info') else 'N/A'}")
        if self.gpu_available:
            self._log(f"🎮 GPU Device: {self.gpu_name}")
            self._log(f"💾 GPU Memory: {self.gpu_memory_gb:.2f} GB")
        else:
            self._log("⚠️ No GPU detected. Will use CPU.")
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
    def __init__(self, gui_logger, stream):
        self.gui_logger = gui_logger
        self.stream = stream

    def write(self, msg):
        if msg and msg.strip():
            self.stream.write(msg)
            self.stream.flush()
            if self.gui_logger:
                self.gui_logger.log(msg.strip(), "console")

    def flush(self):
        self.stream.flush()


class MemoryOptimizer:
    @staticmethod
    def get_system_memory_available_gb():
        return psutil.virtual_memory().available / 1e9

    @staticmethod
    def clear_cache():
        gc.collect()


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
    def _add_atr(df, p=14):
        pc = df['close'].shift(1)
        df['atr'] = pd.concat([
            df['high'] - df['low'],
            (df['high'] - pc).abs(),
            (df['low'] - pc).abs()
        ], axis=1).max(axis=1).ewm(alpha=1/p, min_periods=p, adjust=False).mean()
        return df

    @staticmethod
    def _add_bollinger(df, p=20, s=2):
        mid = df['close'].rolling(p).mean()
        std = df['close'].rolling(p).std()
        br = (mid + s * std - (mid - s * std)).replace(0, np.nan)
        df['bb_percent_b'] = (df['close'] - (mid - s * std)) / br
        df['bb_bandwidth'] = br / mid
        return df

    @staticmethod
    def _add_cci(df, p=20):
        tp = (df['high'] + df['low'] + df['close']) / 3
        md = tp.rolling(p).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True)
        df['cci'] = (tp - tp.rolling(p).mean()) / (0.015 * md.replace(0, np.nan))
        return df

    @staticmethod
    def _add_cmf(df, p=20):
        mfv = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low']).replace(0, np.nan) * df['tick_volume']
        df['cmf'] = mfv.rolling(p).sum() / df['tick_volume'].rolling(p).sum().replace(0, np.nan)
        return df

    @staticmethod
    def _add_donchian(df, p=20):
        h = df['high'].rolling(p).max()
        l = df['low'].rolling(p).min()
        r = (h - l).replace(0, np.nan)
        df['donchian_position'] = (df['close'] - l) / r
        df['donchian_width'] = r / df['close']
        return df

    @staticmethod
    def _add_ema(df):
        for n in [9, 21, 50, 200]:
            df[f'close_vs_ema_{n}'] = df['close'] / df['close'].ewm(span=n, adjust=False).mean() - 1
        df['ema_9_21_spread'] = df['close_vs_ema_9'] - df['close_vs_ema_21']
        df['ema_21_50_spread'] = df['close_vs_ema_21'] - df['close_vs_ema_50']
        df['ema_21_slope_5'] = df['close'].ewm(span=21, adjust=False).mean().diff(5) / df['close']
        return df

    @staticmethod
    def _add_force_index(df, p=13):
        df['force_index'] = (df['close'].diff() * df['tick_volume']).ewm(span=p, adjust=False).mean()
        return df

    @staticmethod
    def _add_macd(df, f=12, s=26, sig=9):
        ml = df['close'].ewm(span=f, adjust=False).mean() - df['close'].ewm(span=s, adjust=False).mean()
        df['macd'] = ml
        df['macd_signal'] = ml.ewm(span=sig, adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        return df

    @staticmethod
    def _add_mfi(df, p=14):
        tp = (df['high'] + df['low'] + df['close']) / 3
        rmf = tp * df['tick_volume']
        pos = rmf.where(tp > tp.shift(1), 0.0).rolling(p).sum()
        neg = rmf.where(tp < tp.shift(1), 0.0).rolling(p).sum()
        df['mfi'] = 100 - (100 / (1 + pos / neg.replace(0, np.nan)))
        return df

    @staticmethod
    def _add_obv(df):
        df['obv'] = (np.sign(df['close'].diff()).fillna(0) * df['tick_volume']).cumsum()
        df['obv_slope_5'] = df['obv'].diff(5) / df['tick_volume'].rolling(20).sum().replace(0, np.nan)
        return df

    @staticmethod
    def _add_price_structure(df):
        r = (df['high'] - df['low']).replace(0, np.nan)
        df['hl_range_pct'] = r / df['close']
        df['body_pct'] = (df['close'] - df['open']).abs() / df['close']
        df['body_range_ratio'] = (df['close'] - df['open']).abs() / r
        df['close_location_value'] = (df['close'] - df['low']) / r
        mx = df[['open', 'close']].max(axis=1)
        mn = df[['open', 'close']].min(axis=1)
        df['upper_shadow_pct'] = (df['high'] - mx) / df['close']
        df['lower_shadow_pct'] = (mn - df['low']) / df['close']
        return df

    @staticmethod
    def _add_realized_vol(df):
        r = np.log(df['close'] / df['close'].shift(1))
        df['realized_vol_20'] = r.rolling(20).std()
        df['realized_vol_100'] = r.rolling(100).std()
        return df

    @staticmethod
    def _add_roc(df):
        for n in [5, 10, 20]:
            df[f'roc_{n}'] = df['close'].pct_change(n) * 100
        return df

    @staticmethod
    def _add_rsi(df, p=14):
        d = df['close'].diff()
        g = d.clip(lower=0)
        l = -d.clip(upper=0)
        avg_gain = g.ewm(alpha=1/p, min_periods=p, adjust=False).mean()
        avg_loss = l.ewm(alpha=1/p, min_periods=p, adjust=False).mean()
        df['rsi'] = 100 - (100 / (1 + avg_gain / avg_loss.replace(0, np.nan)))
        return df

    @staticmethod
    def _add_rvwap(df, p=20):
        tp = (df['high'] + df['low'] + df['close']) / 3
        df['rvwap'] = (tp * df['tick_volume']).rolling(p).sum() / df['tick_volume'].rolling(p).sum().replace(0, np.nan)
        df['close_vs_rvwap'] = df['close'] / df['rvwap'] - 1
        return df

    @staticmethod
    def _add_stochastic(df, p=14, s=3):
        l = df['low'].rolling(p).min()
        h = df['high'].rolling(p).max()
        df['stoch_k'] = 100 * (df['close'] - l) / (h - l).replace(0, np.nan)
        df['stoch_d'] = df['stoch_k'].rolling(s).mean()
        return df

    @staticmethod
    def _add_volume_metrics(df):
        m = df['tick_volume'].rolling(20).mean()
        s = df['tick_volume'].rolling(20).std()
        df['volume_ratio_20'] = df['tick_volume'] / m.replace(0, np.nan)
        df['volume_zscore_20'] = (df['tick_volume'] - m) / s.replace(0, np.nan)
        df['return_x_volume_1'] = df['close'].pct_change() * df['tick_volume']
        df['amihud_illiquidity_20'] = (df['close'].pct_change().abs() / (df['close'] * df['tick_volume']).replace(0, np.nan)).rolling(20).mean()
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
        self.log_widget.configure(font=("Ubuntu Mono", 12))
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

    def load_csv_data(self, symbol, timeframe, folder):
        csv_file = Path(folder) / symbol / f"{symbol}_{timeframe}.csv"
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

    def scan_folder(self, folder):
        folder = Path(folder)
        if not folder.exists():
            return [], []
        symbols = []
        all_timeframes = set()
        for d in sorted(folder.iterdir()):
            if not d.is_dir():
                continue
            symbol = d.name
            found = []
            for timeframe in Config.TIMEFRAMES:
                if (d / f"{symbol}_{timeframe}.csv").exists():
                    found.append(timeframe)
                    all_timeframes.add(timeframe)
            if found:
                symbols.append((symbol, found))
                self._log(f"   ✓ {symbol}: {', '.join(found)}")
        return symbols, sorted(all_timeframes)

    def _prepare_single_series(self, df, symbol, horizon, requested_features):
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
        df['returns_5'] = df['close'].pct_change(5)
        df['returns_10'] = df['close'].pct_change(10)

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

    def build_dataframe_for_timeframe(self, folder, feature_cols, horizon, timeframe):
        symbols, _ = self.scan_folder(folder)
        symbols_with_tf = [s for s, tfs in symbols if timeframe in tfs]
        if not symbols_with_tf:
            raise ValueError(f"No symbols found with timeframe {timeframe} in {folder}")

        frames = []
        all_features = []
        for symbol in symbols_with_tf:
            df = self.load_csv_data(symbol, timeframe, folder)
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
        result["group_id"] = result["group_id"].astype(str)
        result = result.sort_values(["group_id", "time_idx"]).reset_index(drop=True)

        self.feature_cols = all_features
        self._log(f"✅ [{timeframe}] Prepared {len(result):,} rows, {result['group_id'].nunique()} symbols, {len(self.feature_cols)} features")
        return result


class StreamingXGBoostTrainer:
    def __init__(self, log_callback=None, timeframe=None):
        self.log_callback = log_callback or print
        self.timeframe = timeframe
        self.gpu_manager = GPUManager(verbose=False)
        self.model = None
        self.is_trained = False
        self.stop_flag = False
        self.device_used = "cpu"
        self.iterations = Config.DEFAULT_ITERATIONS
        self.learning_rate = Config.DEFAULT_LEARNING_RATE
        self.max_depth = Config.DEFAULT_MAX_DEPTH
        self.reg_lambda = Config.DEFAULT_L2_REG
        self.subsample = Config.DEFAULT_SUBSAMPLE
        self.colsample = Config.DEFAULT_COLSAMPLE
        self.training_history = {}
        preferred = " -> ".join(self.gpu_manager.preferred_devices())
        self._log(f"🚀 XGBoost [{timeframe}] initialized. Device attempt order: {preferred.upper()}")

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
            n = len(group)
            if n < 5:
                train_parts.append(group)
                continue
            n_val = max(1, int(n * Config.VALIDATION_FRACTION))
            n_val = min(n_val, n - 1)
            train_parts.append(group.iloc[:-n_val])
            val_parts.append(group.iloc[-n_val:])
        train_df = pd.concat(train_parts, ignore_index=True)
        val_df = pd.concat(val_parts, ignore_index=True)
        return train_df, val_df

    def _compute_directional_accuracy(self, model, X_val, val_df):
        if "close" not in val_df.columns:
            return None
        preds = model.predict(X_val)
        last_close = val_df["close"].to_numpy()
        actual_target = val_df["target"].to_numpy()
        actual_direction = np.sign(actual_target - last_close)
        predicted_direction = np.sign(preds - last_close)
        valid_mask = actual_direction != 0
        if valid_mask.sum() == 0:
            return None
        return float(((actual_direction == predicted_direction) & valid_mask).sum() / valid_mask.sum())

    def _build_model(self, device, iterations, learning_rate, max_depth):
        return xgb.XGBRegressor(
            n_estimators=iterations,
            learning_rate=learning_rate,
            max_depth=max_depth,
            reg_lambda=self.reg_lambda,
            subsample=self.subsample,
            colsample_bytree=self.colsample,
            objective="reg:squarederror",
            eval_metric="rmse",
            early_stopping_rounds=Config.EARLY_STOPPING_ROUNDS,
            tree_method="hist",
            device=device,
            enable_categorical=True,
            random_state=Config.SEED,
            n_jobs=-1,
            verbosity=0,
        )

    def train_single_timeframe(self, data, feature_cols, timeframe, iterations=1000, learning_rate=0.05, max_depth=6, **kwargs):
        self.stop_flag = False
        self.timeframe = timeframe

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

        self._log(f"📊 [{timeframe}] Train: {len(train_df):,} | Val: {len(val_df):,} | Depth: {max_depth}, Iter: {iterations}")
        last_error = None
        fitted = None
        used_device = None

        for device in self.gpu_manager.preferred_devices():
            if self.stop_flag:
                break
            self._log(f"🚀 [{timeframe}] Attempting device='{device}'...")
            try:
                model = self._build_model(device, iterations, learning_rate, max_depth)
                fitted = model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
                used_device = device
                break
            except Exception as exc:
                last_error = exc
                self._log(f"⚠️ [{timeframe}] device='{device}' failed: {exc}")

        if fitted is None:
            raise last_error if last_error else RuntimeError("No device succeeded in training")

        self.model = fitted
        self.device_used = used_device
        self.is_trained = True
        self._log(f"✅ [{timeframe}] Trained using device='{used_device}'")

        directional_accuracy = self._compute_directional_accuracy(self.model, X_val, val_df)
        if directional_accuracy is not None:
            self._log(f"🎯 [{timeframe}] Directional accuracy: {directional_accuracy * 100:.1f}% (50% = coin flip)")

        self.training_history = {
            "timeframe": timeframe,
            "device_used": used_device,
            "best_iteration": getattr(self.model, "best_iteration", iterations),
            "best_score": getattr(self.model, "best_score", None),
            "train_rows": len(train_df),
            "val_rows": len(val_df),
            "directional_accuracy": directional_accuracy,
        }
        self._log(f"🏁 [{timeframe}] Training complete (device={used_device})")
        return True

    def save_model(self, model_dir, filename_stem, feature_cols, horizon, timeframe):
        if not self.is_trained or self.model is None:
            self._log("⚠️ No trained model available")
            return False

        output_dir = Path(model_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        model_path = output_dir / f"{filename_stem}{Config.MODEL_EXT}"
        metadata_path = output_dir / f"{filename_stem}{Config.METADATA_EXT}"

        self.model.save_model(str(model_path))

        metadata = {
            "model_type": "xgboost.XGBRegressor",
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

        self._log(f"✅ XGBoost model saved: {model_path}")
        self._log(f"✅ Metadata saved: {metadata_path}")
        return True


class XGBoostTrainerGUI(CTk):
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
        self.is_training = False
        self.stop_flag = False
        self.current_trainer = None

        self.data_preparer = StreamingDataPreparer(self.gui_logger.log)

        self._create_ui()
        self._load_state()  # applied while the window is hidden
        self.gui_logger.set_log_widget(self.log_text)

        self.gpu_manager.log_system_info()

        self.after(500, self.scan_data)
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
                self.csv_var.set(csv)
                self.csv_folder = Path(csv)
            if model:
                self.model_var.set(model)
                self.model_folder = Path(model)

        if cp.has_section("params"):
            self.horizon.set(cp.get("params", "horizon", fallback=str(Config.DEFAULT_HORIZON)))
            self.iters.set(cp.get("params", "iterations", fallback=str(Config.DEFAULT_ITERATIONS)))
            self.lr.set(cp.get("params", "learning_rate", fallback=str(Config.DEFAULT_LEARNING_RATE)))
            self.depth.set(cp.get("params", "max_depth", fallback=str(Config.DEFAULT_MAX_DEPTH)))

        if cp.has_section("features"):
            self.feat_var.set(cp.get("features", "base_features", fallback="time,open,high,low,close,tick_volume"))

        if cp.has_section("timeframes"):
            for tf in Config.TIMEFRAMES:
                if tf in self.tf_vars:
                    self.tf_vars[tf].set(cp.getboolean("timeframes", tf, fallback=True))

        if cp.has_section("indicators"):
            for var_name in self.ind_vars:
                self.ind_vars[var_name].set(cp.getboolean("indicators", var_name, fallback=True))

    def _save_state(self):
        cp = configparser.ConfigParser()
        # Store wm_geometry() directly. It is frame-relative (same coordinate
        # system that geometry() consumes), so a save/restore round-trip is
        # exact. Using winfo_x()/winfo_y() instead caused the ~37px downward
        # creep per launch because those return client-area coords.
        cp["window"] = {"geometry": self.wm_geometry()}
        cp["paths"] = {"csv_folder": str(self.csv_var.get()), "model_folder": str(self.model_var.get())}
        cp["params"] = {
            "horizon": self.horizon.get(),
            "iterations": self.iters.get(),
            "learning_rate": self.lr.get(),
            "max_depth": self.depth.get(),
        }
        cp["features"] = {"base_features": self.feat_var.get()}
        cp["timeframes"] = {tf: str(var.get()) for tf, var in self.tf_vars.items()}
        cp["indicators"] = {k: str(v.get()) for k, v in self.ind_vars.items()}

        with open(self.INI_PATH, "w") as f:
            cp.write(f)

    def _create_ui(self):
        main_frame = CTkFrame(self, fg_color="transparent")
        main_frame.pack(fill="both", expand=True, padx=10, pady=5)
        title_frame = CTkFrame(main_frame, corner_radius=6)
        title_frame.pack(fill="x", pady=(0, 4))

        CTkLabel(title_frame, text="Copyright Su Nie | BSD-3C License | https://github.com/can87683", font=("Ubuntu", 16), text_color="yellow").pack(pady=(0, 4))
        CTkLabel(title_frame, text="🎯 XGBoost Model Trainer", font=("Ubuntu", 20, "bold")).pack(pady=4)

        UsageRow(main_frame)
        self._create_csv(main_frame)
        self._create_model(main_frame)
        self._create_params(main_frame)
        self._create_feats(main_frame)
        self._create_tfs(main_frame)
        self._create_inds(main_frame)
        self._create_ctrls(main_frame)
        self._create_stat(main_frame)
        self._create_log(main_frame)

    def _create_csv(self, parent):
        frame = CTkFrame(parent, corner_radius=6)
        frame.pack(fill="x", pady=(0, 4))
        folder_frame = CTkFrame(frame, fg_color="transparent")
        folder_frame.pack(fill="x", padx=8, pady=3)
        CTkButton(folder_frame, text="📁 CSV Folder", command=self.select_csv, width=120, height=28).pack(side="left", padx=(0, 6))
        self.csv_var = tk.StringVar(value=str(self.csv_folder))
        CTkEntry(folder_frame, textvariable=self.csv_var, state="readonly", height=28).pack(side="left", fill="x", expand=True)

    def _create_model(self, parent):
        frame = CTkFrame(parent, corner_radius=6)
        frame.pack(fill="x", pady=(0, 4))
        folder_frame = CTkFrame(frame, fg_color="transparent")
        folder_frame.pack(fill="x", padx=8, pady=3)
        CTkButton(folder_frame, text="📁 Model Folder", command=self.select_model, width=120, height=28).pack(side="left", padx=(0, 6))
        self.model_var = tk.StringVar(value=str(self.model_folder))
        CTkEntry(folder_frame, textvariable=self.model_var, state="readonly", height=28).pack(side="left", fill="x", expand=True)
        self.stat_lbl = CTkLabel(frame, text="📊 Status: Ready", font=("Ubuntu", 10))
        self.stat_lbl.pack(anchor="w", padx=8, pady=1)

    def _create_params(self, parent):
        frame = CTkFrame(parent, corner_radius=6)
        frame.pack(fill="x", pady=(0, 4))
        param_frame = CTkFrame(frame, fg_color="transparent")
        param_frame.pack(fill="x", padx=8, pady=3)

        self.horizon = tk.StringVar(value=str(Config.DEFAULT_HORIZON))
        self.iters = tk.StringVar(value=str(Config.DEFAULT_ITERATIONS))
        self.lr = tk.StringVar(value=str(Config.DEFAULT_LEARNING_RATE))
        self.depth = tk.StringVar(value=str(Config.DEFAULT_MAX_DEPTH))

        row = CTkFrame(param_frame, fg_color="transparent")
        row.pack(fill="x", pady=1)
        for col in range(8):
            row.grid_columnconfigure(col, weight=1 if col % 2 else 0)
        CTkLabel(row, text="Horizon:", width=70).grid(row=0, column=0, padx=(0, 3), sticky="w")
        CTkEntry(row, textvariable=self.horizon, width=60).grid(row=0, column=1, padx=(0, 10), sticky="w")
        CTkLabel(row, text="Iterations:", width=70).grid(row=0, column=2, padx=(0, 3), sticky="w")
        CTkEntry(row, textvariable=self.iters, width=60).grid(row=0, column=3, padx=(0, 10), sticky="w")
        CTkLabel(row, text="LR:", width=70).grid(row=0, column=4, padx=(0, 3), sticky="w")
        CTkEntry(row, textvariable=self.lr, width=60).grid(row=0, column=5, padx=(0, 10), sticky="w")
        CTkLabel(row, text="Depth:", width=70).grid(row=0, column=6, padx=(0, 3), sticky="w")
        CTkEntry(row, textvariable=self.depth, width=60).grid(row=0, column=7, sticky="w")

    def _create_feats(self, parent):
        frame = CTkFrame(parent, corner_radius=6)
        frame.pack(fill="x", pady=(0, 4))
        feature_frame = CTkFrame(frame, fg_color="transparent")
        feature_frame.pack(fill="x", padx=8, pady=3)
        self.feat_var = tk.StringVar(value="time,open,high,low,close,tick_volume")
        CTkLabel(feature_frame, text="Base Features:", font=("Ubuntu", 10)).pack(anchor="w", pady=(0, 2))
        CTkEntry(feature_frame, textvariable=self.feat_var, height=28).pack(fill="x", pady=(0, 2))

    def _create_tfs(self, parent):
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
            CTkCheckBox(checkboxes_frame, text=tf, variable=var, width=70, font=("Ubuntu", 10)).grid(row=0, column=i, padx=3, sticky="w")

    def _create_inds(self, parent):
        frame = CTkFrame(parent, corner_radius=6)
        frame.pack(fill="x", pady=(0, 4))
        indicator_frame = CTkFrame(frame, fg_color="transparent")
        indicator_frame.pack(fill="x", padx=8, pady=3)
        CTkLabel(indicator_frame, text="Technical Indicators (inputs):", font=("Ubuntu", 10)).pack(anchor="w", pady=(0, 2))
        checkboxes_frame = CTkFrame(indicator_frame, fg_color="transparent")
        checkboxes_frame.pack(fill="x")

        self.ind_vars = {}
        cols_per_row = 6
        for i, (label, var_name, default) in enumerate(self.INDICATOR_CONFIG):
            row_idx = i // cols_per_row
            col_idx = i % cols_per_row
            var = tk.BooleanVar(value=default)
            self.ind_vars[var_name] = var
            CTkCheckBox(checkboxes_frame, text=label, variable=var, width=95, font=("Ubuntu", 10)).grid(
                row=row_idx, column=col_idx, padx=4, pady=2, sticky="w"
            )
        CTkLabel(
            indicator_frame,
            text="Uncheck indicators you don't want as model inputs",
            font=("Ubuntu", 8),
            text_color="gray",
        ).pack(anchor="w", pady=(4, 0))

    def _create_ctrls(self, parent):
        frame = CTkFrame(parent, corner_radius=6)
        frame.pack(fill="x", pady=(0, 4))
        button_frame = CTkFrame(frame, fg_color="transparent")
        button_frame.pack(pady=4)
        self.train_btn = CTkButton(
            button_frame, text="🚀 Start Training", command=self.start_train,
            width=130, height=34, fg_color="#2e7d32", hover_color="#1b5e20",
            font=("Ubuntu", 12, "bold"),
        )
        self.train_btn.pack(side="left", padx=4)
        self.stop_btn = CTkButton(
            button_frame, text="⏹️ Stop Training", command=self.stop_train,
            width=130, height=34, fg_color="#c62828", hover_color="#b71c1c",
            font=("Ubuntu", 12, "bold"), state="disabled",
        )
        self.stop_btn.pack(side="left", padx=4)

    def _create_stat(self, parent):
        frame = CTkFrame(parent, corner_radius=6)
        frame.pack(fill="x", pady=(0, 4))
        self.stat_txt = CTkTextbox(frame, height=50, font=("Ubuntu Mono", 10))
        self.stat_txt.pack(fill="x", padx=8, pady=3)
        self.stat_txt.insert("1.0", "Ready")
        self.stat_txt.configure(state="disabled")

    def _create_log(self, parent):
        frame = CTkFrame(parent, corner_radius=6)
        frame.pack(fill="both", expand=True)
        self.log_text = CTkTextbox(frame, font=("Ubuntu Mono", 12))
        self.log_text.pack(fill="both", expand=True, padx=8, pady=3)

    def update_stat(self, message):
        if self._is_closing:
            return
        self.stat_txt.configure(state="normal")
        self.stat_txt.delete("1.0", "end")
        self.stat_txt.insert("1.0", message)
        self.stat_txt.configure(state="disabled")
        self.stat_txt.update_idletasks()

    def select_csv(self):
        folder = filedialog.askdirectory(initialdir=str(self.csv_folder))
        if folder:
            self.csv_var.set(folder)
            self.scan_data()

    def select_model(self):
        folder = filedialog.askdirectory(initialdir=str(self.model_folder))
        if folder:
            self.model_var.set(folder)
            self.model_folder = Path(folder)
            self.model_folder.mkdir(parents=True, exist_ok=True)
            self.gui_logger.log(f"💾 Model folder set to: {self.model_folder}")

    def scan_data(self):
        csv_folder = Path(self.csv_var.get())
        if not csv_folder.exists():
            self.gui_logger.log(f"❌ CSV folder not found: {csv_folder}", "error")
            self.stat_lbl.configure(text="❌ Folder not found")
            self.train_btn.configure(state="disabled")
            return
        self.gui_logger.log(f"📁 Scanning CSV folder: {csv_folder}")
        symbols, timeframes = self.data_preparer.scan_folder(csv_folder)
        if not symbols:
            self.gui_logger.log("⚠️ No symbols found", "warning")
            self.stat_lbl.configure(text="⚠️ No symbols found")
            self.train_btn.configure(state="disabled")
            return
        self.stat_lbl.configure(text=f"✅ Found {len(symbols)} symbols, {len(timeframes)} timeframes")
        self.gui_logger.log(f"✅ Loaded {len(symbols)} symbols", "success")
        self.train_btn.configure(state="normal")

    def start_train(self):
        if self.is_training:
            self.gui_logger.log("⏳ Training already in progress", "warning")
            return

        csv_folder = self.csv_var.get()
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
            self.gui_logger.log("=" * 60)
            self.gui_logger.log("🚀 Starting Per-Timeframe XGBoost Training")
            self.gui_logger.log("=" * 60)
            self.gui_logger.log(f"📂 CSV Folder: {csv_folder}")
            self.gui_logger.log(f"💾 System RAM: {MemoryOptimizer.get_system_memory_available_gb():.1f} GB available")

            flags = {k: v.get() for k, v in self.ind_vars.items()}
            self.data_preparer.set_indicator_flags(**flags)
            enabled = [k.replace("enable_", "").upper() for k, v in self.ind_vars.items() if v.get()]
            self.gui_logger.log(f"📊 Indicators enabled ({len(enabled)}): {', '.join(enabled)}")

            feature_input = self.feat_var.get().strip()
            if feature_input:
                feature_cols = [f.strip() for f in feature_input.split(',') if f.strip()]
            else:
                feature_cols = Config.DEFAULT_FEATURES

            horizon = int(self.horizon.get())
            iterations = int(self.iters.get())
            learning_rate = float(self.lr.get())
            max_depth = int(self.depth.get())
            model_folder = self.model_var.get()

            results = {}

            for timeframe in selected_timeframes:
                if self.stop_flag or self._is_closing:
                    self.gui_logger.log("⏹️ Training stopped by user", "warning")
                    break

                self.gui_logger.log("-" * 60)
                self.gui_logger.log(f"▶️ Timeframe: {timeframe}")
                self.update_stat(f"[{timeframe}] Building dataset...")

                data = self.data_preparer.build_dataframe_for_timeframe(csv_folder, feature_cols, horizon, timeframe)
                feature_columns = self.data_preparer.feature_cols
                self.update_stat(f"[{timeframe}] Training XGBoost model...")

                trainer = StreamingXGBoostTrainer(self.gui_logger.log, timeframe=timeframe)
                self.current_trainer = trainer

                success = trainer.train_single_timeframe(
                    data=data,
                    feature_cols=feature_columns,
                    timeframe=timeframe,
                    iterations=iterations,
                    learning_rate=learning_rate,
                    max_depth=max_depth,
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
            self.update_stat(f"Done. {summary}")

            self.is_training = False
            self.current_trainer = None
            if not self._is_closing:
                self.train_btn.configure(state="normal")
                self.stop_btn.configure(state="disabled")

        threading.Thread(target=training_task, daemon=True).start()

    def stop_train(self):
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
                if messagebox.askyesno("Quit", "Do you want to quit AI Predict Train (XGBoost)?"):
                    do_exit()
                else:
                    self._is_closing = False

        self.after(0, confirm_exit)


if __name__ == "__main__":
    app = XGBoostTrainerGUI()
    app.mainloop()