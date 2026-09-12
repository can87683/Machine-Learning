#!/usr/bin/env python3
# ai_predict_train_xgboost_ctk.py
# BSD-C-3 License


import setproctitle
setproctitle.setproctitle("ai_predict_train_xgboost_ctk")

import customtkinter as ctk
from customtkinter import CTk, CTkFrame, CTkLabel, CTkButton, CTkEntry, CTkTextbox, CTkCheckBox
import tkinter as tk
from tkinter import filedialog, messagebox
import threading
import subprocess
from pathlib import Path
import numpy as np
import pandas as pd
import pickle
import gc
import psutil
from datetime import datetime
import traceback
import sys
import warnings
import xgboost as xgb

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


class Config:
    HOME = Path.home()
    MODEL_DIR = HOME / "ai_model_ml"
    DATASET_DIR = HOME / "mt5_data"

    MAX_ROWS_PER_SYMBOL = 20000
    DEFAULT_FEATURES = ['time', 'open', 'high', 'low', 'close', 'volume']

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

    @classmethod
    def ensure_directories(cls):
        cls.MODEL_DIR.mkdir(parents=True, exist_ok=True)
        cls.DATASET_DIR.mkdir(parents=True, exist_ok=True)

Config.ensure_directories()
ctk.set_appearance_mode(Config.THEME)
ctk.set_default_color_theme(Config.COLOR_THEME)
MODEL_DIR = Config.MODEL_DIR
DATASET_DIR = Config.DATASET_DIR


class GPUManager:
    def __init__(self, verbose=True):
        self.verbose = verbose
        self.gpu_available = False
        self._detect_gpu()

    def _log(self, msg):
        if self.verbose: print(f"[GPUManager] {msg}")

    def _detect_gpu(self):
        try:
            res = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], capture_output=True, text=True, timeout=3)
            if res.returncode == 0 and res.stdout.strip():
                self.gpu_available = True
                self._log(f"✅ GPU detected: {res.stdout.strip().splitlines()[0]}")
            else: self._log("ℹ️ No GPU detected, using CPU")
        except Exception:
            self._log("ℹ️ No GPU detected, using CPU")

    def preferred_devices(self):
        return ["cuda", "cpu"] if self.gpu_available else ["cpu"]


class UsageRow:
    def __init__(self, parent):
        self.usage_frame = CTkFrame(parent, height=40, corner_radius=5)
        self.usage_frame.pack(fill="x", pady=(0, 10))
        self.usage_label = CTkLabel(self.usage_frame, text="Loading system usage...", font=("Ubuntu Mono", 20), text_color="Lime")
        self.usage_label.pack(expand=True, fill="x", padx=10, pady=5)
        self._update_usage()

    def _update_usage(self):
        try:
            cpu, dram = psutil.cpu_percent(interval=0.5), psutil.virtual_memory().percent
            gpu_percent = vram_percent = 0
            try:
                import GPUtil
                gpus = GPUtil.getGPUs()
                if gpus: gpu_percent, vram_percent = gpus[0].load * 100, gpus[0].memoryUtil * 100
            except Exception: pass
            self.usage_label.configure(text=f"CPU: {cpu:.1f}%  DRAM: {dram:.1f}%  GPU: {gpu_percent:.1f}%  VRAM: {vram_percent:.1f}%")
        except Exception: self.usage_label.configure(text="⚠️ System monitoring unavailable")
        self.usage_label.after(2000, self._update_usage)


class ConsoleLogger:
    def __init__(self, gui_logger, stream):
        self.gui_logger, self.stream = gui_logger, stream
    def write(self, msg):
        if msg and msg.strip():
            self.stream.write(msg); self.stream.flush()
            if self.gui_logger: self.gui_logger.log(msg.strip(), "console")
    def flush(self): self.stream.flush()


class MemoryOptimizer:
    @staticmethod
    def get_system_memory_available_gb(): return psutil.virtual_memory().available / 1e9
    @staticmethod
    def clear_cache(): gc.collect()


class FeatureEngineer:
    @staticmethod
    def add_time_features(df):
        if 'time' not in df.columns: return df
        if not pd.api.types.is_datetime64_any_dtype(df['time']): df['time'] = pd.to_datetime(df['time'])
        df['hour'] = df['time'].dt.hour; df['day_of_week'] = df['time'].dt.dayofweek; df['month'] = df['time'].dt.month
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24); df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7); df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12); df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        return df

    @staticmethod
    def _add_atr(df, p=14):
        pc = df['close'].shift(1)
        df['atr'] = pd.concat([df['high'] - df['low'], (df['high'] - pc).abs(), (df['low'] - pc).abs()], axis=1).max(axis=1).ewm(alpha=1/p, min_periods=p, adjust=False).mean()
        return df

    @staticmethod
    def _add_bollinger(df, p=20, s=2):
        mid = df['close'].rolling(p).mean(); std = df['close'].rolling(p).std()
        br = (mid + s*std - (mid - s*std)).replace(0, np.nan)
        df['bb_percent_b'] = (df['close'] - (mid - s*std)) / br; df['bb_bandwidth'] = br / mid
        return df

    @staticmethod
    def _add_cci(df, p=20):
        tp = (df['high'] + df['low'] + df['close']) / 3
        md = tp.rolling(p).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True)
        df['cci'] = (tp - tp.rolling(p).mean()) / (0.015 * md.replace(0, np.nan))
        return df

    @staticmethod
    def _add_cmf(df, p=20):
        mfv = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low']).replace(0, np.nan) * df['volume']
        df['cmf'] = mfv.rolling(p).sum() / df['volume'].rolling(p).sum().replace(0, np.nan)
        return df

    @staticmethod
    def _add_donchian(df, p=20):
        h, l = df['high'].rolling(p).max(), df['low'].rolling(p).min()
        r = (h - l).replace(0, np.nan)
        df['donchian_position'] = (df['close'] - l) / r; df['donchian_width'] = r / df['close']
        return df

    @staticmethod
    def _add_ema(df):
        for n in [9, 21, 50, 200]: df[f'close_vs_ema_{n}'] = df['close'] / df['close'].ewm(span=n, adjust=False).mean() - 1
        df['ema_9_21_spread'] = df['close_vs_ema_9'] - df['close_vs_ema_21']
        df['ema_21_50_spread'] = df['close_vs_ema_21'] - df['close_vs_ema_50']
        df['ema_21_slope_5'] = df['close'].ewm(span=21, adjust=False).mean().diff(5) / df['close']
        return df

    @staticmethod
    def _add_force_index(df, p=13):
        df['force_index'] = (df['close'].diff() * df['volume']).ewm(span=p, adjust=False).mean()
        return df

    @staticmethod
    def _add_macd(df, f=12, s=26, sig=9):
        ml = df['close'].ewm(span=f, adjust=False).mean() - df['close'].ewm(span=s, adjust=False).mean()
        df['macd'] = ml; df['macd_signal'] = ml.ewm(span=sig, adjust=False).mean(); df['macd_hist'] = df['macd'] - df['macd_signal']
        return df

    @staticmethod
    def _add_mfi(df, p=14):
        tp = (df['high'] + df['low'] + df['close']) / 3; rmf = tp * df['volume']
        pos = rmf.where(tp > tp.shift(1), 0.0).rolling(p).sum()
        neg = rmf.where(tp < tp.shift(1), 0.0).rolling(p).sum()
        df['mfi'] = 100 - (100 / (1 + pos / neg.replace(0, np.nan)))
        return df

    @staticmethod
    def _add_obv(df):
        df['obv'] = (np.sign(df['close'].diff()).fillna(0) * df['volume']).cumsum()
        df['obv_slope_5'] = df['obv'].diff(5) / df['volume'].rolling(20).sum().replace(0, np.nan)
        return df

    @staticmethod
    def _add_price_structure(df):
        r = (df['high'] - df['low']).replace(0, np.nan)
        df['hl_range_pct'] = r / df['close']; df['body_pct'] = (df['close'] - df['open']).abs() / df['close']
        df['body_range_ratio'] = (df['close'] - df['open']).abs() / r; df['close_location_value'] = (df['close'] - df['low']) / r
        mx, mn = df[['open', 'close']].max(axis=1), df[['open', 'close']].min(axis=1)
        df['upper_shadow_pct'] = (df['high'] - mx) / df['close']; df['lower_shadow_pct'] = (mn - df['low']) / df['close']
        return df

    @staticmethod
    def _add_realized_vol(df):
        r = np.log(df['close'] / df['close'].shift(1))
        df['realized_vol_20'] = r.rolling(20).std(); df['realized_vol_100'] = r.rolling(100).std()
        return df

    @staticmethod
    def _add_roc(df):
        for n in [5, 10, 20]: df[f'roc_{n}'] = df['close'].pct_change(n) * 100
        return df

    @staticmethod
    def _add_rsi(df, p=14):
        d = df['close'].diff(); g = d.clip(lower=0); l = -d.clip(upper=0)
        df['rsi'] = 100 - (100 / (1 + g.ewm(alpha=1/p, min_periods=p, adjust=False).mean() / l.ewm(alpha=1/p, min_periods=p, adjust=False).mean().replace(0, np.nan)))
        return df

    @staticmethod
    def _add_rvwap(df, p=20):
        tp = (df['high'] + df['low'] + df['close']) / 3
        df['rvwap'] = (tp * df['volume']).rolling(p).sum() / df['volume'].rolling(p).sum().replace(0, np.nan)
        df['close_vs_rvwap'] = df['close'] / df['rvwap'] - 1
        return df

    @staticmethod
    def _add_stochastic(df, p=14, s=3):
        l, h = df['low'].rolling(p).min(), df['high'].rolling(p).max()
        df['stoch_k'] = 100 * (df['close'] - l) / (h - l).replace(0, np.nan); df['stoch_d'] = df['stoch_k'].rolling(s).mean()
        return df

    @staticmethod
    def _add_volume_metrics(df):
        m, s = df['volume'].rolling(20).mean(), df['volume'].rolling(20).std()
        df['volume_ratio_20'] = df['volume'] / m.replace(0, np.nan); df['volume_zscore_20'] = (df['volume'] - m) / s.replace(0, np.nan)
        df['return_x_volume_1'] = df['close'].pct_change() * df['volume']
        df['amihud_illiquidity_20'] = (df['close'].pct_change().abs() / (df['close'] * df['volume']).replace(0, np.nan)).rolling(20).mean()
        return df


class GUILogger:
    def __init__(self):
        self.messages, self.log_widget, self.ready = [], None, False
        self._out, self._err = sys.stdout, sys.stderr

    def set_log_widget(self, w):
        self.log_widget, self.ready = w, True; w.configure(font=("Ubuntu Mono", 11))
        for m, l in self.messages: self._write(m, l)
        self.messages.clear()

    def _write(self, msg, lvl):
        if self.log_widget:
            try: self.log_widget.insert("end", f"[{datetime.now().strftime('%H:%M:%S')}] {msg}\n"); self.log_widget.see("end")
            except Exception: print(msg, flush=True)

    def log(self, msg, lvl="info"):
        c = f"[{datetime.now().strftime('%H:%M:%S')}] [{lvl.upper()}] {msg}"
        try: self._out.write(c + "\n"); self._out.flush()
        except Exception: print(c, flush=True)
        if self.ready and self.log_widget: self._write(msg, lvl)
        else: self.messages.append((msg, lvl))

    def enable_console_redirect(self): sys.stdout, sys.stderr = ConsoleLogger(self, self._out), ConsoleLogger(self, self._err)
    def disable_console_redirect(self): sys.stdout, sys.stderr = self._out, self._err


class StreamingDataPreparer:
    def __init__(self, log_callback=None):
        self.log_callback = log_callback or print
        self.feature_cols, self.target_column = [], "target"
        self.enable_atr = self.enable_bollinger = self.enable_cci = self.enable_cmf = True
        self.enable_donchian = self.enable_ema = self.enable_force_index = self.enable_macd = True
        self.enable_mfi = self.enable_obv = self.enable_price_structure = self.enable_realized_vol = True
        self.enable_roc = self.enable_rsi = self.enable_rvwap = self.enable_stochastic = self.enable_volume_metrics = True

    def _log(self, m):
        if self.log_callback: self.log_callback(m)

    def set_indicator_flags(self, **kwargs):
        for k, v in kwargs.items():
            if hasattr(self, k): setattr(self, k, v)

    def load_csv_data(self, symbol, tf, folder):
        f = Path(folder) / symbol / f"{symbol}_{tf}.csv"
        if not f.exists(): return None
        try:
            df = pd.read_csv(f)
            if "time" not in df.columns: return None
            df["time"] = pd.to_datetime(df["time"], unit="s") if pd.api.types.is_numeric_dtype(df["time"]) else pd.to_datetime(df["time"])
            return df.sort_values("time").drop_duplicates("time").reset_index(drop=True)
        except Exception as e: self._log(f"❌ Error loading {f}: {e}"); return None

    def scan_folder(self, folder):
        folder = Path(folder)
        if not folder.exists(): return [], []
        syms, tfs = [], set()
        for d in sorted(folder.iterdir()):
            if not d.is_dir(): continue
            s = d.name
            found = [t for t in Config.TIMEFRAMES if (d / f"{s}_{t}.csv").exists()]
            if found: syms.append((s, found)); tfs.update(found); self._log(f"   ✓ {s}: {', '.join(found)}")
        return syms, sorted(tfs)

    def _prepare_single_series(self, df, sym, horizon, req_feats):
        if df is None or len(df) < 100: return None
        df = df.iloc[-Config.MAX_ROWS_PER_SYMBOL:].copy() if len(df) > Config.MAX_ROWS_PER_SYMBOL else df.copy()
        df = FeatureEngineer.add_time_features(df)
        eng = ["hour_sin", "hour_cos", "day_sin", "day_cos", "month_sin", "month_cos", "returns_1", "returns_5", "returns_10"]
        df['returns_1'], df['returns_5'], df['returns_10'] = df['close'].pct_change(), df['close'].pct_change(5), df['close'].pct_change(10)

        if self.enable_atr: df = FeatureEngineer._add_atr(df); eng.append("atr")
        if self.enable_bollinger: df = FeatureEngineer._add_bollinger(df); eng.extend(["bb_percent_b", "bb_bandwidth"])
        if self.enable_cci: df = FeatureEngineer._add_cci(df); eng.append("cci")
        if self.enable_cmf: df = FeatureEngineer._add_cmf(df); eng.append("cmf")
        if self.enable_donchian: df = FeatureEngineer._add_donchian(df); eng.extend(["donchian_position", "donchian_width"])
        if self.enable_ema: df = FeatureEngineer._add_ema(df); eng.extend(["close_vs_ema_9", "close_vs_ema_21", "close_vs_ema_50", "close_vs_ema_200", "ema_9_21_spread", "ema_21_50_spread", "ema_21_slope_5"])
        if self.enable_force_index: df = FeatureEngineer._add_force_index(df); eng.append("force_index")
        if self.enable_macd: df = FeatureEngineer._add_macd(df); eng.extend(["macd", "macd_signal", "macd_hist"])
        if self.enable_mfi: df = FeatureEngineer._add_mfi(df); eng.append("mfi")
        if self.enable_obv: df = FeatureEngineer._add_obv(df); eng.append("obv_slope_5")
        if self.enable_price_structure: df = FeatureEngineer._add_price_structure(df); eng.extend(["hl_range_pct", "body_pct", "body_range_ratio", "close_location_value", "upper_shadow_pct", "lower_shadow_pct"])
        if self.enable_realized_vol: df = FeatureEngineer._add_realized_vol(df); eng.extend(["realized_vol_20", "realized_vol_100"])
        if self.enable_roc: df = FeatureEngineer._add_roc(df); eng.extend(["roc_5", "roc_10", "roc_20"])
        if self.enable_rsi: df = FeatureEngineer._add_rsi(df); eng.append("rsi")
        if self.enable_rvwap: df = FeatureEngineer._add_rvwap(df); eng.append("close_vs_rvwap")
        if self.enable_stochastic: df = FeatureEngineer._add_stochastic(df); eng.extend(["stoch_k", "stoch_d"])
        if self.enable_volume_metrics: df = FeatureEngineer._add_volume_metrics(df); eng.extend(["volume_ratio_20", "volume_zscore_20", "return_x_volume_1", "amihud_illiquidity_20"])

        req_feats = [c for c in req_feats if c != "time"]
        avail = [c for c in list(dict.fromkeys(req_feats + eng)) if c in df.columns]
        for c in ["open", "high", "low", "close", "volume"]:
            if c in df.columns and c not in avail: avail.append(c)
        if "close" not in df.columns or not avail: return None

        df[avail] = df[avail].replace([np.inf, -np.inf], np.nan).ffill().bfill().fillna(0.0)
        df[self.target_column] = df["close"].shift(-horizon)
        df = df.dropna(subset=[self.target_column]).copy()
        if len(df) < 50: return None

        df["group_id"], df["time_idx"] = str(sym), np.arange(len(df), dtype=np.int64)
        return df[["time", "time_idx", "group_id", self.target_column] + avail], avail

    def build_dataframe_for_timeframe(self, folder, feats, horizon, tf):
        syms, _ = self.scan_folder(folder)
        syms_tf = [s for s, t in syms if tf in t]
        if not syms_tf: raise ValueError(f"No symbols for {tf}")

        frames, all_f = [], []
        for s in syms_tf:
            p = self._prepare_single_series(self.load_csv_data(s, tf, folder), s, horizon, feats)
            if p: frames.append(p[0]); all_f.extend(p[1]); self._log(f"   📊 {s}_{tf}: {len(p[0])} rows")
        if not frames: raise ValueError(f"No data for {tf}")

        all_f = list(dict.fromkeys(all_f))
        res = pd.concat(frames, ignore_index=True, sort=False)
        for c in all_f:
            if c not in res.columns: res[c] = 0.0
        res[all_f] = res[all_f].replace([np.inf, -np.inf], np.nan).ffill().bfill().fillna(0.0).astype(np.float32)
        res[self.target_column] = res[self.target_column].replace([np.inf, -np.inf], np.nan).ffill().bfill().astype(np.float32)
        res["group_id"] = res["group_id"].astype(str)
        res = res.sort_values(["group_id", "time_idx"]).reset_index(drop=True)

        self.feature_cols = all_f
        self._log(f"✅ [{tf}] Prepared {len(res):,} rows, {res['group_id'].nunique()} symbols, {len(all_f)} features")
        return res


class StreamingXGBoostTrainer:
    def __init__(self, log_callback=None, timeframe=None):
        self.log_callback, self.timeframe = log_callback or print, timeframe
        self.gpu_manager = GPUManager(verbose=False)
        self.model, self.is_trained, self.stop_flag, self.device_used = None, False, False, "cpu"
        self.iterations, self.learning_rate, self.max_depth = Config.DEFAULT_ITERATIONS, Config.DEFAULT_LEARNING_RATE, Config.DEFAULT_MAX_DEPTH
        self.reg_lambda, self.subsample, self.colsample = Config.DEFAULT_L2_REG, Config.DEFAULT_SUBSAMPLE, Config.DEFAULT_COLSAMPLE
        self.training_history = {}
        self._log(f"🚀 XGBoost [{timeframe}] initialized. Devices: {' -> '.join(self.gpu_manager.preferred_devices()).upper()}")

    def _log(self, m):
        if self.log_callback: self.log_callback(m)

    def stop_training(self): self.stop_flag = True; self._log("⏹️ Stop requested")

    def _chronological_split(self, data):
        tr, vl = [], []
        for _, g in data.groupby("group_id"):
            g = g.sort_values("time_idx"); n = max(1, int(len(g) * Config.VALIDATION_FRACTION))
            tr.append(g.iloc[:-n]); vl.append(g.iloc[-n:])
        return pd.concat(tr, ignore_index=True), pd.concat(vl, ignore_index=True)

    def _compute_directional_accuracy(self, model, X_val, val_df):
        try:
            if "close" not in val_df.columns: return None
            preds, lc, at = model.predict(X_val), val_df["close"].to_numpy(), val_df["target"].to_numpy()
            ad, pd_dir = np.sign(at - lc), np.sign(preds - lc)
            vm = ad != 0
            return float(((ad == pd_dir) & vm).sum() / vm.sum()) if vm.sum() > 0 else None
        except Exception: return None

    def _build_model(self, device, it, lr, md):
        return xgb.XGBRegressor(n_estimators=it, learning_rate=lr, max_depth=md, reg_lambda=self.reg_lambda, subsample=self.subsample,
                                colsample_bytree=self.colsample, objective="reg:squarederror", eval_metric="rmse",
                                early_stopping_rounds=Config.EARLY_STOPPING_ROUNDS, tree_method="hist", device=device,
                                enable_categorical=True, random_state=42, n_jobs=-1, verbosity=0)

    def train_single_timeframe(self, data, feature_cols, timeframe, iterations=1000, learning_rate=0.05, max_depth=6, **kwargs):
        try:
            self.stop_flag, self.timeframe = False, timeframe
            if len(data) == 0: raise ValueError("Empty data")
            tr, vl = self._chronological_split(data)
            if len(tr) == 0 or len(vl) == 0: raise ValueError("Empty split")

            x_cols = feature_cols + ["group_id"]
            X_tr, y_tr, X_vl, y_vl = tr[x_cols].copy(), tr["target"], vl[x_cols].copy(), vl["target"]
            X_tr["group_id"], X_vl["group_id"] = X_tr["group_id"].astype("category"), X_vl["group_id"].astype("category")

            self._log(f"📊 [{timeframe}] Train: {len(tr):,} | Val: {len(vl):,} | Depth: {max_depth}, Iter: {iterations}")
            last_err, fitted, used_dev = None, None, None

            for dev in self.gpu_manager.preferred_devices():
                if self.stop_flag: break
                self._log(f"🚀 [{timeframe}] Trying device='{dev}'...")
                try:
                    fitted = self._build_model(dev, iterations, learning_rate, max_depth).fit(X_tr, y_tr, eval_set=[(X_vl, y_vl)], verbose=False)
                    used_dev = dev; break
                except Exception as e: last_err = e; self._log(f"⚠️ [{timeframe}] device='{dev}' failed ({e})")

            if not fitted: raise last_err or RuntimeError("No device succeeded")
            self.model, self.device_used, self.is_trained = fitted, used_dev, True
            self._log(f"✅ [{timeframe}] Trained on '{used_dev}'")

            da = self._compute_directional_accuracy(self.model, X_vl, vl)
            self.training_history = {"timeframe": timeframe, "device_used": used_dev, "best_iteration": getattr(self.model, 'best_iteration', iterations),
                                     "best_score": getattr(self.model, 'best_score', None), "directional_accuracy": da}
            if da: self._log(f"🎯 [{timeframe}] Directional accuracy: {da*100:.1f}%")
            return True
        except Exception as e: self._log(f"❌ [{timeframe}] Error: {e}"); traceback.print_exc(); return False

    def save_model(self, model_dir, stem, feature_cols, horizon, timeframe):
        if not self.is_trained: return False
        try:
            out = Path(model_dir); out.mkdir(parents=True, exist_ok=True)
            mp, pp = out / f"{stem}{Config.MODEL_EXT}", out / f"{stem}{Config.METADATA_EXT}"
            self.model.save_model(str(mp))
            with open(pp, "wb") as f:
                pickle.dump({"model_type": "xgboost.XGBRegressor", "timeframe": timeframe, "model_path": str(mp), "feature_columns": feature_cols,
                             "categorical_features": ["group_id"], "device_used": self.device_used, "horizon": horizon,
                             "directional_accuracy": self.training_history.get("directional_accuracy"), "training_history": self.training_history,
                             "training_time": datetime.now().isoformat()}, f)
            self._log(f"✅ Model saved: {mp}"); return True
        except Exception as e: self._log(f"❌ Save error: {e}"); return False


class XGBoostTrainerGUI(CTk):
    def __init__(self):
        super().__init__()
        self.title(Config.APP_TITLE); self.geometry(Config.APP_GEOMETRY); self.resizable(False, False)
        self.gui_logger = GUILogger(); self.gui_logger.enable_console_redirect()
        self.csv_folder, self.model_folder = Config.DATASET_DIR, Config.MODEL_DIR
        self.is_training, self.stop_flag, self.current_trainer = False, False, None
        self.data_preparer = StreamingDataPreparer(self.gui_logger.log)
        self._create_ui(); self.gui_logger.set_log_widget(self.log_text)
        self.after(500, self.scan_data); self.protocol("WM_DELETE_WINDOW", self._on_closing)

    def _create_ui(self):
        m = CTkFrame(self, fg_color="transparent"); m.pack(fill="both", expand=True, padx=10, pady=5)
        t = CTkFrame(m, corner_radius=6); t.pack(fill="x", pady=(0, 4))
        CTkLabel(t, text="🎯 XGBoost Model Trainer", font=("Ubuntu", 16, "bold")).pack(pady=4)
        CTkLabel(t, text="BSD-C-3 License", font=("Ubuntu", 18), text_color="yellow").pack(pady=(0, 4))
        UsageRow(m)

        self._create_csv(m); self._create_model(m); self._create_params(m)
        self._create_feats(m); self._create_tfs(m); self._create_inds(m)
        self._create_ctrls(m); self._create_stat(m); self._create_log(m)

    def _create_csv(self, p):
        f = CTkFrame(p, corner_radius=6); f.pack(fill="x", pady=(0, 4))
        ff = CTkFrame(f, fg_color="transparent"); ff.pack(fill="x", padx=8, pady=3)
        CTkButton(ff, text="📁 CSV Folder", command=self.select_csv, width=120, height=28).pack(side="left", padx=(0, 6))
        self.csv_var = tk.StringVar(value=str(self.csv_folder))
        CTkEntry(ff, textvariable=self.csv_var, state="readonly", height=28).pack(side="left", fill="x", expand=True)

    def _create_model(self, p):
        f = CTkFrame(p, corner_radius=6); f.pack(fill="x", pady=(0, 4))
        ff = CTkFrame(f, fg_color="transparent"); ff.pack(fill="x", padx=8, pady=3)
        CTkButton(ff, text="📁 Model Folder", command=self.select_model, width=120, height=28).pack(side="left", padx=(0, 6))
        self.model_var = tk.StringVar(value=str(self.model_folder))
        CTkEntry(ff, textvariable=self.model_var, state="readonly", height=28).pack(side="left", fill="x", expand=True)
        self.stat_lbl = CTkLabel(f, text="📊 Status: Ready", font=("Ubuntu", 10)); self.stat_lbl.pack(anchor="w", padx=8, pady=1)

    def _create_params(self, p):
        f = CTkFrame(p, corner_radius=6); f.pack(fill="x", pady=(0, 4))
        pf = CTkFrame(f, fg_color="transparent"); pf.pack(fill="x", padx=8, pady=3)
        self.horizon = tk.StringVar(value=str(Config.DEFAULT_HORIZON))
        self.iters = tk.StringVar(value=str(Config.DEFAULT_ITERATIONS))
        self.lr = tk.StringVar(value=str(Config.DEFAULT_LEARNING_RATE))
        self.depth = tk.StringVar(value=str(Config.DEFAULT_MAX_DEPTH))
        r = CTkFrame(pf, fg_color="transparent"); r.pack(fill="x", pady=1)
        for c in range(8): r.grid_columnconfigure(c, weight=1 if c%2 else 0)
        CTkLabel(r, text="Horizon:", width=70).grid(row=0, column=0, padx=(0, 3), sticky="w")
        CTkEntry(r, textvariable=self.horizon, width=60).grid(row=0, column=1, padx=(0, 10), sticky="w")
        CTkLabel(r, text="Iterations:", width=70).grid(row=0, column=2, padx=(0, 3), sticky="w")
        CTkEntry(r, textvariable=self.iters, width=60).grid(row=0, column=3, padx=(0, 10), sticky="w")
        CTkLabel(r, text="LR:", width=70).grid(row=0, column=4, padx=(0, 3), sticky="w")
        CTkEntry(r, textvariable=self.lr, width=60).grid(row=0, column=5, padx=(0, 10), sticky="w")
        CTkLabel(r, text="Depth:", width=70).grid(row=0, column=6, padx=(0, 3), sticky="w")
        CTkEntry(r, textvariable=self.depth, width=60).grid(row=0, column=7, sticky="w")

    def _create_feats(self, p):
        f = CTkFrame(p, corner_radius=6); f.pack(fill="x", pady=(0, 4))
        ff = CTkFrame(f, fg_color="transparent"); ff.pack(fill="x", padx=8, pady=3)
        self.feat_var = tk.StringVar(value="time,open,high,low,close,volume")
        CTkLabel(ff, text="Base Features:", font=("Ubuntu", 10)).pack(anchor="w", pady=(0, 2))
        CTkEntry(ff, textvariable=self.feat_var, height=28).pack(fill="x", pady=(0, 2))

    def _create_tfs(self, p):
        f = CTkFrame(p, corner_radius=6); f.pack(fill="x", pady=(0, 4))
        tf = CTkFrame(f, fg_color="transparent"); tf.pack(fill="x", padx=8, pady=3)
        CTkLabel(tf, text="Timeframes (one model each):", font=("Ubuntu", 10)).pack(anchor="w", pady=(0, 2))
        self.tf_vars = {}; cf = CTkFrame(tf, fg_color="transparent"); cf.pack(fill="x")
        for i, t in enumerate(Config.TIMEFRAMES):
            v = tk.BooleanVar(value=True); self.tf_vars[t] = v
            CTkCheckBox(cf, text=t, variable=v, width=70, font=("Ubuntu", 10)).grid(row=0, column=i, padx=3, sticky="w")

    def _create_inds(self, p):
        f = CTkFrame(p, corner_radius=6); f.pack(fill="x", pady=(0, 4))
        inf = CTkFrame(f, fg_color="transparent"); inf.pack(fill="x", padx=8, pady=3)
        CTkLabel(inf, text="Technical Indicators (inputs):", font=("Ubuntu", 10)).pack(anchor="w", pady=(0, 2))
        cf = CTkFrame(inf, fg_color="transparent"); cf.pack(fill="x")
        inds = [("ATR", "enable_atr"), ("Bollinger", "enable_bollinger"), ("CCI", "enable_cci"), ("CMF", "enable_cmf"),
                ("Donchian", "enable_donchian"), ("EMA", "enable_ema"), ("Force Index", "enable_force_index"), ("MACD", "enable_macd"),
                ("MFI", "enable_mfi"), ("OBV", "enable_obv"), ("Price Struct", "enable_price_structure"), ("Realized Vol", "enable_realized_vol"),
                ("ROC", "enable_roc"), ("RSI", "enable_rsi"), ("RVWAP", "enable_rvwap"), ("Stochastic", "enable_stochastic"), ("Volume", "enable_volume_metrics")]
        self.ind_vars = {}
        for i, (l, v) in enumerate(inds):
            r, c = i // 6, i % 6
            var = tk.BooleanVar(value=True); self.ind_vars[v] = var
            CTkCheckBox(cf, text=l, variable=var, width=95, font=("Ubuntu", 10)).grid(row=r, column=c, padx=4, pady=2, sticky="w")
        CTkLabel(inf, text="Uncheck indicators you don't want as model inputs", font=("Ubuntu", 8), text_color="gray").pack(anchor="w", pady=(4, 0))

    def _create_ctrls(self, p):
        f = CTkFrame(p, corner_radius=6); f.pack(fill="x", pady=(0, 4))
        bf = CTkFrame(f, fg_color="transparent"); bf.pack(pady=4)
        self.train_btn = CTkButton(bf, text="🚀 Start Training", command=self.start_train, width=130, height=34, fg_color="#2e7d32", hover_color="#1b5e20", font=("Ubuntu", 12, "bold"))
        self.train_btn.pack(side="left", padx=4)
        self.stop_btn = CTkButton(bf, text="⏹️ Stop Training", command=self.stop_train, width=130, height=34, fg_color="#c62828", hover_color="#b71c1c", font=("Ubuntu", 12, "bold"), state="disabled")
        self.stop_btn.pack(side="left", padx=4)

    def _create_stat(self, p):
        f = CTkFrame(p, corner_radius=6); f.pack(fill="x", pady=(0, 4))
        self.stat_txt = CTkTextbox(f, height=50, font=("Ubuntu Mono", 10)); self.stat_txt.pack(fill="x", padx=8, pady=3)
        self.stat_txt.insert("1.0", "Ready"); self.stat_txt.configure(state="disabled")

    def _create_log(self, p):
        f = CTkFrame(p, corner_radius=6); f.pack(fill="both", expand=True)
        self.log_text = CTkTextbox(f, font=("Ubuntu Mono", 10))
        self.log_text.pack(fill="both", expand=True, padx=8, pady=3)

    def update_stat(self, m):
        self.stat_txt.configure(state="normal"); self.stat_txt.delete("1.0", "end"); self.stat_txt.insert("1.0", m); self.stat_txt.configure(state="disabled")

    def select_csv(self):
        f = filedialog.askdirectory(initialdir=str(self.csv_folder))
        if f: self.csv_var.set(f); self.scan_data()

    def select_model(self):
        f = filedialog.askdirectory(initialdir=str(self.model_folder))
        if f:
            self.model_var.set(f)
            global MODEL_DIR; MODEL_DIR = Path(f); MODEL_DIR.mkdir(parents=True, exist_ok=True)

    def scan_data(self):
        cf = Path(self.csv_var.get())
        if not cf.exists(): self.stat_lbl.configure(text="❌ Folder not found"); self.train_btn.configure(state="disabled"); return
        s, t = self.data_preparer.scan_folder(cf)
        if not s: self.stat_lbl.configure(text="⚠️ No symbols found"); self.train_btn.configure(state="disabled"); return
        self.stat_lbl.configure(text=f"✅ Found {len(s)} symbols, {len(t)} timeframes"); self.train_btn.configure(state="normal")

    def start_train(self):
        if self.is_training: return
        cf = Path(self.csv_var.get())
        stf = [t for t, v in self.tf_vars.items() if v.get()]
        if not stf: return
        self.is_training, self.stop_flag = True, False
        self.train_btn.configure(state="disabled"); self.stop_btn.configure(state="normal")

        def task():
            try:
                self.gui_logger.log("=" * 60); self.gui_logger.log("🚀 Starting Per-Timeframe XGBoost Training")
                self.data_preparer.set_indicator_flags(**{k: v.get() for k, v in self.ind_vars.items()})
                self.gui_logger.log(f"📊 Indicators: {', '.join(k.replace('enable_', '').upper() for k, v in self.ind_vars.items() if v.get())}")

                fi = self.feat_var.get().strip()
                fc = [f.strip() for f in fi.split(',') if f.strip()] if fi else Config.DEFAULT_FEATURES
                h, it, lr, md = int(self.horizon.get()), int(self.iters.get()), float(self.lr.get()), int(self.depth.get())
                mf = self.model_var.get(); res = {}

                for tf in stf:
                    if self.stop_flag: break
                    self.update_stat(f"[{tf}] Building dataset...")
                    try: data = self.data_preparer.build_dataframe_for_timeframe(cf, fc, h, tf)
                    except ValueError: res[tf] = "skipped"; continue

                    self.update_stat(f"[{tf}] Training XGBoost...")
                    tr = StreamingXGBoostTrainer(self.gui_logger.log, timeframe=tf); self.current_trainer = tr
                    ok = tr.train_single_timeframe(data, self.data_preparer.feature_cols, tf, it, lr, md)
                    if ok and not tr.stop_flag:
                        stem = f"{Config.MODEL_PREFIX}_{tf}"
                        if tr.save_model(mf, stem, self.data_preparer.feature_cols, h, tf): res[tf] = f"trained ({tr.device_used})"
                        else: res[tf] = "save failed"
                    else: res[tf] = "stopped" if tr.stop_flag else "failed"
                    del tr; self.current_trainer = None; MemoryOptimizer.clear_cache()

                self.gui_logger.log(f"🏁 Summary: {', '.join(f'{k}: {v}' for k, v in res.items())}"); self.update_stat("Done.")
            except Exception as e: self.gui_logger.log(f"❌ Error: {e}", "error"); traceback.print_exc()
            finally: self.is_training, self.current_trainer = False, None; self.train_btn.configure(state="normal"); self.stop_btn.configure(state="disabled")

        threading.Thread(target=task, daemon=True).start()

    def stop_train(self):
        self.stop_flag = True
        if self.current_trainer: self.current_trainer.stop_training()

    def _on_closing(self):
        self.gui_logger.disable_console_redirect()
        if self.is_training:
            if messagebox.askyesno("Training", "Stop and quit?"):
                self.stop_flag = True
                if self.current_trainer: self.current_trainer.stop_training()
                self.after(100, self.destroy)
        else:
            if messagebox.askyesno("Quit", "Quit?"): self.destroy()

def main():
    XGBoostTrainerGUI().mainloop()

if __name__ == "__main__":
    main()