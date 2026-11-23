#!/usr/bin/env python3
# ai_predict_train_dl.py
# Copyright Su Nie | BSD-3C License | https://github.com/can87683

"""
customtkinter gputil numpy onnx onnxruntime pandas psutil setproctitle torch
"""

import os
import sys
import setproctitle
setproctitle.setproctitle(os.path.basename(sys.argv[0]))

import configparser
import customtkinter as ctk
import datetime
import gc
import GPUtil
import json
import numpy as np
import onnx
import onnxruntime
import pandas as pd
import pathlib
import psutil
import queue
import re
import threading
import tkinter as tk
import tkinter.filedialog as filedialog
import tkinter.messagebox as messagebox
import torch
import torch.nn as nn
import torch.nn.functional as functional
import traceback
import warnings


class Config:
    HOME = pathlib.Path.home()
    MODEL_DIR = HOME / "ai_model_ml"
    DATASET_DIR = HOME / "mt5_data"

    REQUIRED_COLUMNS = ['time', 'open', 'high', 'low', 'close', 'tick_volume']
    RAW_CHANNELS = ['open', 'high', 'low', 'close', 'tick_volume']
    SYMBOL_PATTERN = r"(?!\.)[A-Za-z0-9._-]+"
    MODEL_PREFIX = "ai_predict"
    MAX_ROWS_PER_SYMBOL = 20000
    WINDOW = 128
    MIN_SAMPLES = 1000
    RANDOM_STATE = 42
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    DEFAULT_HORIZON = 10
    DEFAULT_EPOCHS = 40
    DEFAULT_LEARNING_RATE = 0.001
    DEFAULT_BATCH_SIZE = 256
    DEFAULT_HIDDEN = 64
    DEFAULT_DROPOUT = 0.1
    DEFAULT_WEIGHT_DECAY = 0.0001

    MIN_SAMPLES_BY_TIMEFRAME = {
        "M5": 1000,
        "M15": 1000,
        "H1": 1000,
        "H4": 1000,
        "D1": 700,
        "W1": 490,
    }
    TIMEFRAME_HYPERPARAMS = {
        "M5":  {"epochs": 30, "hidden": 64, "dropout": 0.10, "weight_decay": 0.0001},
        "M15": {"epochs": 30, "hidden": 64, "dropout": 0.10, "weight_decay": 0.0001},
        "H1":  {"epochs": 40, "hidden": 64, "dropout": 0.15, "weight_decay": 0.0005},
        "H4":  {"epochs": 50, "hidden": 48, "dropout": 0.20, "weight_decay": 0.001},
        "D1":  {"epochs": 60, "hidden": 32, "dropout": 0.25, "weight_decay": 0.005},
        "W1":  {"epochs": 80, "hidden": 32, "dropout": 0.30, "weight_decay": 0.01},
    }
    TIMEFRAMES = ['M5', 'M15', 'H1', 'H4', 'D1', 'W1']

    APP_TITLE = "DL/NN ML Trainer"
    THEME = "dark"
    COLOR_THEME = "dark-blue"

    @classmethod
    def ensure_directories(cls):
        cls.MODEL_DIR.mkdir(parents=True, exist_ok=True)
        cls.DATASET_DIR.mkdir(parents=True, exist_ok=True)


class UsageRow:
    FRAME_HEIGHT = 40
    FRAME_CORNER_RADIUS = 2
    FRAME_BG = "#1c1c1c"
    FRAME_PADY = 2

    FONT = ("TkDefaultFont", 20)
    TEXT_COLOR = "Lime"
    PLACEHOLDER_TEXT = "Loading system usage..."

    LABEL_PADX = 2
    LABEL_PADY = 2

    REFRESH_MS = 2000

    def __init__(self, parent):
        self.usage_frame = ctk.CTkFrame(
            parent,
            height=self.FRAME_HEIGHT,
            corner_radius=self.FRAME_CORNER_RADIUS,
            fg_color=self.FRAME_BG,
        )
        self.usage_frame.pack(fill="x", pady=self.FRAME_PADY)
        self.usage_frame.pack_propagate(False)

        self.usage_label = ctk.CTkLabel(
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
        cpu = psutil.cpu_percent(interval=None)
        dram = psutil.virtual_memory().percent
        gpu_percent = vram_percent = 0
        gpus = GPUtil.getGPUs()
        if gpus:
            gpu_percent = gpus[0].load * 100
            vram_percent = gpus[0].memoryUtil * 100
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


class GUILogger:
    MAX_LOG_LINES = 2000
    ERROR_COLOR = "yellow"
    POLL_MS = 50

    def __init__(self):
        self.queue = queue.Queue()
        self.log_widget = None
        self._original_stdout = sys.stdout
        self._original_stderr = sys.stderr

    def set_log_widget(self, widget):
        self.log_widget = widget
        self.log_widget.tag_config("error", foreground=self.ERROR_COLOR)
        self._drain()

    def _drain(self):
        while not self.queue.empty():
            entry, tag = self.queue.get_nowait()
            self._insert(entry, tag)
        self.log_widget.after(self.POLL_MS, self._drain)

    def _insert(self, entry, tag):
        self.log_widget.insert("end", entry, tag)
        lines = int(self.log_widget.index("end-1c").split(".")[0])
        if lines > self.MAX_LOG_LINES:
            self.log_widget.delete("1.0", f"{lines - self.MAX_LOG_LINES + 1}.0")
        self.log_widget.see("end")

    def log(self, message, level="info"):
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        self._original_stdout.write(f"[{timestamp}] [{level.upper()}] {message}\n")
        self._original_stdout.flush()
        tag = "error" if level == "error" else None
        self.queue.put((f"[{timestamp}] {message}\n", tag))

    def enable_console_redirect(self):
        sys.stdout = ConsoleLogger(self, self._original_stdout, "console")
        sys.stderr = ConsoleLogger(self, self._original_stderr, "error")

    def close(self):
        sys.stdout = self._original_stdout
        sys.stderr = self._original_stderr


class MemoryOptimizer:
    @staticmethod
    def get_system_memory_available_gb():
        return psutil.virtual_memory().available / 1e9

    @staticmethod
    def clear_cache():
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class CNNBackbone(nn.Module):
    def __init__(self, n_channels, window, hidden, dropout):
        super().__init__()
        self.out_dim = hidden * 3 + n_channels
        self.net = nn.Sequential(
            nn.Conv1d(n_channels, hidden, 5, padding=2), nn.BatchNorm1d(hidden), nn.ReLU(),
            nn.Conv1d(hidden, hidden, 5, stride=2, padding=2), nn.BatchNorm1d(hidden), nn.ReLU(),
            nn.Conv1d(hidden, hidden, 5, stride=2, padding=2), nn.BatchNorm1d(hidden), nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        h = self.net(x.transpose(1, 2))
        return torch.cat([h.mean(dim=2), h.amax(dim=2), h[:, :, -1], x[:, -1]], dim=1)


class GRUBackbone(nn.Module):
    LAYERS = 2

    def __init__(self, n_channels, window, hidden, dropout):
        super().__init__()
        self.out_dim = hidden
        self.rnn = nn.GRU(n_channels, hidden, self.LAYERS, batch_first=True, dropout=dropout)

    def forward(self, x):
        return self.rnn(x)[0][:, -1]


class LSTMBackbone(nn.Module):
    LAYERS = 2

    def __init__(self, n_channels, window, hidden, dropout):
        super().__init__()
        self.out_dim = hidden
        self.rnn = nn.LSTM(n_channels, hidden, self.LAYERS, batch_first=True, dropout=dropout)

    def forward(self, x):
        return self.rnn(x)[0][:, -1]


class PatchTSTBackbone(nn.Module):
    PATCH_LEN = 16
    STRIDE = 16
    LAYERS = 2
    D_MODEL_DIV = 2

    def __init__(self, n_channels, window, hidden, dropout):
        super().__init__()
        d_model = hidden // self.D_MODEL_DIV
        self.out_dim = hidden
        self.starts = list(range(0, window - self.PATCH_LEN + 1, self.STRIDE))
        self.embed = nn.Linear(self.PATCH_LEN, d_model)
        self.pos = nn.Parameter(torch.zeros(1, len(self.starts), d_model))
        self.blocks = nn.Sequential(*[AttentionBlock(d_model, len(self.starts), dropout) for _ in range(self.LAYERS)])
        self.norm = nn.LayerNorm(d_model)
        self.flat = nn.Sequential(nn.Dropout(dropout), nn.Linear(n_channels * len(self.starts) * d_model, hidden))

    def forward(self, x):
        xt = x.transpose(1, 2)
        patches = torch.stack([xt[:, :, s:s + self.PATCH_LEN] for s in self.starts], dim=2)
        z = self.norm(self.blocks(self.embed(patches.reshape(-1, len(self.starts), self.PATCH_LEN)) + self.pos))
        return self.flat(z.reshape(x.shape[0], -1))


class TCNBlock(nn.Module):
    def __init__(self, c_in, c_out, kernel, dilation, dropout):
        super().__init__()
        self.pad = (kernel - 1) * dilation
        self.conv1 = nn.Conv1d(c_in, c_out, kernel, dilation=dilation)
        self.conv2 = nn.Conv1d(c_out, c_out, kernel, dilation=dilation)
        self.drop = nn.Dropout(dropout)
        self.skip = nn.Conv1d(c_in, c_out, 1) if c_in != c_out else nn.Identity()

    def forward(self, x):
        h = self.drop(torch.relu(self.conv1(functional.pad(x, (self.pad, 0)))))
        h = self.drop(torch.relu(self.conv2(functional.pad(h, (self.pad, 0)))))
        return torch.relu(h + self.skip(x))


class TCNBackbone(nn.Module):
    KERNEL = 3
    DILATIONS = (1, 2, 4, 8, 16)

    def __init__(self, n_channels, window, hidden, dropout):
        super().__init__()
        self.out_dim = hidden
        channels = [n_channels] + [hidden] * len(self.DILATIONS)
        self.blocks = nn.Sequential(*[
            TCNBlock(channels[i], channels[i + 1], self.KERNEL, d, dropout) for i, d in enumerate(self.DILATIONS)
        ])

    def forward(self, x):
        return self.blocks(x.transpose(1, 2))[:, :, -1]


class AttentionBlock(nn.Module):
    HEADS = 4

    def __init__(self, d_model, tokens, dropout):
        super().__init__()
        self.tokens = tokens
        self.d_head = d_model // self.HEADS
        self.scale = 1.0 / (self.d_head ** 0.5)
        self.norm1 = nn.LayerNorm(d_model)
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(nn.Linear(d_model, 2 * d_model), nn.GELU(), nn.Dropout(dropout), nn.Linear(2 * d_model, d_model))
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        q, k, v = self.qkv(self.norm1(x)).reshape(-1, self.tokens, 3, self.HEADS, self.d_head).permute(2, 0, 3, 1, 4).unbind(0)
        att = torch.softmax(q @ k.transpose(-2, -1) * self.scale, dim=-1)
        x = x + self.drop(self.proj((att @ v).transpose(1, 2).reshape(-1, self.tokens, self.HEADS * self.d_head)))
        return x + self.drop(self.mlp(self.norm2(x)))


class SequenceModel(nn.Module):
    ARCHITECTURES = {
        "CNN": CNNBackbone,
        "GRU": GRUBackbone,
        "LSTM": LSTMBackbone,
        "PatchTST": PatchTSTBackbone,
        "TCN": TCNBackbone,
    }
    INPUT_CLIP = 10.0

    def __init__(self, arch, hidden, dropout):
        super().__init__()
        n = len(Config.RAW_CHANNELS)
        self.register_buffer("in_mean", torch.zeros(n))
        self.register_buffer("in_std", torch.ones(n))
        self.register_buffer("y_mean", torch.zeros(1))
        self.register_buffer("y_std", torch.ones(1))
        self.backbone = self.ARCHITECTURES[arch](n, Config.WINDOW, hidden, dropout)
        self.head = nn.Sequential(nn.Linear(self.backbone.out_dim, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, 1))

    def standardized(self, raw):
        z = torch.clamp((raw - self.in_mean) / self.in_std, -self.INPUT_CLIP, self.INPUT_CLIP)
        return self.head(self.backbone(z))

    def forward(self, raw):
        return self.standardized(raw) * self.y_std + self.y_mean


class SignalExport(nn.Module):
    def __init__(self, model, threshold):
        super().__init__()
        self.model = model
        self.register_buffer("threshold", torch.tensor([threshold], dtype=torch.float32))

    def forward(self, raw):
        prediction = self.model(raw)
        signal = torch.where(prediction.abs() > self.threshold, torch.sign(prediction), torch.zeros_like(prediction))
        return prediction, signal


class StreamingDataPreparer:
    TIMEFRAME_SECONDS = {
        "M5": 300,
        "M15": 900,
        "H1": 3600,
        "H4": 14400,
        "D1": 86400,
        "W1": 604800,
    }
    PRICE_COLUMNS = ("open", "high", "low", "close")
    NUMERIC_COLUMNS = ("open", "high", "low", "close", "tick_volume")

    def __init__(self, log_callback):
        self.log_callback = log_callback

    def _log(self, message):
        self.log_callback(message)

    def load_csv_data(self, symbol, timeframe, csv_folder):
        if not re.fullmatch(Config.SYMBOL_PATTERN, symbol):
            raise ValueError(f"Unsafe symbol name: {symbol!r}")

        if timeframe not in self.TIMEFRAME_SECONDS:
            raise ValueError(f"Unsupported timeframe: {timeframe}")

        csv_path = pathlib.Path(csv_folder) / symbol / f"{symbol}_{timeframe}.csv"
        df = pd.read_csv(csv_path)

        missing = [column for column in Config.REQUIRED_COLUMNS if column not in df.columns]
        if missing:
            raise ValueError(f"{csv_path.name} is missing required columns: {missing}")

        if pd.api.types.is_numeric_dtype(df["time"]):
            df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
        else:
            df["time"] = pd.to_datetime(df["time"], utc=True)

        for column in self.NUMERIC_COLUMNS:
            df[column] = pd.to_numeric(df[column], errors="coerce")

        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.dropna(subset=Config.REQUIRED_COLUMNS)
        df = df.sort_values("time").drop_duplicates("time", keep="last")
        df = df.reset_index(drop=True)

        if df.empty:
            raise ValueError(f"{csv_path.name} contains no valid OHLCV rows")

        prices = df[list(self.PRICE_COLUMNS)]
        invalid_prices = (
            (prices <= 0).any(axis=1)
            | (df["high"] < df["low"])
            | (df["high"] < df[["open", "close"]].max(axis=1))
            | (df["low"] > df[["open", "close"]].min(axis=1))
            | (df["tick_volume"] < 0)
        )
        if invalid_prices.any():
            raise ValueError(
                f"{csv_path.name} contains {int(invalid_prices.sum())} invalid OHLCV rows"
            )

        return df

    def scan_folder(self, csv_folder):
        root = pathlib.Path(csv_folder)
        if not root.is_dir():
            raise ValueError(f"CSV folder does not exist: {root}")

        symbols = []
        all_timeframes = set()

        for symbol_dir in sorted(root.iterdir()):
            if not symbol_dir.is_dir():
                continue

            symbol = symbol_dir.name
            if not re.fullmatch(Config.SYMBOL_PATTERN, symbol):
                continue

            found = [
                timeframe for timeframe in self.TIMEFRAME_SECONDS
                if (symbol_dir / f"{symbol}_{timeframe}.csv").is_file()
            ]

            if found:
                symbols.append((symbol, found))
                all_timeframes.update(found)
                self._log(f"   ✓ {symbol}: {', '.join(found)}")

        return symbols, sorted(
            all_timeframes,
            key=lambda tf: list(self.TIMEFRAME_SECONDS).index(tf),
        )

    def build_samples_for_symbol(self, csv_folder, symbol, timeframe, horizon):
        if horizon < 1:
            raise ValueError("Forecast horizon must be at least one bar")

        df = self.load_csv_data(symbol, timeframe, csv_folder)
        df = df.iloc[-Config.MAX_ROWS_PER_SYMBOL:].reset_index(drop=True)

        minimum_samples = Config.MIN_SAMPLES_BY_TIMEFRAME[timeframe]
        required_bars = Config.WINDOW + horizon + minimum_samples

        self._log(
            f"🔎 {symbol}_{timeframe}: {len(df):,} valid bars, "
            f"{required_bars:,} required "
            f"(window={Config.WINDOW}, horizon={horizon}, "
            f"minimum_samples={minimum_samples})"
        )

        if len(df) < required_bars:
            self._log(
                f"⚠️ {symbol}_{timeframe}: insufficient valid bars for training"
            )
            return None

        timestamps = df["time"]
        timestamp_seconds = (
            timestamps - pd.Timestamp(0, tz="UTC")
        ).dt.total_seconds().to_numpy()
        deltas = np.diff(timestamp_seconds)
        expected_seconds = self.TIMEFRAME_SECONDS[timeframe]

        gap_indices = np.flatnonzero(
            deltas > expected_seconds * 1.5
        ) + 1

        raw = np.column_stack([
            df["open"].to_numpy(dtype=np.float32),
            df["high"].to_numpy(dtype=np.float32),
            df["low"].to_numpy(dtype=np.float32),
            df["close"].to_numpy(dtype=np.float32),
            df["tick_volume"].to_numpy(dtype=np.float32),
        ]).astype(np.float32)

        if not np.isfinite(raw).all():
            raise ValueError(f"{symbol}_{timeframe}: non-finite input features")

        close = df["close"].to_numpy(dtype=np.float64)
        target = np.full(len(df), np.nan, dtype=np.float64)
        target[:-horizon] = close[horizon:] / close[:-horizon] - 1.0

        valid_target = np.isfinite(target[:-horizon])
        if not valid_target.all():
            raise ValueError(f"{symbol}_{timeframe}: invalid forward-return targets")

        if len(gap_indices):
            self._log(
                f"⚠️ {symbol}_{timeframe}: {len(gap_indices)} timestamp gaps "
                f"larger than 1.5 expected bars; session closures may be included"
            )

        sample_count = len(df) - Config.WINDOW + 1 - horizon

        self._log(
            f"📊 {symbol}_{timeframe}: {len(df):,} bars, "
            f"{sample_count:,} samples, {len(gap_indices)} detected gaps"
        )

        return {
            "raw": raw,
            "target": target,
            "time": timestamps.to_numpy(),
            "gap_indices": gap_indices,
            "timeframe_seconds": expected_seconds,
        }


class SequenceTrainer:
    PT_EXT = ".pt"
    ONNX_EXT = ".onnx"
    INFO_EXT = ".info.json"

    ONNX_OPSET = 14
    EVAL_BATCH = 512
    GRAD_CLIP = 1.0
    EARLY_STOPPING_PATIENCE = 8
    LOG_EVERY_EPOCHS = 5
    SIGNAL_QUANTILE = 0.70
    PARITY_SAMPLES = 32

    PARITY_REL_TOL = 1e-2

    TRAIN_FRACTION = 0.65
    VALIDATION_FRACTION = 0.15
    CALIBRATION_FRACTION = 0.10
    MIN_SPLIT_SAMPLES = 10

    def __init__(self, log_callback, symbol, timeframe, arch):
        self.log_callback = log_callback
        self.symbol = symbol
        self.timeframe = timeframe
        self.arch = arch
        self.model = None
        self.is_trained = False
        self.stop_flag = False
        self.training_history = {}
        self.threshold = 0.0
        self.parity_windows = None
        self.parity_time = None

        self._log(
            f"🚀 {arch} trainer [{symbol} {timeframe}] initialized "
            f"({Config.DEVICE})"
        )

    def _log(self, message):
        self.log_callback(message)

    def stop_training(self):
        self.stop_flag = True
        self._log("⏹️ Training stop requested")

    def _batch(self, windows, t_idx):
        x = windows[t_idx - Config.WINDOW + 1]
        return torch.from_numpy(
            np.ascontiguousarray(x.transpose(0, 2, 1))
        ).to(Config.DEVICE)

    def _predict(self, windows, t_idx):
        self.model.eval()
        predictions = []

        with torch.no_grad():
            for start in range(0, len(t_idx), self.EVAL_BATCH):
                batch_indices = t_idx[start:start + self.EVAL_BATCH]
                output = self.model(
                    self._batch(windows, batch_indices)
                ).squeeze(1)
                predictions.append(output.cpu().numpy())

        return np.concatenate(predictions)

    def train_single_timeframe(
        self,
        data,
        epochs,
        learning_rate,
        batch_size,
        horizon,
        auto_tune,
        status_callback,):

        self.stop_flag = False
        self.is_trained = False
        self.training_history = {}

        tag = f"{self.symbol} {self.timeframe} {self.arch}"

        if horizon < 1 or epochs < 1 or batch_size < 1 or learning_rate <= 0:
            raise ValueError(f"[{tag}] Invalid training parameters")

        if auto_tune:
            hp = Config.TIMEFRAME_HYPERPARAMS[self.timeframe]
            epochs = hp["epochs"]
            hidden = hp["hidden"]
            dropout = hp["dropout"]
            weight_decay = hp["weight_decay"]
        else:
            hidden = Config.DEFAULT_HIDDEN
            dropout = Config.DEFAULT_DROPOUT
            weight_decay = Config.DEFAULT_WEIGHT_DECAY

        raw = np.asarray(data["raw"], dtype=np.float32)
        target = np.asarray(data["target"], dtype=np.float64)

        if raw.ndim != 2 or raw.shape[1] != len(Config.RAW_CHANNELS):
            raise ValueError(f"[{tag}] Invalid raw input shape: {raw.shape}")

        if len(raw) != len(target):
            raise ValueError(f"[{tag}] Raw and target lengths do not match")

        if not np.isfinite(raw).all():
            raise ValueError(f"[{tag}] Input data contains non-finite values")

        samples = np.arange(
            Config.WINDOW - 1,
            len(raw) - horizon,
            dtype=np.int64,
        )
        samples = samples[np.isfinite(target[samples])]

        n_samples = len(samples)
        train_end = int(n_samples * self.TRAIN_FRACTION)
        val_end = int(
            n_samples * (
                self.TRAIN_FRACTION + self.VALIDATION_FRACTION
            )
        )
        calibration_end = int(
            n_samples * (
                self.TRAIN_FRACTION
                + self.VALIDATION_FRACTION
                + self.CALIBRATION_FRACTION
            )
        )

        # The horizon-sized embargo prevents target overlap between splits.
        train_t = samples[:train_end]
        val_t = samples[train_end + horizon:val_end]
        calibration_t = samples[
            val_end + horizon:calibration_end
        ]
        test_t = samples[calibration_end + horizon:]

        split_sizes = (
            len(train_t),
            len(val_t),
            len(calibration_t),
            len(test_t),
        )
        if min(split_sizes) < self.MIN_SPLIT_SAMPLES:
            raise ValueError(
                f"[{tag}] Not enough samples for chronological "
                f"train/validation/calibration/test splits: {split_sizes}. "
                f"Use more history or a shorter forecast horizon."
            )

        y_train = target[train_t]
        y_val = target[val_t]
        y_calibration = target[calibration_t]
        y_test = target[test_t]

        y_mean = float(y_train.mean())
        y_std = float(y_train.std())

        if not np.isfinite(y_mean) or not np.isfinite(y_std) or y_std <= 1e-12:
            raise ValueError(
                f"[{tag}] Training target has invalid or near-zero "
                f"standard deviation"
            )

        y_train_std = ((y_train - y_mean) / y_std).astype(np.float32)
        y_val_std = ((y_val - y_mean) / y_std).astype(np.float32)

        windows = np.lib.stride_tricks.sliding_window_view(
            raw,
            Config.WINDOW,
            axis=0,
        )

        torch.manual_seed(Config.RANDOM_STATE)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(Config.RANDOM_STATE)

        rng = np.random.default_rng(Config.RANDOM_STATE)
        self.model = SequenceModel(
            self.arch,
            hidden,
            dropout,
        ).to(Config.DEVICE)

        n_params = sum(
            parameter.numel()
            for parameter in self.model.parameters()
        )

        self._log(
            f"[{tag}] Samples: train={len(train_t):,}, "
            f"validation={len(val_t):,}, calibration={len(calibration_t):,}, "
            f"test={len(test_t):,}; embargo={horizon} bars"
        )
        self._log(
            f"[{tag}] Window: {Config.WINDOW} bars x "
            f"{len(Config.RAW_CHANNELS)} channels -> "
            f"{len(Config.RAW_CHANNELS)} channels"
        )
        self._log(
            f"[{tag}] Params: {n_params:,}, hidden={hidden}, "
            f"dropout={dropout}, weight_decay={weight_decay}, "
            f"epochs={epochs}, lr={learning_rate}, batch={batch_size}"
        )

        n_channels = len(Config.RAW_CHANNELS)
        total = torch.zeros(n_channels, dtype=torch.float64)
        total_sq = torch.zeros(n_channels, dtype=torch.float64)
        count = 0

        # Input normalization is fitted on training windows only.
        with torch.no_grad():
            for start in range(0, len(train_t), self.EVAL_BATCH):
                batch_indices = train_t[start:start + self.EVAL_BATCH]
                batch = self._batch(windows, batch_indices).reshape(-1, n_channels).double().cpu()
                total += batch.sum(dim=0)
                total_sq += (batch * batch).sum(dim=0)
                count += batch.shape[0]

        mean = total / count
        variance = torch.clamp(
            total_sq / count - mean * mean,
            min=0.0,
        )
        std = torch.sqrt(variance).clamp(min=1e-6)

        if not torch.isfinite(mean).all() or not torch.isfinite(std).all():
            raise ValueError(
                f"[{tag}] Non-finite input normalization values"
            )

        self.model.in_mean.copy_(mean.float())
        self.model.in_std.copy_(std.float())
        self.model.y_mean.fill_(y_mean)
        self.model.y_std.fill_(y_std)

        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=epochs,
        )

        best = {
            "val": float("inf"),
            "epoch": 0,
            "state": None,
        }
        stale = 0
        epochs_run = 0

        for epoch in range(1, epochs + 1):
            self.model.train()
            order = rng.permutation(len(train_t))
            running_loss = 0.0

            for start in range(0, len(order), batch_size):
                if self.stop_flag:
                    return False

                picks = order[start:start + batch_size]
                batch_indices = train_t[picks]

                optimizer.zero_grad(set_to_none=True)

                output = self.model.standardized(
                    self._batch(windows, batch_indices)
                ).squeeze(1)

                expected = torch.from_numpy(
                    y_train_std[picks]
                ).to(Config.DEVICE)

                loss = functional.smooth_l1_loss(output, expected)
                loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.GRAD_CLIP,
                )
                optimizer.step()

                running_loss += loss.item() * len(picks)

            scheduler.step()
            epochs_run = epoch

            val_prediction = self._predict(windows, val_t)
            val_prediction_std = torch.from_numpy(
                ((val_prediction - y_mean) / y_std).astype(np.float32)
            )
            val_loss = float(
                functional.smooth_l1_loss(
                    val_prediction_std,
                    torch.from_numpy(y_val_std),
                )
            )
            train_loss = running_loss / len(order)

            if val_loss < best["val"]:
                best["val"] = val_loss
                best["epoch"] = epoch
                best["state"] = {
                    key: value.detach().cpu().clone()
                    for key, value in self.model.state_dict().items()
                }
                stale = 0
            else:
                stale += 1

            status_callback(
                f"[{tag}] Epoch {epoch}/{epochs} "
                f"train={train_loss:.5f} val={val_loss:.5f} "
                f"best={best['val']:.5f}@{best['epoch']}"
            )

            if epoch % self.LOG_EVERY_EPOCHS == 0:
                self._log(
                    f"[{tag}] Epoch {epoch}/{epochs} "
                    f"train={train_loss:.5f} val={val_loss:.5f} "
                    f"best={best['val']:.5f}@{best['epoch']}"
                )

            if stale >= self.EARLY_STOPPING_PATIENCE:
                self._log(
                    f"[{tag}] Early stop at epoch {epoch}; "
                    f"best epoch {best['epoch']}"
                )
                break

        if self.stop_flag:
            return False

        if best["state"] is None:
            raise RuntimeError(
                f"[{tag}] Training did not produce a valid model"
            )

        self.model.load_state_dict(best["state"])
        self.model.eval()

        # Threshold is calibrated on a separate chronological period.
        calibration_prediction = self._predict(windows, calibration_t)
        self.threshold = float(
            np.quantile(
                np.abs(calibration_prediction),
                self.SIGNAL_QUANTILE,
            )
        )

        # All reported metrics use the untouched test split.
        test_prediction = self._predict(windows, test_t)
        valid = y_test != 0
        signaled = (
            (np.abs(test_prediction) > self.threshold)
            & valid
        )

        directional_accuracy = (
            float(
                (
                    np.sign(test_prediction[valid])
                    == np.sign(y_test[valid])
                ).mean()
            )
            if valid.any() else 0.0
        )

        signal_hit_rate = (
            float(
                (
                    np.sign(test_prediction[signaled])
                    == np.sign(y_test[signaled])
                ).mean()
            )
            if signaled.any() else 0.0
        )

        signal_avg_return = (
            float(
                (
                    np.sign(test_prediction[signaled])
                    * y_test[signaled]
                ).mean()
            )
            if signaled.any() else 0.0
        )

        self.training_history = {
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "architecture": self.arch,
            "parameters": n_params,
            "hidden": hidden,
            "dropout": dropout,
            "weight_decay": weight_decay,
            "train_samples": int(len(train_t)),
            "validation_samples": int(len(val_t)),
            "calibration_samples": int(len(calibration_t)),
            "test_samples": int(len(test_t)),
            "best_epoch": int(best["epoch"]),
            "epochs_run": int(epochs_run),
            "validation_loss": float(best["val"]),
            "test_l2": float(
                np.mean((test_prediction - y_test) ** 2)
            ),
            "test_directional_accuracy": directional_accuracy,
            "signal_threshold": self.threshold,
            "test_signal_coverage": float(
                signaled.sum() / len(y_test)
            ),
            "test_signal_hit_rate": signal_hit_rate,
            "test_signal_avg_return": signal_avg_return,
            "test_metrics_are_before_costs": True,
        }

        self._log(
            f"[{tag}] Test directional accuracy: "
            f"{directional_accuracy * 100:.1f}% "
            f"(test L2 {self.training_history['test_l2']:.3e})"
        )
        self._log(
            f"[{tag}] Calibrated threshold: {self.threshold:.3e}; "
            f"test coverage: "
            f"{self.training_history['test_signal_coverage'] * 100:.1f}%; "
            f"hit rate: {signal_hit_rate * 100:.1f}%; "
            f"average return/signal: {signal_avg_return * 100:.4f}% "
            f"(before costs)"
        )

        tail = test_t[-self.PARITY_SAMPLES:]
        self.parity_windows = np.ascontiguousarray(
            windows[tail - Config.WINDOW + 1].transpose(0, 2, 1)
        )
        self.parity_time = data["time"][test_t[-1]]
        self.is_trained = True

        return True

    def save_model(self, model_dir, filename_stem, horizon):
        if not self.is_trained or self.model is None:
            raise RuntimeError(
                "Cannot export a model that has not completed training"
            )

        if self.parity_windows is None or len(self.parity_windows) == 0:
            raise RuntimeError(
                "Model export requires parity test windows"
            )

        if horizon < 1:
            raise ValueError("Forecast horizon must be positive")

        if not re.fullmatch(Config.SYMBOL_PATTERN, self.symbol):
            raise ValueError(f"Unsafe symbol name: {self.symbol!r}")

        if not re.fullmatch(r"[A-Za-z0-9._-]+", filename_stem):
            raise ValueError(f"Unsafe model filename: {filename_stem!r}")

        output_dir = pathlib.Path(model_dir).expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)

        pt_path = output_dir / f"{filename_stem}{self.PT_EXT}"
        onnx_path = output_dir / f"{filename_stem}{self.ONNX_EXT}"
        json_path = output_dir / f"{filename_stem}{self.INFO_EXT}"

        pt_temp = output_dir / f".{filename_stem}.tmp{self.PT_EXT}"
        onnx_temp = output_dir / f".{filename_stem}.tmp{self.ONNX_EXT}"
        json_temp = output_dir / f".{filename_stem}.tmp{self.INFO_EXT}"

        tag = f"{self.symbol} {self.timeframe} {self.arch}"

        # Prepare the trained model on CPU for portable inference exports.
        self.model.cpu().eval()

        export = SignalExport(
            self.model,
            self.threshold,
        ).eval()

        windows = torch.from_numpy(self.parity_windows)

        # 1. TorchScript export.
        with torch.no_grad():
            traced = torch.jit.trace(export, windows[-1:])

        traced.save(str(pt_temp))
        os.replace(pt_temp, pt_path)

        self._log(f"✅ TorchScript saved: {pt_path}")

        # 2. Build metadata-only JSON (filenames only, no absolute paths).
        last_window = self.parity_windows[-1]
        parity_timestamp = int(
            pd.Timestamp(self.parity_time).timestamp()
        )

        info = {
            "model_type": "torch.SequenceModel",
            "architecture": self.arch,
            "target_type": "forward_return",
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "horizon": int(horizon),
            "window": int(Config.WINDOW),
            "channels": int(len(Config.RAW_CHANNELS)),
            "raw_channels": list(Config.RAW_CHANNELS),
            "signal_threshold": float(self.threshold),
            "signal_quantile": float(self.SIGNAL_QUANTILE),
            "pt_file": pt_path.name,
            "onnx_file": onnx_path.name,
            "info_file": json_path.name,
            "onnx_validation": "pending",
            "parity_max_diff": None,
            "parity_time": parity_timestamp,
            "parity_input": last_window.reshape(-1).tolist(),
            "training_history": self.training_history,
            "training_time": datetime.datetime.now().isoformat(),
        }

        json_temp.write_text(json.dumps(info, indent=2, allow_nan=False))
        os.replace(json_temp, json_path)

        self._log(f"✅ JSON metadata saved: {json_path}")

        # 3. ONNX export.
        with warnings.catch_warnings(), torch.no_grad():
            warnings.simplefilter("ignore")

            torch.onnx.export(
                export,
                windows[-1:],
                str(onnx_temp),
                input_names=["input"],
                output_names=["prediction", "signal"],
                opset_version=self.ONNX_OPSET,
                dynamo=False,
            )

        onnx_model = onnx.load(str(onnx_temp))

        onnx_metadata = {
            "architecture": self.arch,
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "horizon": str(horizon),
            "window": str(Config.WINDOW),
            "channels": ",".join(Config.RAW_CHANNELS),
            "signal_threshold": repr(self.threshold),
        }

        for key, value in onnx_metadata.items():
            entry = onnx_model.metadata_props.add()
            entry.key = key
            entry.value = value

        onnx.save(onnx_model, str(onnx_temp))
        onnx.checker.check_model(onnx_model)

        # CPU execution is used as an ONNX reference.
        # This does not represent MT5 OpenCL execution.
        session = onnxruntime.InferenceSession(
            str(onnx_temp),
            providers=["CPUExecutionProvider"],
        )

        reference_prediction = []
        reference_signal = []
        onnx_prediction = []
        onnx_signal = []

        with torch.no_grad():
            for index in range(len(windows)):
                current_window = windows[index:index + 1]

                prediction, signal = traced(current_window)

                reference_prediction.append(prediction.item())
                reference_signal.append(signal.item())

                ort_prediction, ort_signal = session.run(
                    None,
                    {"input": current_window.numpy()},
                )

                onnx_prediction.append(ort_prediction.item())
                onnx_signal.append(ort_signal.item())

        reference_prediction = np.asarray(reference_prediction)
        reference_signal = np.asarray(reference_signal)
        onnx_prediction = np.asarray(onnx_prediction)
        onnx_signal = np.asarray(onnx_signal)

        tolerance = (
            self.PARITY_REL_TOL * self.model.y_std.item()
        )

        differences = np.abs(
            reference_prediction - onnx_prediction
        )

        max_difference = float(differences.max())
        worst_index = int(np.argmax(differences))

        away_from_threshold = (
            np.abs(
                np.abs(reference_prediction) - self.threshold
            ) > tolerance
        )

        prediction_parity_ok = max_difference <= tolerance

        signal_parity_ok = bool(
            (
                reference_signal[away_from_threshold]
                == onnx_signal[away_from_threshold]
            ).all()
        )

        del session, onnx_model

        # 4. Publish ONNX only when parity succeeds.
        if prediction_parity_ok and signal_parity_ok:
            os.replace(onnx_temp, onnx_path)

            info["onnx_validation"] = "passed"
            info["parity_max_diff"] = max_difference
            info["parity_output"] = [
                float(reference_prediction[-1]),
                float(reference_signal[-1]),
            ]

            self._log(f"✅ ONNX saved for MT5: {onnx_path}")
            self._log(
                f"[{tag}] TorchScript/ONNX parity OK: "
                f"max difference {max_difference:.3e}; "
                f"tolerance {tolerance:.3e}"
            )

        else:
            onnx_temp.unlink()

            info["onnx_validation"] = "failed"
            info["parity_max_diff"] = max_difference
            info["parity_worst_sample"] = worst_index
            info["parity_tolerance"] = tolerance
            info["parity_torch_prediction"] = float(
                reference_prediction[worst_index]
            )
            info["parity_onnx_prediction"] = float(
                onnx_prediction[worst_index]
            )
            info["parity_signal_ok"] = signal_parity_ok

            self._log(
                f"⚠️ [{tag}] ONNX parity FAILED: "
                f"difference={max_difference:.6e}, "
                f"tolerance={tolerance:.6e}, "
                f"sample={worst_index}, "
                f"signal_parity={signal_parity_ok}"
            )

            self._log(
                f"⚠️ [{tag}] ONNX rejected. "
                "TorchScript and JSON remain available."
            )

        # 5. Update the metadata-only JSON with final ONNX status.
        json_temp.write_text(json.dumps(info, indent=2, allow_nan=False))
        os.replace(json_temp, json_path)

        self._log(f"✅ JSON metadata updated: {json_path}")
        self._log(
            f"[{tag}] Export process completed: "
            f"PT=available, JSON=available, "
            f"ONNX={info['onnx_validation']}"
        )


class PredictNNTrainerGUI(ctk.CTk):
    INI_PATH = pathlib.Path(__file__).with_suffix(".ini")
    UI_POLL_MS = 50

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
    FONT_LICENSE = ("TkDefaultFont", 14)
    FONT_NAME = ("TkDefaultFont", 16, "bold")
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

        self.ui_queue = queue.Queue()
        self.symbol_timeframes = {}
        self.is_training = False
        self.stop_flag = False
        self.current_trainer = None
        self.data_preparer = StreamingDataPreparer(self.gui_logger.log)

        self._create_ui()
        self._load_state()
        self.gui_logger.set_log_widget(self.log_text)
        self.after(self.UI_POLL_MS, self._poll_ui)
        self.after(500, self.scan_data_source)
        self.protocol("WM_DELETE_WINDOW", self._on_closing)

        self.update_idletasks()
        self.deiconify()

    def _poll_ui(self):
        while not self.ui_queue.empty():
            function, argument = self.ui_queue.get_nowait()
            function(argument)
        self.after(self.UI_POLL_MS, self._poll_ui)

    def update_status(self, message):
        self.ui_queue.put((self._set_status, message))

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

        frame = self._section(main)
        ctk.CTkLabel(frame, text="Copyright Su Nie | BSD-3C License | https://github.com/can87683",
            font=self.FONT_LICENSE,
            text_color=self.TITLE_COLOR).pack(fill="x",
            padx=self.PAD, pady=self.PAD)

        ctk.CTkLabel(frame, text="DeepLearn ML Trainer",
            font=self.FONT_NAME,
            text_color=self.TEXT_COLOR).pack(fill="x",
            padx=self.PAD, pady=self.PAD)

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
        self.epochs = tk.StringVar(value=str(Config.DEFAULT_EPOCHS))
        self.learning_rate = tk.StringVar(value=str(Config.DEFAULT_LEARNING_RATE))
        self.batch_size = tk.StringVar(value=str(Config.DEFAULT_BATCH_SIZE))
        self.auto_tune_var = tk.BooleanVar(value=True)
        frame = self._section(main, "Parameters")
        row = self._row(frame)
        params = (("Horizon:", self.horizon), ("Epochs:", self.epochs), ("LR:", self.learning_rate), ("Batch:", self.batch_size))
        for i, (text, var) in enumerate(params):
            row.grid_columnconfigure(i * 2 + 1, weight=1)
            self._label(row, text).grid(row=0, column=i * 2, padx=self.PAD, pady=self.PAD, sticky="w")
            self._entry(row, var, width=self.ENTRY_W).grid(row=0, column=i * 2 + 1, padx=self.PAD, pady=self.PAD, sticky="ew")
        self._check(frame, "Auto-tune epochs / hidden / dropout / weight decay per timeframe", self.auto_tune_var, self.CHECK_W).pack(fill="x", padx=self.PAD, pady=self.PAD)

        self.arch_vars = {}
        frame = self._section(main, "Models (one per symbol/timeframe each)")
        row = self._row(frame)
        for i, name in enumerate(SequenceModel.ARCHITECTURES):
            self.arch_vars[name] = tk.BooleanVar(value=True)
            row.grid_columnconfigure(i, weight=1)
            self._check(row, name, self.arch_vars[name], self.CHECK_W).grid(row=0, column=i, padx=self.PAD, pady=self.PAD, sticky="ew")

        self.tf_vars = {}
        frame = self._section(main, "Timeframes (one model per symbol)")
        row = self._row(frame)
        for i, tf in enumerate(Config.TIMEFRAMES):
            self.tf_vars[tf] = tk.BooleanVar(value=True)
            row.grid_columnconfigure(i, weight=1)
            self._check(row, tf, self.tf_vars[tf], self.TF_CHECK_W).grid(row=0, column=i, padx=self.PAD, pady=self.PAD, sticky="ew")

        test_fraction = 1 - SequenceTrainer.TRAIN_FRACTION - SequenceTrainer.VALIDATION_FRACTION - SequenceTrainer.CALIBRATION_FRACTION
        pipeline = (
            f"Input: last {Config.WINDOW} completed bars [1,{Config.WINDOW},{len(Config.RAW_CHANNELS)}] = {', '.join(Config.RAW_CHANNELS)}\n"
            f"Scaler: fit on train windows only, clip ±{SequenceModel.INPUT_CLIP:g}, baked into the graph\n"
            f"Target: close[t+horizon]/close[t]-1, Huber loss, AdamW + cosine LR, grad clip {SequenceTrainer.GRAD_CLIP:g}\n"
            f"Splits: {SequenceTrainer.TRAIN_FRACTION:.0%} train / {SequenceTrainer.VALIDATION_FRACTION:.0%} validation / "
            f"{SequenceTrainer.CALIBRATION_FRACTION:.0%} calibration / {test_fraction:.0%} test, chronological, embargo = horizon; "
            f"early stop patience {SequenceTrainer.EARLY_STOPPING_PATIENCE}, best epoch restored\n"
            f"Signal: sign(prediction) if |prediction| > {SequenceTrainer.SIGNAL_QUANTILE:.0%} calibration quantile of |prediction|, baked into the graph; metrics reported on test split only, before costs\n"
            f"Export: TorchScript .pt + ONNX opset {SequenceTrainer.ONNX_OPSET} (outputs: prediction, signal) + parity check + .info.json"
        )
        frame = self._section(main, "Pipeline")
        self._label(frame, pipeline, anchor="w", justify="left", wraplength=self.WRAP_W).pack(fill="x", padx=self.PAD, pady=self.PAD)

        frame = self._section(main, "Controls")
        row = self._row(frame)
        for c in range(2):
            row.grid_columnconfigure(c, weight=1)
        self.train_btn = self._button(row, "🚀 Start", self.start_training, width=self.BUTTON_W, fg_color=self.TRAIN_COLOR, hover_color=self.TRAIN_HOVER)
        self.train_btn.grid(row=0, column=0, padx=self.PAD, pady=self.PAD, sticky="ew")
        self.stop_btn = self._button(row, "⏹️ Stop", self.stop_training, width=self.BUTTON_W, fg_color=self.STOP_COLOR, hover_color=self.STOP_HOVER, state="disabled")
        self.stop_btn.grid(row=0, column=1, padx=self.PAD, pady=self.PAD, sticky="ew")

        frame = self._section(main, "Status")
        self.status_label = self._label(frame, "📊 No folder selected", anchor="w")
        self.status_label.pack(fill="x", padx=self.PAD, pady=self.PAD)
        self.status_text = self._textbox(frame, height=self.STATUS_H)
        self.status_text.pack(fill="x", padx=self.PAD, pady=self.PAD)
        self.status_text.insert("1.0", "Ready")
        self.status_text.configure(state="disabled")

        frame = self._section(main, "Log", expand=True)
        self.log_text = self._textbox(frame)
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

        self.geometry(cp.get("window", "geometry"))

        self.csv_path_var.set(cp["paths"]["csv_folder"])
        self.model_path_var.set(cp["paths"]["model_folder"])
        self.horizon.set(cp["params"]["horizon"])
        self.epochs.set(cp["params"]["epochs"])
        self.learning_rate.set(cp["params"]["learning_rate"])
        self.batch_size.set(cp["params"]["batch_size"])
        self.auto_tune_var.set(cp.getboolean("params", "auto_tune"))

        for name, var in self.arch_vars.items():
            var.set(cp.getboolean("models", name))

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
            "epochs": self.epochs.get(),
            "learning_rate": self.learning_rate.get(),
            "batch_size": self.batch_size.get(),
            "auto_tune": str(self.auto_tune_var.get()),
        }
        cp["models"] = {
            name: str(var.get()) for name, var in self.arch_vars.items()
        }
        cp["timeframes"] = {
            tf: str(var.get()) for tf, var in self.tf_vars.items()
        }

        with open(self.INI_PATH, "w") as file:
            cp.write(file)

    def update_status(self, message):
        self.after(0, self._set_status, message)

    def _set_status(self, message):
        self.status_text.configure(state="normal")
        self.status_text.delete("1.0", "end")
        self.status_text.insert("1.0", message)
        self.status_text.configure(state="disabled")

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

        selected_timeframes = [tf for tf, var in self.tf_vars.items() if var.get()]
        selected_archs = [name for name, var in self.arch_vars.items() if var.get()]
        if not selected_timeframes or not selected_archs:
            self.gui_logger.log("❌ Select at least one model and one timeframe", "error")
            return

        csv_folder = self.csv_path_var.get()
        model_folder = self.model_path_var.get()
        horizon = int(self.horizon.get())
        epochs = int(self.epochs.get())
        learning_rate = float(self.learning_rate.get())
        batch_size = int(self.batch_size.get())
        auto_tune = self.auto_tune_var.get()
        symbol_timeframes = dict(self.symbol_timeframes)

        self.stop_flag = False
        self._set_training(True)

        def training_task():
            self.gui_logger.log("=" * 60)
            self.gui_logger.log("🚀 Starting Per-Symbol Per-Timeframe NN Training")
            self.gui_logger.log("=" * 60)
            self.gui_logger.log(f"📂 CSV Folder: {csv_folder}")
            self.gui_logger.log(f"💾 System RAM: {MemoryOptimizer.get_system_memory_available_gb():.1f} GB available, device: {Config.DEVICE}")
            self.gui_logger.log(f"🧮 Mandatory columns: {', '.join(Config.REQUIRED_COLUMNS)}")
            self.gui_logger.log(f"🤖 Models: {', '.join(selected_archs)}")

            jobs = [(symbol, tf) for tf in selected_timeframes for symbol, tfs in symbol_timeframes.items() if tf in tfs]
            self.gui_logger.log(f"🧱 Models to train: {len(jobs) * len(selected_archs)}")

            results = {}

            for symbol, timeframe in jobs:
                if self.stop_flag:
                    self.gui_logger.log("⏹️ Training stopped by user", "warning")
                    break

                tag = f"{symbol} {timeframe}"
                self.gui_logger.log("-" * 60)
                self.gui_logger.log(f"▶️ {tag}")
                self.update_status(f"[{tag}] Building samples...")

                data = self.data_preparer.build_samples_for_symbol(csv_folder, symbol, timeframe, horizon)
                if data is None:
                    self.gui_logger.log(f"⚠️ [{tag}] Skipped: not enough bars for window + horizon + {Config.MIN_SAMPLES} samples", "warning")
                    results[tag] = "skipped"
                    continue

                for arch in selected_archs:
                    if self.stop_flag:
                        break

                    job_tag = f"{arch} {tag}"
                    self.update_status(f"[{job_tag}] Training...")
                    trainer = SequenceTrainer(self.gui_logger.log, symbol, timeframe, arch)
                    self.current_trainer = trainer

                    success = trainer.train_single_timeframe(
                        data=data,
                        epochs=epochs,
                        learning_rate=learning_rate,
                        batch_size=batch_size,
                        horizon=horizon,
                        auto_tune=auto_tune,
                        status_callback=self.update_status,
                    )

                    if trainer.stop_flag:
                        results[job_tag] = "stopped"
                        self.stop_flag = True
                    elif success:
                        filename_stem = f"{Config.MODEL_PREFIX}_{arch.lower()}_{symbol}_{timeframe}"
                        trainer.save_model(model_dir=model_folder, filename_stem=filename_stem, horizon=horizon)
                        self.gui_logger.log(f"✅ [{job_tag}] Model saved: {filename_stem}{SequenceTrainer.PT_EXT}", "success")
                        results[job_tag] = "trained"

                    del trainer
                    self.current_trainer = None
                    MemoryOptimizer.clear_cache()

                del data

            counts = {status: list(results.values()).count(status) for status in set(results.values())}
            summary = ", ".join(f"{status}: {n}" for status, n in sorted(counts.items()))
            self.gui_logger.log("=" * 60)
            self.gui_logger.log(f"🏁 Summary: {summary}")
            self.update_status(f"Done. {summary}")
            self.ui_queue.put((self._set_training, False))

        threading.Thread(target=training_task, daemon=True).start()

    def stop_training(self):
        self.stop_flag = True
        if self.current_trainer is not None:
            self.current_trainer.stop_training()
        self.gui_logger.log("⏹️ Stopping training...", "warning")

    def _on_closing(self):
        message = "Training is running.\n\nStop training and quit?" if self.is_training else "Quit AI Predict Train?"
        if not messagebox.askyesno("Quit", message):
            return
        if self.is_training:
            self.stop_training()
        self._save_state()
        self.gui_logger.close()
        self.destroy()


if __name__ == "__main__":
    app = PredictNNTrainerGUI()
    app.mainloop()