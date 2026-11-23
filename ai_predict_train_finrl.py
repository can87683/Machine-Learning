#!/usr/bin/env python3
# ai_predict_train_finrl.py
# Copyright Su Nie | BSD-3C License | https://github.com/can87683

"""
customtkinter gymnasium numpy pandas psutil setproctitle stable-baselines3
"""

import os
import sys
import setproctitle
setproctitle.setproctitle(os.path.basename(sys.argv[0]))

import customtkinter as ctk
import tkinter as tk
from tkinter import filedialog, messagebox
import threading
import configparser
from pathlib import Path
import numpy as np
import pandas as pd
import psutil
from datetime import datetime
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO, SAC
import onnxconverter_common
import onnxmltools


class Config:
    HOME = Path.home()
    MODEL_DIR = HOME / "ai_model_ml_symbol"
    DATASET_DIR = HOME / "mt5_data"

    MAX_ROWS_PER_SYMBOL = 20000
    MAX_DATASET_ROWS = 200000
    DEFAULT_FEATURES = ['time', 'open', 'high', 'low', 'close', 'tick_volume']

    DEFAULT_ALGO = "PPO"
    DEFAULT_TIMESTEPS = 50000
    DEFAULT_LR = 0.0003

    TIMEFRAMES = ['M5', 'M15', 'H1', 'H4', 'D1', 'W1']

    MODEL_PREFIX = "ai_predict_finrl"
    MODEL_EXT = ".zip"
    METADATA_EXT = ".json"
    ONNX_EXT = ".onnx"
    ONNX_OPSET = 14

    APP_TITLE = "FinRL Model Trainer"
    APP_GEOMETRY = "640x1020"
    THEME = "dark"
    COLOR_THEME = "dark-blue"

    SEED = 42


class GPUManager:
    def __init__(self, log_callback):
        self.log_callback = log_callback
        self.gpu_available = False
        self.gpu_name = None
        self.gpu_memory_gb = 0.0

        import subprocess
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
            self.log_callback(f"GPU detected: {self.gpu_name} ({self.gpu_memory_gb:.2f} GB)")
        else:
            self.log_callback("No GPU detected, using CPU")

    def log_system_info(self):
        self.log_callback("=" * 60)
        self.log_callback("System & Library Information")
        self.log_callback("=" * 60)
        self.log_callback(f"Python: {sys.version.split()[0]}")
        import stable_baselines3
        self.log_callback(f"stable-baselines3: {stable_baselines3.__version__}")
        import gymnasium
        self.log_callback(f"gymnasium: {gymnasium.__version__}")
        if self.gpu_available:
            self.log_callback(f"GPU Device: {self.gpu_name}")
            self.log_callback(f"GPU Memory: {self.gpu_memory_gb:.2f} GB")
        else:
            self.log_callback("No GPU detected. Will use CPU.")
        self.log_callback("=" * 60)


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
    def __init__(self, gui_logger, original_stream):
        self.gui_logger = gui_logger
        self.original_stream = original_stream

    def write(self, message):
        if message and message.strip():
            self.original_stream.write(message)
            self.original_stream.flush()
            self.gui_logger.log(message.strip(), "console")

    def flush(self):
        self.original_stream.flush()


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


class TradingEnv(gym.Env):
    def __init__(self, df, initial_balance=10000):
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.initial_balance = initial_balance
        self.current_step = 0
        self.balance = initial_balance
        self.shares_held = 0

        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.balance = self.initial_balance
        self.shares_held = 0
        return self._next_observation(), {}

    def _next_observation(self):
        row = self.df.iloc[self.current_step]
        return np.array([
            row['close_norm'],
            row['tick_volume_norm'],
            row['rsi'],
            row['macd'],
            row['macd_signal'],
            row['atr'],
        ], dtype=np.float32)

    def step(self, action):
        current_price = self.df.iloc[self.current_step]['close']
        action = action[0]

        if action > 0:
            cost = self.balance * action
            shares_bought = cost / current_price
            self.shares_held += shares_bought
            self.balance -= cost
        elif action < 0:
            shares_sold = self.shares_held * abs(action)
            self.balance += shares_sold * current_price
            self.shares_held -= shares_sold

        self.current_step += 1
        done = self.current_step >= len(self.df) - 1

        next_obs = self._next_observation()
        next_price = self.df.iloc[self.current_step]['close'] if not done else current_price

        portfolio_value = self.balance + (self.shares_held * next_price)
        reward = (portfolio_value - self.initial_balance) / self.initial_balance

        return next_obs, reward, done, False, {}


class FinRLDataPreparer:
    def __init__(self, log_callback):
        self.log_callback = log_callback

    def scan_folder(self, folder):
        folder = Path(folder)
        symbols = []
        all_timeframes = set()
        for d in sorted(folder.iterdir()):
            symbol = d.name
            found = [tf for tf in Config.TIMEFRAMES if (d / f"{symbol}_{tf}.csv").exists()]
            if found:
                symbols.append((symbol, found))
                all_timeframes.update(found)
                self.log_callback(f"   {symbol}: {', '.join(found)}")
        return symbols, sorted(all_timeframes)

    def load_and_prepare(self, symbol, timeframe, folder):
        csv_file = Path(folder) / symbol / f"{symbol}_{timeframe}.csv"
        df = pd.read_csv(csv_file)

        df["time"] = pd.to_datetime(df["time"], unit="s") if pd.api.types.is_numeric_dtype(df["time"]) else pd.to_datetime(df["time"])
        df = df.sort_values("time").drop_duplicates("time").reset_index(drop=True)
        df = df.iloc[-Config.MAX_ROWS_PER_SYMBOL:].copy()

        df['returns'] = df['close'].pct_change()
        df['close_norm'] = df['close'] / df['close'].max()
        df['tick_volume_norm'] = df['tick_volume'] / df['tick_volume'].max()

        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss.replace(0, np.nan)
        df['rsi'] = 100 - (100 / (1 + rs))

        exp1 = df['close'].ewm(span=12, adjust=False).mean()
        exp2 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = exp1 - exp2
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()

        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        df['atr'] = true_range.rolling(14).mean()

        df = df.fillna(0.0)
        self.log_callback(f"   {symbol}_{timeframe}: {len(df)} steps prepared")
        return df


class StreamingFinRLTrainer:
    def __init__(self, log_callback, timeframe):
        self.log_callback = log_callback
        self.timeframe = timeframe
        self.model = None
        self.is_trained = False
        self.stop_flag = False
        self.training_history = {}
        self.log_callback(f"FinRL [{timeframe}] initialized")

    def stop_training(self):
        self.stop_flag = True
        self.log_callback("Training stop requested")

    def train_single_timeframe(self, df, timeframe, algo, timesteps, lr):
        self.stop_flag = False
        self.timeframe = timeframe

        env = TradingEnv(df)
        self.log_callback(f"[{timeframe}] Training {algo} for {timesteps} timesteps...")

        if algo == "PPO":
            self.model = PPO("MlpPolicy", env, learning_rate=lr, verbose=0, seed=Config.SEED, device="cpu")
        elif algo == "SAC":
            self.model = SAC("MlpPolicy", env, learning_rate=lr, verbose=0, seed=Config.SEED, device="cpu")
        else:
            raise ValueError(f"Unsupported algorithm: {algo}")

        self.model.learn(total_timesteps=timesteps)
        self.is_trained = not self.stop_flag

        self.training_history = {
            "timeframe": timeframe,
            "algorithm": algo,
            "timesteps": timesteps,
            "learning_rate": lr,
        }

    def save_model(self, model_dir, stem, timeframe, export_onnx):
        output_dir = Path(model_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        model_path = output_dir / f"{stem}_{timeframe}{Config.MODEL_EXT}"
        onnx_path = output_dir / f"{stem}_{timeframe}{Config.ONNX_EXT}"
        metadata_path = output_dir / f"{stem}_{timeframe}{Config.METADATA_EXT}"

        self.model.save(str(model_path))

        if export_onnx:
            import torch
            policy = self.model.policy
            obs_shape = policy.observation_space.shape
            dummy_obs = torch.zeros(1, *obs_shape, dtype=torch.float32)
            torch.onnx.export(
                policy,
                dummy_obs,
                str(onnx_path),
                input_names=["observation"],
                output_names=["action"],
                opset_version=Config.ONNX_OPSET,
                dynamo=False,
            )

        import pickle
        metadata = {
            "model_type": "stable_baselines3.PPO" if self.training_history["algorithm"] == "PPO" else "stable_baselines3.SAC",
            "timeframe": timeframe,
            "model_path": str(model_path),
            "onnx_path": str(onnx_path) if export_onnx else None,
            "onnx_exported": bool(export_onnx),
            "algorithm": self.training_history["algorithm"],
            "observation_shape": list(self.model.policy.observation_space.shape),
            "action_shape": list(self.model.policy.action_space.shape),
            "training_history": self.training_history,
            "training_time": datetime.now().isoformat(),
        }
        with open(metadata_path, "wb") as file:
            pickle.dump(metadata, file)

        self.log_callback(f"FinRL model saved: {model_path}")
        if export_onnx:
            self.log_callback(f"ONNX model saved: {onnx_path}")
        self.log_callback(f"Metadata saved: {metadata_path}")


class FinRLTrainerGUI(ctk.CTk):
    INI_PATH = Path(__file__).with_suffix(".ini")

    def __init__(self):
        ctk.set_appearance_mode(Config.THEME)
        ctk.set_default_color_theme(Config.COLOR_THEME)
        Config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
        Config.DATASET_DIR.mkdir(parents=True, exist_ok=True)

        super().__init__()
        self.title(Config.APP_TITLE)
        self.geometry(Config.APP_GEOMETRY)
        self.resizable(False, False)
        self._is_closing = False
        self.withdraw()

        self.gui_logger = GUILogger()
        self.gui_logger.enable_console_redirect()

        self.gpu_manager = GPUManager(log_callback=self.gui_logger.log)

        self.csv_folder = Config.DATASET_DIR
        self.model_folder = Config.MODEL_DIR
        self.is_training = False
        self.stop_flag = False
        self.current_trainer = None
        self.data_preparer = FinRLDataPreparer(self.gui_logger.log)

        self._create_ui()
        self._load_state()
        self.gui_logger.set_log_widget(self.log_text)

        self.gpu_manager.log_system_info()

        self.after(500, self.scan_data)
        self.protocol("WM_DELETE_WINDOW", self._on_closing)

        self.update_idletasks()
        self.deiconify()

    def _load_state(self):
        if not self.INI_PATH.exists():
            return
        cp = configparser.ConfigParser()
        cp.read(str(self.INI_PATH))

        if cp.has_section("window"):
            self.geometry(cp["window"]["geometry"])

        if cp.has_section("paths"):
            self.csv_var.set(cp["paths"]["csv_folder"])
            self.csv_folder = Path(cp["paths"]["csv_folder"])
            self.model_var.set(cp["paths"]["model_folder"])
            self.model_folder = Path(cp["paths"]["model_folder"])

        if cp.has_section("params"):
            self.algo.set(cp["params"]["algorithm"])
            self.timesteps.set(cp["params"]["timesteps"])
            self.lr.set(cp["params"]["learning_rate"])
            self.onnx_var.set(cp["params"].getboolean("onnx_export"))

        if cp.has_section("timeframes"):
            for tf in Config.TIMEFRAMES:
                self.tf_vars[tf].set(cp["timeframes"].getboolean(tf))

    def _save_state(self):
        cp = configparser.ConfigParser()
        cp["window"] = {"geometry": self.wm_geometry()}
        cp["paths"] = {"csv_folder": str(self.csv_var.get()), "model_folder": str(self.model_var.get())}
        cp["params"] = {
            "algorithm": self.algo.get(),
            "timesteps": self.timesteps.get(),
            "learning_rate": self.lr.get(),
            "onnx_export": str(self.onnx_var.get()),
        }
        cp["timeframes"] = {tf: str(var.get()) for tf, var in self.tf_vars.items()}

        with open(self.INI_PATH, "w") as f:
            cp.write(f)

    def _create_ui(self):
        main_frame = ctk.CTkFrame(self, fg_color="transparent")
        main_frame.pack(fill="both", expand=True, padx=10, pady=5)

        title_frame = ctk.CTkFrame(main_frame, corner_radius=6)
        title_frame.pack(fill="x", pady=(0, 4))

        ctk.CTkLabel(
            title_frame,
            text="Copyright Su Nie | BSD-3C License | https://github.com/can87683",
            font=("TkDefaultFont", 14),
            text_color="yellow",
        ).pack(pady=(0, 2))

        ctk.CTkLabel(
            title_frame,
            text="FinRL Model Trainer",
            font=("TkDefaultFont", 16, "bold"),
        ).pack(pady=4)

        self.usage_row = UsageRow(main_frame)
        self._create_csv(main_frame)
        self._create_model(main_frame)
        self._create_params(main_frame)
        self._create_tfs(main_frame)
        self._create_ctrls(main_frame)
        self._create_stat(main_frame)
        self._create_log(main_frame)

    def _create_csv(self, parent):
        frame = ctk.CTkFrame(parent, corner_radius=6)
        frame.pack(fill="x", pady=(0, 4))
        folder_frame = ctk.CTkFrame(frame, fg_color="transparent")
        folder_frame.pack(fill="x", padx=8, pady=3)
        ctk.CTkButton(folder_frame, text="CSV Folder", command=self.select_csv, width=120, height=28).pack(side="left", padx=(0, 6))
        self.csv_var = tk.StringVar(value=str(self.csv_folder))
        ctk.CTkEntry(folder_frame, textvariable=self.csv_var, state="readonly", height=28).pack(side="left", fill="x", expand=True)

    def _create_model(self, parent):
        frame = ctk.CTkFrame(parent, corner_radius=6)
        frame.pack(fill="x", pady=(0, 4))
        folder_frame = ctk.CTkFrame(frame, fg_color="transparent")
        folder_frame.pack(fill="x", padx=8, pady=3)
        ctk.CTkButton(folder_frame, text="Model Folder", command=self.select_model, width=120, height=28).pack(side="left", padx=(0, 6))
        self.model_var = tk.StringVar(value=str(self.model_folder))
        ctk.CTkEntry(folder_frame, textvariable=self.model_var, state="readonly", height=28).pack(side="left", fill="x", expand=True)
        self.stat_lbl = ctk.CTkLabel(frame, text="Status: Ready", font=("TkDefaultFont", 10))
        self.stat_lbl.pack(anchor="w", padx=8, pady=1)

    def _create_params(self, parent):
        frame = ctk.CTkFrame(parent, corner_radius=6)
        frame.pack(fill="x", pady=(0, 4))
        param_frame = ctk.CTkFrame(frame, fg_color="transparent")
        param_frame.pack(fill="x", padx=8, pady=3)

        self.algo = tk.StringVar(value=Config.DEFAULT_ALGO)
        self.timesteps = tk.StringVar(value=str(Config.DEFAULT_TIMESTEPS))
        self.lr = tk.StringVar(value=str(Config.DEFAULT_LR))

        row = ctk.CTkFrame(param_frame, fg_color="transparent")
        row.pack(fill="x", pady=1)
        ctk.CTkLabel(row, text="Algorithm:", width=80).grid(row=0, column=0, padx=(0, 3), sticky="w")
        ctk.CTkEntry(row, textvariable=self.algo, width=60).grid(row=0, column=1, padx=(0, 20), sticky="w")
        ctk.CTkLabel(row, text="Timesteps:", width=80).grid(row=0, column=2, padx=(0, 3), sticky="w")
        ctk.CTkEntry(row, textvariable=self.timesteps, width=80).grid(row=0, column=3, padx=(0, 20), sticky="w")
        ctk.CTkLabel(row, text="LR:", width=80).grid(row=0, column=4, padx=(0, 3), sticky="w")
        ctk.CTkEntry(row, textvariable=self.lr, width=80).grid(row=0, column=5, sticky="w")

    def _create_tfs(self, parent):
        frame = ctk.CTkFrame(parent, corner_radius=6)
        frame.pack(fill="x", pady=(0, 4))
        tf_frame = ctk.CTkFrame(frame, fg_color="transparent")
        tf_frame.pack(fill="x", padx=8, pady=3)
        ctk.CTkLabel(tf_frame, text="Timeframes (one model each):", font=("TkDefaultFont", 10)).pack(anchor="w", pady=(0, 2))
        self.tf_vars = {}
        checkboxes_frame = ctk.CTkFrame(tf_frame, fg_color="transparent")
        checkboxes_frame.pack(fill="x")
        for i, tf in enumerate(Config.TIMEFRAMES):
            var = tk.BooleanVar(value=True)
            self.tf_vars[tf] = var
            ctk.CTkCheckBox(checkboxes_frame, text=tf, variable=var, width=70, font=("TkDefaultFont", 10)).grid(row=0, column=i, padx=3, sticky="w")

    def _create_ctrls(self, parent):
        frame = ctk.CTkFrame(parent, corner_radius=6)
        frame.pack(fill="x", pady=(0, 4))
        button_frame = ctk.CTkFrame(frame, fg_color="transparent")
        button_frame.pack(pady=4)

        self.onnx_var = tk.BooleanVar(value=True)
        self.onnx_checkbox = ctk.CTkCheckBox(
            button_frame,
            text="ONNX Export",
            variable=self.onnx_var,
            font=("TkDefaultFont", 11),
            width=110,
        )
        self.onnx_checkbox.pack(side="left", padx=(0, 12))

        self.train_btn = ctk.CTkButton(
            button_frame, text="Start Training", command=self.start_train,
            width=130, height=34, fg_color="#2e7d32", hover_color="#1b5e20",
            font=("TkDefaultFont", 12, "bold"),
        )
        self.train_btn.pack(side="left", padx=4)
        self.stop_btn = ctk.CTkButton(
            button_frame, text="Stop Training", command=self.stop_train,
            width=130, height=34, fg_color="#c62828", hover_color="#b71c1c",
            font=("TkDefaultFont", 12, "bold"), state="disabled",
        )
        self.stop_btn.pack(side="left", padx=4)

    def _create_stat(self, parent):
        frame = ctk.CTkFrame(parent, corner_radius=6)
        frame.pack(fill="x", pady=(0, 4))
        self.stat_txt = ctk.CTkTextbox(frame, height=50, font=("Ubuntu Mono", 10))
        self.stat_txt.pack(fill="x", padx=8, pady=3)
        self.stat_txt.insert("1.0", "Ready")
        self.stat_txt.configure(state="disabled")

    def _create_log(self, parent):
        frame = ctk.CTkFrame(parent, corner_radius=6)
        frame.pack(fill="both", expand=True)
        self.log_text = ctk.CTkTextbox(frame, font=("Ubuntu Mono", 12))
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
            self.gui_logger.log(f"Model folder set to: {self.model_folder}")

    def scan_data(self):
        csv_folder = Path(self.csv_var.get())
        self.gui_logger.log(f"Scanning CSV folder: {csv_folder}")
        symbols, timeframes = self.data_preparer.scan_folder(csv_folder)
        self.stat_lbl.configure(text=f"Found {len(symbols)} symbols, {len(timeframes)} timeframes")
        self.gui_logger.log(f"Loaded {len(symbols)} symbols", "success")
        self.train_btn.configure(state="normal")

    def start_train(self):
        if self.is_training:
            self.gui_logger.log("Training already in progress", "warning")
            return

        csv_folder = Path(self.csv_var.get())
        selected_timeframes = [tf for tf, var in self.tf_vars.items() if var.get()]

        self.is_training = True
        self.stop_flag = False
        self.train_btn.configure(state="disabled")
        self.stop_btn.configure(state="normal")

        def training_task():
            self.gui_logger.log("=" * 60)
            self.gui_logger.log("Starting Per-Timeframe FinRL Training")
            self.gui_logger.log("=" * 60)
            self.gui_logger.log(f"CSV Folder: {csv_folder}")

            algo = self.algo.get()
            timesteps = int(self.timesteps.get())
            learning_rate = float(self.lr.get())
            model_folder = self.model_var.get()
            export_onnx = bool(self.onnx_var.get())
            self.gui_logger.log(f"ONNX export: {'enabled' if export_onnx else 'disabled'}")

            symbols, _ = self.data_preparer.scan_folder(csv_folder)
            results = {}

            for timeframe in selected_timeframes:
                if self.stop_flag or self._is_closing:
                    self.gui_logger.log("Training stopped by user", "warning")
                    break

                self.gui_logger.log("-" * 60)
                self.gui_logger.log(f"Timeframe: {timeframe}")
                self.update_stat(f"[{timeframe}] Preparing environment...")

                all_dfs = []
                for symbol, timeframes in symbols:
                    if timeframe not in timeframes:
                        continue
                    df = self.data_preparer.load_and_prepare(symbol, timeframe, csv_folder)
                    all_dfs.append(df)

                combined_df = pd.concat(all_dfs, ignore_index=True)

                self.update_stat(f"[{timeframe}] Training {algo} for {timesteps} steps...")
                trainer = StreamingFinRLTrainer(self.gui_logger.log, timeframe=timeframe)
                self.current_trainer = trainer

                trainer.train_single_timeframe(combined_df, timeframe, algo=algo, timesteps=timesteps, lr=learning_rate)

                if not trainer.stop_flag:
                    trainer.save_model(model_folder, Config.MODEL_PREFIX, timeframe, export_onnx)
                    results[timeframe] = "trained"
                else:
                    results[timeframe] = "stopped"
                    self.stop_flag = True

                del trainer
                self.current_trainer = None

            summary = ", ".join(f"{tf}: {status}" for tf, status in results.items())
            self.gui_logger.log("=" * 60)
            self.gui_logger.log(f"Summary: {summary}")
            self.update_stat(f"Done. {summary}")

            self.is_training = False
            self.current_trainer = None
            self.train_btn.configure(state="normal")
            self.stop_btn.configure(state="disabled")

        threading.Thread(target=training_task, daemon=True).start()

    def stop_train(self):
        self.stop_flag = True
        if self.current_trainer is not None:
            self.current_trainer.stop_training()
        self.gui_logger.log("Stopping training...", "warning")

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
                if messagebox.askyesno("Quit", "Do you want to quit AI Predict Train (FinRL)?"):
                    do_exit()
                else:
                    self._is_closing = False

        self.after(0, confirm_exit)


if __name__ == "__main__":
    app = FinRLTrainerGUI()
    app.mainloop()