#!/usr/bin/env python3
# ai_predict_train_finrl.py
# Copyright Su Nie | BSD-3C License | https://github.com/can87683

import setproctitle
setproctitle.setproctitle("ai_predict_train_finrl")

import customtkinter as ctk
import tkinter as tk
from tkinter import filedialog, messagebox
import threading
from pathlib import Path
import numpy as np
import pandas as pd
import psutil
import sys
from datetime import datetime
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO, SAC

import warnings
warnings.filterwarnings("ignore")

class Config:
    HOME = Path.home()
    MODEL_DIR = HOME / "ai_model_ml"
    DATASET_DIR = HOME / "mt5_data"
    MAX_ROWS_PER_SYMBOL = 20000
    DEFAULT_ALGO = "PPO"
    DEFAULT_TIMESTEPS = 50000
    DEFAULT_LR = 0.0003
    TIMEFRAMES = ['M5', 'M15', 'H1', 'H4', 'D1', 'W1']
    MODEL_PREFIX = "ai_predict_finrl"
    MODEL_EXT = ".zip"
    APP_TITLE = "🚀 FinRL Model Trainer"
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


class UsageRow:
    def __init__(self, parent):
        self.usage_frame = ctk.CTkFrame(parent, height=40, corner_radius=5)
        self.usage_frame.pack(fill="x", pady=(0, 10))
        self.usage_label = ctk.CTkLabel(self.usage_frame, text="Loading system usage...", font=("Ubuntu Mono", 20), text_color="Lime")
        self.usage_label.pack(expand=True, fill="x", padx=10, pady=5)
        self._update_usage()

    def _update_usage(self):
        cpu = psutil.cpu_percent(interval=0.5)
        dram = psutil.virtual_memory().percent
        gpu_percent = 0.0
        vram_percent = 0.0
        import GPUtil
        gpus = GPUtil.getGPUs()
        if gpus:
            gpu_percent = gpus[0].load * 100
            vram_percent = gpus[0].memoryUtil * 100
        self.usage_label.configure(text=f"CPU: {cpu:.1f}%  DRAM: {dram:.1f}%  GPU: {gpu_percent:.1f}%  VRAM: {vram_percent:.1f}%")
        self.usage_label.after(2000, self._update_usage)


class GUILogger:
    def __init__(self):
        self.messages = []
        self.log_widget = None
        self.ready = False
        self._out = sys.stdout

    def set_log_widget(self, w):
        self.log_widget = w
        self.ready = True
        w.configure(font=("Ubuntu Mono", 11))
        for m, lvl in self.messages:
            self._write(m, lvl)
        self.messages.clear()

    def _write(self, msg, lvl):
        if self.log_widget:
            self.log_widget.insert("end", f"[{datetime.now().strftime('%H:%M:%S')}] {msg}\n")
            self.log_widget.see("end")

    def log(self, msg, lvl="info"):
        c = f"[{datetime.now().strftime('%H:%M:%S')}] [{lvl.upper()}] {msg}"
        print(c, flush=True)
        if self.ready and self.log_widget:
            self._write(msg, lvl)
        else:
            self.messages.append((msg, lvl))


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
            row['volume_norm'],
            row['rsi'],
            row['macd'],
            row['macd_signal'],
            row['atr']
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
    def __init__(self, log_callback=None):
        self.log_callback = log_callback or print

    def _log(self, m):
        if self.log_callback:
            self.log_callback(m)

    def load_and_prepare(self, symbol, tf, folder):
        f = Path(folder) / symbol / f"{symbol}_{tf}.csv"
        if not f.exists():
            return None

        df = pd.read_csv(f)
        if "time" not in df.columns or "close" not in df.columns:
            return None

        df = df.sort_values("time").drop_duplicates("time").reset_index(drop=True)
        df = df.iloc[-Config.MAX_ROWS_PER_SYMBOL:].copy()

        df['returns'] = df['close'].pct_change()
        df['close_norm'] = df['close'] / df['close'].max()
        df['volume_norm'] = df['volume'] / df['volume'].max()

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
        self._log(f"   📊 {symbol}_{tf}: {len(df)} steps prepared")
        return df


class StreamingFinRLTrainer:
    def __init__(self, log_callback=None, timeframe=None):
        self.log_callback = log_callback or print
        self.timeframe = timeframe
        self.model = None
        self.is_trained = False
        self.stop_flag = False
        self._log(f"🚀 FinRL [{timeframe}] initialized")

    def _log(self, m):
        if self.log_callback:
            self.log_callback(m)

    def stop_training(self):
        self.stop_flag = True
        self._log("⏹️ Stop requested")

    def train_single_timeframe(self, df, timeframe, algo="PPO", timesteps=50000, lr=0.0003):
        self.stop_flag = False
        self.timeframe = timeframe

        env = TradingEnv(df)

        self._log(f"📊 [{timeframe}] Training {algo} for {timesteps} timesteps...")

        if algo == "PPO":
            self.model = PPO("MlpPolicy", env, learning_rate=lr, verbose=0)
        else:
            self.model = SAC("MlpPolicy", env, learning_rate=lr, verbose=0)

        self.model.learn(total_timesteps=timesteps)
        self.is_trained = not self.stop_flag
        return self.is_trained

    def save_model(self, model_dir, stem, timeframe):
        if not self.is_trained:
            return False
        out = Path(model_dir)
        out.mkdir(parents=True, exist_ok=True)
        mp = out / f"{stem}_{timeframe}{Config.MODEL_EXT}"

        self.model.save(str(mp))
        self._log(f"✅ Model saved: {mp}")
        return True


class FinRLTrainerGUI(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title(Config.APP_TITLE)
        self.geometry(Config.APP_GEOMETRY)
        self.resizable(False, False)
        self.gui_logger = GUILogger()
        self.csv_folder = Config.DATASET_DIR
        self.model_folder = Config.MODEL_DIR
        self.is_training = False
        self.stop_flag = False
        self.current_trainer = None
        self.data_preparer = FinRLDataPreparer(self.gui_logger.log)
        self._create_ui()
        self.gui_logger.set_log_widget(self.log_text)
        self.after(500, self.scan_data)
        self.protocol("WM_DELETE_WINDOW", self._on_closing)

    def _create_ui(self):
        m = ctk.CTkFrame(self, fg_color="transparent")
        m.pack(fill="both", expand=True, padx=10, pady=5)
        t = ctk.CTkFrame(m, corner_radius=6)
        t.pack(fill="x", pady=(0, 4))

        ctk.CTkLabel(t, text="Copyright Su Nie | BSD-3C License | https://github.com/can87683", font=("Ubuntu", 16), text_color="yellow").pack(pady=(0, 4))

        ctk.CTkLabel(t, text="🎯 FinRL Model Trainer", font=("Ubuntu", 20, "bold")).pack(pady=4)
        UsageRow(m)

        self._create_csv(m)
        self._create_model(m)
        self._create_params(m)
        self._create_tfs(m)
        self._create_ctrls(m)
        self._create_stat(m)
        self._create_log(m)

    def _create_csv(self, p):
        f = ctk.CTkFrame(p, corner_radius=6)
        f.pack(fill="x", pady=(0, 4))
        ff = ctk.CTkFrame(f, fg_color="transparent")
        ff.pack(fill="x", padx=8, pady=3)
        ctk.CTkButton(ff, text="📁 CSV Folder", command=self.select_csv, width=120, height=28).pack(side="left", padx=(0, 6))
        self.csv_var = tk.StringVar(value=str(self.csv_folder))
        ctk.CTkEntry(ff, textvariable=self.csv_var, state="readonly", height=28).pack(side="left", fill="x", expand=True)

    def _create_model(self, p):
        f = ctk.CTkFrame(p, corner_radius=6)
        f.pack(fill="x", pady=(0, 4))
        ff = ctk.CTkFrame(f, fg_color="transparent")
        ff.pack(fill="x", padx=8, pady=3)
        ctk.CTkButton(ff, text="📁 Model Folder", command=self.select_model, width=120, height=28).pack(side="left", padx=(0, 6))
        self.model_var = tk.StringVar(value=str(self.model_folder))
        ctk.CTkEntry(ff, textvariable=self.model_var, state="readonly", height=28).pack(side="left", fill="x", expand=True)
        self.stat_lbl = ctk.CTkLabel(f, text="📊 Status: Ready", font=("Ubuntu", 10))
        self.stat_lbl.pack(anchor="w", padx=8, pady=1)

    def _create_params(self, p):
        f = ctk.CTkFrame(p, corner_radius=6)
        f.pack(fill="x", pady=(0, 4))
        pf = ctk.CTkFrame(f, fg_color="transparent")
        pf.pack(fill="x", padx=8, pady=3)
        self.algo = tk.StringVar(value=Config.DEFAULT_ALGO)
        self.timesteps = tk.StringVar(value=str(Config.DEFAULT_TIMESTEPS))
        self.lr = tk.StringVar(value=str(Config.DEFAULT_LR))
        r = ctk.CTkFrame(pf, fg_color="transparent")
        r.pack(fill="x", pady=1)
        ctk.CTkLabel(r, text="Algorithm:", width=80).grid(row=0, column=0, padx=(0, 3), sticky="w")
        ctk.CTkEntry(r, textvariable=self.algo, width=60).grid(row=0, column=1, padx=(0, 20), sticky="w")
        ctk.CTkLabel(r, text="Timesteps:", width=80).grid(row=0, column=2, padx=(0, 3), sticky="w")
        ctk.CTkEntry(r, textvariable=self.timesteps, width=80).grid(row=0, column=3, padx=(0, 20), sticky="w")
        ctk.CTkLabel(r, text="LR:", width=80).grid(row=0, column=4, padx=(0, 3), sticky="w")
        ctk.CTkEntry(r, textvariable=self.lr, width=80).grid(row=0, column=5, sticky="w")

    def _create_tfs(self, p):
        f = ctk.CTkFrame(p, corner_radius=6)
        f.pack(fill="x", pady=(0, 4))
        tf = ctk.CTkFrame(f, fg_color="transparent")
        tf.pack(fill="x", padx=8, pady=3)
        ctk.CTkLabel(tf, text="Timeframes:", font=("Ubuntu", 10)).pack(anchor="w", pady=(0, 2))
        self.tf_vars = {}
        cf = ctk.CTkFrame(tf, fg_color="transparent")
        cf.pack(fill="x")
        for i, t in enumerate(Config.TIMEFRAMES):
            v = tk.BooleanVar(value=True)
            self.tf_vars[t] = v
            ctk.CTkCheckBox(cf, text=t, variable=v, width=70, font=("Ubuntu", 10)).grid(row=0, column=i, padx=3, sticky="w")

    def _create_ctrls(self, p):
        f = ctk.CTkFrame(p, corner_radius=6)
        f.pack(fill="x", pady=(0, 4))
        bf = ctk.CTkFrame(f, fg_color="transparent")
        bf.pack(pady=4)
        self.train_btn = ctk.CTkButton(bf, text="🚀 Start Training", command=self.start_train, width=130, height=34, fg_color="#2e7d32", hover_color="#1b5e20", font=("Ubuntu", 12, "bold"))
        self.train_btn.pack(side="left", padx=4)
        self.stop_btn = ctk.CTkButton(bf, text="⏹️ Stop Training", command=self.stop_train, width=130, height=34, fg_color="#c62828", hover_color="#b71c1c", font=("Ubuntu", 12, "bold"), state="disabled")
        self.stop_btn.pack(side="left", padx=4)

    def _create_stat(self, p):
        f = ctk.CTkFrame(p, corner_radius=6)
        f.pack(fill="x", pady=(0, 4))
        self.stat_txt = ctk.CTkTextbox(f, height=50, font=("Ubuntu Mono", 10))
        self.stat_txt.pack(fill="x", padx=8, pady=3)
        self.stat_txt.insert("1.0", "Ready")
        self.stat_txt.configure(state="disabled")

    def _create_log(self, p):
        f = ctk.CTkFrame(p, corner_radius=6)
        f.pack(fill="both", expand=True)
        self.log_text = ctk.CTkTextbox(f, font=("Ubuntu Mono", 10))
        self.log_text.pack(fill="both", expand=True, padx=8, pady=3)

    def update_stat(self, m):
        self.stat_txt.configure(state="normal")
        self.stat_txt.delete("1.0", "end")
        self.stat_txt.insert("1.0", m)
        self.stat_txt.configure(state="disabled")

    def select_csv(self):
        f = filedialog.askdirectory(initialdir=str(self.csv_folder))
        if f:
            self.csv_var.set(f)
            self.scan_data()

    def select_model(self):
        f = filedialog.askdirectory(initialdir=str(self.model_folder))
        if f:
            self.model_var.set(f)
            global MODEL_DIR
            MODEL_DIR = Path(f)
            MODEL_DIR.mkdir(parents=True, exist_ok=True)

    def scan_data(self):
        cf = Path(self.csv_var.get())
        if not cf.exists():
            self.stat_lbl.configure(text="❌ Folder not found")
            self.train_btn.configure(state="disabled")
            return
        self.stat_lbl.configure(text="✅ Data URI configured")
        self.train_btn.configure(state="normal")

    def start_train(self):
        if self.is_training:
            return
        cf = Path(self.csv_var.get())
        stf = [t for t, v in self.tf_vars.items() if v.get()]
        if not stf:
            return
        self.is_training = True
        self.stop_flag = False
        self.train_btn.configure(state="disabled")
        self.stop_btn.configure(state="normal")

        def task():
            self.gui_logger.log("=" * 60)
            self.gui_logger.log("🚀 Starting FinRL Training Pipeline")

            algo = self.algo.get()
            timesteps = int(self.timesteps.get())
            lr = float(self.lr.get())
            mf = self.model_var.get()
            res = {}

            syms = [d.name for d in cf.iterdir() if d.is_dir()]

            for tf in stf:
                if self.stop_flag:
                    break
                self.update_stat(f"[{tf}] Preparing environment...")

                all_dfs = []
                for s in syms:
                    df = self.data_preparer.load_and_prepare(s, tf, cf)
                    if df is not None:
                        all_dfs.append(df)

                if not all_dfs:
                    res[tf] = "skipped (no data)"
                    continue

                combined_df = pd.concat(all_dfs, ignore_index=True)

                self.update_stat(f"[{tf}] Training {algo}...")
                tr = StreamingFinRLTrainer(self.gui_logger.log, timeframe=tf)
                self.current_trainer = tr

                ok = tr.train_single_timeframe(combined_df, tf, algo=algo, timesteps=timesteps, lr=lr)

                if ok and not tr.stop_flag:
                    stem = Config.MODEL_PREFIX
                    if tr.save_model(mf, stem, tf):
                        res[tf] = "trained"
                    else:
                        res[tf] = "save failed"
                else:
                    res[tf] = "stopped" if tr.stop_flag else "failed"

                del tr
                self.current_trainer = None

            self.gui_logger.log(f"🏁 Summary: {', '.join(f'{k}: {v}' for k, v in res.items())}")
            self.update_stat("Done.")
            self.is_training = False
            self.current_trainer = None
            self.train_btn.configure(state="normal")
            self.stop_btn.configure(state="disabled")

        threading.Thread(target=task, daemon=True).start()

    def stop_train(self):
        self.stop_flag = True
        if self.current_trainer:
            self.current_trainer.stop_training()

    def _on_closing(self):
        if self.is_training:
            if messagebox.askyesno("Training", "Stop and quit?"):
                self.stop_flag = True
                if self.current_trainer:
                    self.current_trainer.stop_training()
                self.after(100, self.destroy)
        else:
            if messagebox.askyesno("Quit", "Quit?"):
                self.destroy()


def main():
    FinRLTrainerGUI().mainloop()

if __name__ == "__main__":
    main()