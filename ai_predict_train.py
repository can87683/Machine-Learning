#!/usr/bin/env python3
# ai_predict_train.py
# Copyright Su Nie | BSD-3C License | https://github.com/can87683

import setproctitle
setproctitle.setproctitle("ai_predict_train.py")

import os
import sys
import threading
import customtkinter as ctk
import tkinter as tk
from tkinter import messagebox, filedialog, StringVar
import configparser
import socket
import urllib.request
import subprocess
import shlex
import psutil
import GPUtil
import shutil


class SystemConfig:
    def __init__(self):
        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["MKL_NUM_THREADS"] = "1"
        os.environ["NUMEXPR_NUM_THREADS"] = "1"
        os.environ["KMP_AFFINITY"] = "disabled"
        os.environ["KMP_SETTINGS"] = "0"
        os.environ["KMP_WARNINGS"] = "FALSE"

        if self.is_wine():
            print("🍷 Wine environment detected - applying threading optimizations")
            os.environ["OPENBLAS_NUM_THREADS"] = "1"
            os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

    def is_wine(self) -> bool:
        return os.environ.get('WINELOADERNOEXEC') is not None

    def is_windows(self) -> bool:
        return sys.platform.startswith('win')


class Config:
    def __init__(self):
        self.title = "AI Predict Trainers"
        self.window_width = 640
        self.window_height = 300
        self.IPROW_FONT_SIZE = 26
        self.USAGE_FONT_SIZE = 20
        self.LIST_FONT_SIZE = 12
        self.file = "ai_predict_train.ini"
        self.parser = configparser.ConfigParser()
        if os.path.exists(self.file):
            self.parser.read(self.file)
        else:
            self.parser.read_dict(self._default_settings())

        if "paths" not in self.parser:
            self.parser["paths"] = {}
        if "python" not in self.parser:
            self.parser["python"] = {"binary_path": sys.executable}

    def _default_settings(self):
        return {
            "window": {"width": str(self.window_width), "height": str(self.window_height), "x": "100", "y": "100"},
            "python": {"binary_path": sys.executable},
            "colors": {
                "frame1_bg": "grey", "frame1_fg": "yellow",
                "frame2_bg": "pink", "frame2_fg": "blue",
                "frame3_bg": "yellow", "frame3_fg": "black",
                "frame4_bg": "lime", "frame4_fg": "red",
                "frame5_bg": "green", "frame5_fg": "yellow",
                "border_color": "grey",
            },
            "paths": {}
        }

    def save(self):
        with open(self.file, "w") as f:
            self.parser.write(f)


class IPRow:
    def __init__(self, parent, config):
        self.frame = ctk.CTkFrame(parent, fg_color="#222222", height=40)
        self.frame.pack(fill="x")
        self.label = ctk.CTkLabel(
            self.frame, text="LAN: --.--.--.--  WAN: --.--.--.--",
            font=("Arial", config.IPROW_FONT_SIZE),
            fg_color="blue", text_color="yellow", anchor="center"
        )
        self.label.pack(expand=True, fill="x")
        self.result_queue = queue.Queue()
        self.frame.after(100, self.check_queue)
        self.frame.after(1000, self.update_ip_label)

    def get_lan_ip(self):
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.settimeout(0.5)
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            s.close()
            return ip
        except OSError:
            return "--.--.--.--"

    def get_public_ip(self):
        try:
            req = urllib.request.Request("https://api.ipify.org", headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=1.5) as response:
                return response.read().decode("utf-8").strip()
        except Exception:
            return "--.--.--.--"

    def update_ip_label(self):
        def fetch():
            try:
                lan = self.get_lan_ip()
                wan = self.get_public_ip()
                self.result_queue.put((lan, wan))
            except Exception:
                self.result_queue.put(("--.--.--.--", "--.--.--.--"))
        threading.Thread(target=fetch, daemon=True).start()
        self.frame.after(60000, self.update_ip_label)

    def check_queue(self):
        try:
            while True:
                lan, wan = self.result_queue.get_nowait()
                self.label.configure(text=f"LAN: {lan}   WAN: {wan}")
        except queue.Empty:
            pass
        self.frame.after(100, self.check_queue)


class UsageRow:
    def __init__(self, parent, config):
        self.frame = ctk.CTkFrame(parent, fg_color="grey", height=40)
        self.frame.pack(fill="x")
        self.label = ctk.CTkLabel(
            self.frame, text="Loading system usage...", font=("Arial", config.USAGE_FONT_SIZE),
            fg_color="yellow", text_color="blue", anchor="center"
        )
        self.label.pack(expand=True, fill="x")
        self.update_usage()

    def update_usage(self):
        def fetch():
            cpu = psutil.cpu_percent(interval=0.5)
            dram = psutil.virtual_memory().percent
            gpu_percent = 0.0
            vram_percent = 0.0
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu = gpus[0]
                gpu_percent = gpu.load * 100
                vram_percent = gpu.memoryUtil * 100

            text = f"CPU: {cpu:.1f}%   DRAM: {dram:.1f}%   GPU: {gpu_percent:.1f}%   VRAM: {vram_percent:.1f}%"
            self.frame.after(0, lambda: self.label.configure(text=text))
            self.frame.after(2000, self.update_usage)
        threading.Thread(target=fetch, daemon=True).start()


class AIMenuGUI:
    def __init__(self):
        self.config = Config()
        self.path_entries = []

        width = int(self.config.parser["window"].get("width", self.config.window_width))
        height = int(self.config.parser["window"].get("height", self.config.window_height))
        pos_x = int(self.config.parser["window"].get("x", 100))
        pos_y = int(self.config.parser["window"].get("y", 100))
        self.python_binary = self.config.parser["python"]["binary_path"]

        self.root = ctk.CTk()
        self.root.title(self.config.title)
        self.root.geometry(f"{width}x{height}+{pos_x}+{pos_y}")
        self.root.resizable(False, False)
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        self.ip_row = IPRow(self.root, self.config)
        self.usage_row = UsageRow(self.root, self.config)
        self.build_add_bar()
        self.build_path_container()
        self.load_paths_from_config()

    def build_add_bar(self):
        add_bar = ctk.CTkFrame(self.root, fg_color=self.config.parser["colors"]["frame5_bg"])
        add_bar.pack(fill="x")

        self.python_btn = ctk.CTkButton(
            add_bar, text="Python Path", font=("Arial", 10, "bold"),
            fg_color="orange", text_color="black", command=self.set_python_path
        )
        self.python_btn.pack(side="left", padx=2, pady=2)

        self.python_label = ctk.CTkLabel(
            add_bar, text=os.path.basename(self.python_binary),
            font=("Arial", 10), fg_color=self.config.parser["colors"]["frame5_bg"],
            text_color="white", anchor="w"
        )
        self.python_label.pack(side="left", padx=2, pady=2)

        ctk.CTkLabel(add_bar, text="  ", font=("Arial", 12), fg_color=self.config.parser["colors"]["frame5_bg"], text_color="white").pack(side="left", padx=2, pady=2)

        self.add_btn = ctk.CTkButton(
            add_bar, text="+ Add Path", font=("Arial", 12),
            command=self.add_path_entry
        )
        self.add_btn.pack(side="left", padx=2, pady=2)

    def build_path_container(self):
        self.frame5_container = ctk.CTkFrame(
            self.root,
            fg_color=self.config.parser["colors"]["frame5_bg"],
            border_color=self.config.parser["colors"]["border_color"],
            border_width=2,
            width=int(self.config.parser["window"].get("width", self.config.window_width)),
            height=650
        )
        self.frame5_container.pack(fill="both", expand=True)

        self.canvas = tk.Canvas(
            self.frame5_container,
            bg=self.config.parser["colors"]["frame5_bg"],
            highlightthickness=0
        )
        self.scrollable_frame = tk.Frame(self.canvas, bg=self.config.parser["colors"]["frame5_bg"])

        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )

        self._canvas_window = self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.bind("<Configure>", self._on_canvas_configure)
        self.canvas.pack(side="left", fill="both", expand=True)

        self.scrollable_frame.grid_columnconfigure(0, weight=1, uniform="paths")
        self.scrollable_frame.grid_columnconfigure(1, weight=1, uniform="paths")

        self.root.bind_all("<MouseWheel>", self._on_mousewheel)
        self.root.bind_all("<Button-4>", self._on_mousewheel)
        self.root.bind_all("<Button-5>", self._on_mousewheel)

    def _on_canvas_configure(self, event):
        self.canvas.itemconfig(self._canvas_window, width=event.width)

    def _on_mousewheel(self, event):
        if event.delta > 0 or event.num == 4:
            self.canvas.yview_scroll(-1, "units")
        else:
            self.canvas.yview_scroll(1, "units")

    def set_python_path(self):
        filename = filedialog.askopenfilename(
            title="Select Python 3.10 Binary",
            filetypes=[("Python Binaries", "python3*"), ("All Executables", "*")]
        )
        result = subprocess.run([filename, "--version"], capture_output=True, text=True, timeout=5)
        if result.returncode == 0 and "Python 3.10" in result.stdout:
            self.config.parser["python"]["binary_path"] = filename
            self.python_binary = filename
            self.python_label.configure(text=os.path.basename(filename))
            self.config.save()
            messagebox.showinfo("Success", f"Python binary set:\n{filename}")
        else:
            messagebox.showwarning("Invalid Python", f"Not Python 3.10:\n{filename}")

    def run_path(self, path: str):
        path = path.strip()
        parts = shlex.split(path)
        executable = parts[0]
        args = parts[1:] if len(parts) > 1 else []
        cwd = None

        if executable.lower() in ['python', 'python3', 'python.exe', 'python3.exe'] and len(args) > 0:
            script_path = args[0]
            cwd = os.path.dirname(os.path.abspath(script_path))
            subprocess.Popen([self.python_binary] + args, cwd=cwd, env=os.environ.copy())
        elif executable in ['bash', 'sh']:
            if len(args) > 0:
                script_path = args[0]
                cwd = os.path.dirname(os.path.abspath(script_path))
            subprocess.Popen([executable] + args, cwd=cwd, env=os.environ.copy())
        else:
            if not os.path.exists(executable) and not os.path.isabs(executable):
                executable = shutil.which(executable)
            cwd = os.path.dirname(os.path.abspath(executable))
            if executable.endswith('.py'):
                cmd = [self.python_binary, executable] + args
            elif executable.endswith('.sh'):
                cmd = ['bash', executable] + args
            else:
                cmd = [executable] + args
            subprocess.Popen(cmd, cwd=cwd, env=os.environ.copy())

    def remove_path_entry(self, idx):
        confirm = messagebox.askyesno("Remove Path", "Are you sure you want to remove this path entry?")
        if not confirm:
            return
        self.path_entries[idx]['frame'].destroy()
        self.path_entries.pop(idx)
        self.refresh_path_layout()
        self.save_paths_to_config()

    def refresh_path_layout(self):
        for i, entry_data in enumerate(self.path_entries):
            col = i % 2
            row = i // 2
            entry_data['frame'].grid(row=row, column=col, padx=2, pady=2, sticky="nsew")
            entry_data['remove_btn'].configure(command=lambda idx=i: self.remove_path_entry(idx))

    def add_path_entry(self, path="", save_immediately=True):
        row_idx = len(self.path_entries)
        col = row_idx % 2
        row = row_idx // 2

        entry_frame = ctk.CTkFrame(self.scrollable_frame, fg_color=self.config.parser["colors"]["frame5_bg"])
        entry_frame.grid(row=row, column=col, padx=2, pady=2, sticky="nsew")
        entry_frame.grid_columnconfigure(0, weight=1)
        entry_frame.grid_columnconfigure(1, weight=0)
        entry_frame.grid_columnconfigure(2, weight=0)
        entry_frame.grid_columnconfigure(3, weight=0)

        full_path = path
        script_name = os.path.basename(full_path) if full_path else "[New]"

        name_label = ctk.CTkLabel(
            entry_frame, text=script_name,
            fg_color=self.config.parser["colors"]["frame5_bg"],
            text_color=self.config.parser["colors"]["frame5_fg"],
            anchor="w", font=("Courier", self.config.LIST_FONT_SIZE, "bold")
        )
        name_label.grid(row=0, column=0, sticky="ew", padx=2, pady=2)

        def open_browser():
            nonlocal full_path
            filename = filedialog.askopenfilename(
                title="Select Script or Binary",
                filetypes=[("Python Scripts", "*.py"), ("Executables", "*.exe"), ("Shell Scripts", "*.sh"), ("All Files", "*.*")]
            )
            if filename:
                full_path = filename
                name_label.configure(text=os.path.basename(filename))
                self.save_paths_to_config()

        btn_width = 24

        browse_btn = ctk.CTkButton(entry_frame, text="Br", width=btn_width, command=open_browser)
        browse_btn.grid(row=0, column=1, padx=2, pady=2)

        run_btn = ctk.CTkButton(entry_frame, text="R", width=btn_width, fg_color="green", text_color="white", command=lambda: self.run_path(full_path))
        run_btn.grid(row=0, column=2, padx=2, pady=2)

        remove_btn = ctk.CTkButton(entry_frame, text="X", width=btn_width, fg_color="red", text_color="white", command=lambda idx=row_idx: self.remove_path_entry(idx))
        remove_btn.grid(row=0, column=3, padx=2, pady=2)

        self.path_entries.append({
            'frame': entry_frame,
            'full_path': full_path,
            'label': name_label,
            'browse_btn': browse_btn,
            'run_btn': run_btn,
            'remove_btn': remove_btn
        })

        if save_immediately:
            self.save_paths_to_config()

    def save_paths_to_config(self):
        self.config.parser["paths"] = {}
        for i, entry_data in enumerate(self.path_entries):
            path = entry_data['full_path'].strip()
            if path:
                self.config.parser["paths"][f"path_{i}"] = path
        self.config.save()

    def load_paths_from_config(self):
        if "paths" in self.config.parser:
            paths_dict = dict(self.config.parser["paths"])
            sorted_keys = sorted(
                paths_dict.keys(),
                key=lambda x: int(x.split('_')[1]) if '_' in x and x.split('_')[1].isdigit() else 0
            )
            for key in sorted_keys:
                path = paths_dict[key]
                if path and path.strip():
                    self.add_path_entry(path, save_immediately=False)
            self.save_paths_to_config()

    def on_closing(self):
        if not messagebox.askyesno("Exit", "Are you sure you want to exit?"):
            return
        geom = self.root.geometry()
        size_part, pos_part = geom.split("+", 1)
        w, h = size_part.split("x")
        x, y = pos_part.split("+")
        self.config.parser["window"]["width"] = w
        self.config.parser["window"]["height"] = h
        self.config.parser["window"]["x"] = x
        self.config.parser["window"]["y"] = y
        self.config.save()
        self.root.destroy()

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    SystemConfig()
    app = AIMenuGUI()
    app.run()