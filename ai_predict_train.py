#!/usr/bin/env python3
# ai_predict_train.py
"""
pip requirements (A=Z): customtkinter, GPUtil, psutil, setproctitle
"""

import setproctitle
import os
import sys
setproctitle.setproctitle(os.path.basename(sys.argv[0]))

import threading
import queue
import signal
import time
import shlex
import shutil
import socket
import subprocess
import configparser
import urllib.request
import tkinter as tk
from tkinter import messagebox, filedialog
import customtkinter as ctk
import psutil
import GPUtil


class SystemConfig:
    def __init__(self):
        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["MKL_NUM_THREADS"] = "1"
        os.environ["NUMEXPR_NUM_THREADS"] = "1"
        os.environ["KMP_AFFINITY"] = "disabled"
        os.environ["KMP_SETTINGS"] = "0"
        os.environ["KMP_WARNINGS"] = "FALSE"
        os.environ["OPENBLAS_NUM_THREADS"] = "1"
        os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

    def is_wine(self) -> bool:
        return os.environ.get('WINELOADERNOEXEC') is not None

    def is_windows(self) -> bool:
        return sys.platform.startswith('win')


class IPRow:
    def __init__(self, parent, gui):
        self.gui = gui
        self.frame = ctk.CTkFrame(parent, fg_color="#222222", height=40)
        self.frame.pack(fill="x")
        self.label = ctk.CTkLabel(
            self.frame, text="LAN: --.--.--.--  WAN: --.--.--.--",
            font=("Arial", gui.IPROW_FONT_SIZE),
            fg_color="blue", text_color="yellow", anchor="center"
        )
        self.label.pack(expand=True, fill="x")
        self.result_queue = queue.Queue()
        self.fetch_in_progress = False
        self.frame.after(gui.QUEUE_POLL_MS, self.check_queue)
        self.frame.after(1000, self.update_ip_label)

    def get_lan_ip(self):
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.settimeout(0.5)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip

    def get_public_ip(self):
        req = urllib.request.Request("https://api.ipify.org", headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=1.5) as response:
            return response.read().decode("utf-8").strip()

    def update_ip_label(self):
        if self.fetch_in_progress:
            self.frame.after(self.gui.REFRESH_IP_MS, self.update_ip_label)
            return

        def fetch():
            self.fetch_in_progress = True
            lan = self.get_lan_ip()
            wan = self.get_public_ip()
            self.result_queue.put((lan, wan))
            self.fetch_in_progress = False
        threading.Thread(target=fetch, daemon=True).start()
        self.frame.after(self.gui.REFRESH_IP_MS, self.update_ip_label)

    def check_queue(self):
        try:
            while True:
                lan, wan = self.result_queue.get_nowait()
                self.label.configure(text=f"LAN: {lan}   WAN: {wan}")
        except queue.Empty:
            pass
        self.frame.after(self.gui.QUEUE_POLL_MS, self.check_queue)


class UsageRow:
    def __init__(self, parent, gui):
        self.gui = gui
        self.frame = ctk.CTkFrame(parent, fg_color="grey", height=40)
        self.frame.pack(fill="x")
        self.label = ctk.CTkLabel(
            self.frame, text="Loading system usage...",
            font=("Arial", gui.USAGE_FONT_SIZE),
            fg_color="yellow", text_color="blue", anchor="center"
        )
        self.label.pack(expand=True, fill="x")
        self.result_queue = queue.Queue()
        self.fetch_in_progress = False
        self.frame.after(gui.QUEUE_POLL_MS, self.check_queue)
        self.frame.after(1000, self.update_usage)

    def update_usage(self):
        if self.fetch_in_progress:
            self.frame.after(self.gui.REFRESH_USAGE_MS, self.update_usage)
            return

        def fetch():
            self.fetch_in_progress = True
            cpu = psutil.cpu_percent(interval=self.gui.USAGE_INTERVAL_S)
            dram = psutil.virtual_memory().percent
            gpu_percent = 0.0
            vram_percent = 0.0
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu = gpus[0]
                gpu_percent = gpu.load * 100
                vram_percent = gpu.memoryUtil * 100
            text = f"CPU: {cpu:.1f}%   DRAM: {dram:.1f}%   GPU: {gpu_percent:.1f}%   VRAM: {vram_percent:.1f}%"
            self.result_queue.put(text)
            self.fetch_in_progress = False
        threading.Thread(target=fetch, daemon=True).start()
        self.frame.after(self.gui.REFRESH_USAGE_MS, self.update_usage)

    def check_queue(self):
        try:
            while True:
                text = self.result_queue.get_nowait()
                self.label.configure(text=text)
        except queue.Empty:
            pass
        self.frame.after(self.gui.QUEUE_POLL_MS, self.check_queue)


class AIMenuGUI:
    title = "Copyright Su Nie | BSD-3C License | https://github.com/can87683"
    window_width = 640
    window_height = 300
    IPROW_FONT_SIZE = 26
    USAGE_FONT_SIZE = 20
    LIST_FONT_SIZE = 12
    BUTTON_WIDTH = 24
    REFRESH_IP_MS = 60000
    REFRESH_USAGE_MS = 2000
    QUEUE_POLL_MS = 100
    USAGE_INTERVAL_S = 0.5
    KILL_GRACE_TICKS = 30
    KILL_GRACE_SLEEP = 0.1
    max_entries = 500

    def __init__(self):
        # Derive ini filename from the script's actual name
        script_path = os.path.abspath(sys.argv[0])
        script_dir = os.path.dirname(script_path)
        script_name = os.path.splitext(os.path.basename(script_path))[0]
        self.file = os.path.join(script_dir, f"{script_name}.ini")

        self.parser = configparser.ConfigParser()
        self.parser.read_dict(self._default_settings())
        self.parser.read(self.file)

        if "paths" not in self.parser:
            self.parser["paths"] = {}
        if "python" not in self.parser:
            self.parser["python"] = {"binary_path": sys.executable}

        self.path_entries = []

        pos_x = int(self.parser["window"].get("x", 100))
        pos_y = int(self.parser["window"].get("y", 100))
        self.python_binary = self.parser["python"]["binary_path"]

        self.root = ctk.CTk()
        self.root.title(self.title)
        self.root.geometry(f"{self.window_width}x{self.window_height}+{pos_x}+{pos_y}")
        self.root.resizable(False, False)
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        self.ip_row = IPRow(self.root, self)
        self.usage_row = UsageRow(self.root, self)
        self.build_add_bar()
        self.build_path_container()
        self.load_paths_from_config()

    def _default_settings(self):
        return {
            "window": {
                "x": "100",
                "y": "100",
            },
            "python": {"binary_path": sys.executable},
            "colors": {
                "frame1_bg": "grey", "frame1_fg": "yellow",
                "frame2_bg": "pink", "frame2_fg": "blue",
                "frame3_bg": "yellow", "frame3_fg": "black",
                "frame4_bg": "lime", "frame4_fg": "red",
                "frame5_bg": "green", "frame5_fg": "yellow",
                "border_color": "grey",
            },
            "paths": {},
        }

    def save(self):
        with open(self.file, "w") as f:
            self.parser.write(f)

    def build_add_bar(self):
        add_bar = ctk.CTkFrame(self.root, fg_color=self.parser["colors"]["frame5_bg"])
        add_bar.pack(fill="x")

        self.python_btn = ctk.CTkButton(
            add_bar, text="Python Path", font=("Arial", 10, "bold"),
            fg_color="orange", text_color="black", command=self.set_python_path
        )
        self.python_btn.pack(side="left", padx=1, pady=1)

        self.python_label = ctk.CTkLabel(
            add_bar, text=os.path.basename(self.python_binary),
            font=("Arial", 10), fg_color=self.parser["colors"]["frame5_bg"],
            text_color="white", anchor="w"
        )
        self.python_label.pack(side="left", padx=1, pady=1)

        ctk.CTkLabel(add_bar, text="  ", font=("Arial", 12),
                     fg_color=self.parser["colors"]["frame5_bg"],
                     text_color="white").pack(side="left", padx=1, pady=1)

        self.add_btn = ctk.CTkButton(
            add_bar, text="+ Add Path", font=("Arial", 12),
            command=self.add_path_entry
        )
        self.add_btn.pack(side="left", padx=1, pady=1)

        self.sort_btn = ctk.CTkButton(
            add_bar, text="Sort A-Z", font=("Arial", 12),
            command=self.sort_path_entries
        )
        self.sort_btn.pack(side="left", padx=1, pady=1)

    def build_path_container(self):
        self.frame5_container = ctk.CTkFrame(
            self.root,
            fg_color=self.parser["colors"]["frame5_bg"],
            border_color=self.parser["colors"]["border_color"],
            border_width=1,
            width=self.window_width,
            height=650
        )
        self.frame5_container.pack(fill="both", expand=True)

        self.canvas = tk.Canvas(
            self.frame5_container,
            bg=self.parser["colors"]["frame5_bg"],
            highlightthickness=0
        )
        self.scrollable_frame = tk.Frame(self.canvas, bg=self.parser["colors"]["frame5_bg"])

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
        if hasattr(event, 'num') and event.num == 4:
            self.canvas.yview_scroll(-1, "units")
        elif hasattr(event, 'num') and event.num == 5:
            self.canvas.yview_scroll(1, "units")
        elif hasattr(event, 'delta'):
            if event.delta > 0:
                self.canvas.yview_scroll(-1, "units")
            else:
                self.canvas.yview_scroll(1, "units")

    def set_python_path(self):
        filename = filedialog.askopenfilename(
            title="Select Python 3.10 Binary",
            filetypes=[("Python Binaries", "python3*"), ("All Executables", "*")]
        )
        if not filename:
            return
        result = subprocess.run([filename, "--version"], capture_output=True, text=True, timeout=5)
        if result.returncode == 0 and "Python 3.10" in result.stdout:
            self.parser["python"]["binary_path"] = filename
            self.python_binary = filename
            self.python_label.configure(text=os.path.basename(filename))
            self.save()
            messagebox.showinfo("Success", f"Python binary set:\n{filename}")
        else:
            messagebox.showwarning("Invalid Python", f"Not Python 3.10:\n{filename}")

    def run_path(self, entry_data):
        path = entry_data['full_path'].strip()
        if not path:
            messagebox.showwarning("Empty Path", "No script selected for this entry.")
            return

        parts = shlex.split(path)
        executable = parts[0]
        args = parts[1:] if len(parts) > 1 else []
        cwd = None

        if executable.lower() in ('python', 'python3', 'python.exe', 'python3.exe') and args:
            cwd = os.path.dirname(os.path.abspath(args[0]))
            cmd = [self.python_binary] + args
        elif executable in ('bash', 'sh'):
            if args:
                cwd = os.path.dirname(os.path.abspath(args[0]))
            cmd = [executable] + args
        else:
            if not os.path.exists(executable) and not os.path.isabs(executable):
                executable = shutil.which(executable) or executable
            cwd = os.path.dirname(os.path.abspath(executable)) if os.path.exists(executable) else None
            if executable.endswith('.py'):
                cmd = [self.python_binary, executable] + args
            elif executable.endswith('.sh'):
                cmd = ['bash', executable] + args
            else:
                cmd = [executable] + args

        proc = subprocess.Popen(
            cmd, cwd=cwd, env=os.environ.copy(),
            start_new_session=True,
        )

        entry_data['proc'] = proc
        entry_data['label'].configure(
            text=f"{os.path.basename(path)}  ● pid {proc.pid}",
            text_color="lime",
        )

        def watch():
            proc.wait()
            if entry_data['frame'].winfo_exists():
                entry_data['label'].configure(
                    text=os.path.basename(path),
                    text_color=self.parser["colors"]["frame5_fg"],
                )
            entry_data['proc'] = None
        threading.Thread(target=watch, daemon=True).start()

    def stop_path(self, entry_data):
        proc = entry_data['proc']
        if proc is None or proc.poll() is not None:
            messagebox.showinfo("Not Running", "This entry is not currently running.")
            return

        pid = proc.pid
        pgid = os.getpgid(pid)
        os.killpg(pgid, signal.SIGTERM)

        def force_kill():
            for _ in range(self.KILL_GRACE_TICKS):
                if proc.poll() is not None:
                    return
                time.sleep(self.KILL_GRACE_SLEEP)
            os.killpg(pgid, signal.SIGKILL)
        threading.Thread(target=force_kill, daemon=True).start()

    def remove_path_entry(self, idx):
        confirm = messagebox.askyesno(
            "Remove Path",
            "Are you sure you want to remove this path entry?\n"
            "If it is running, it will be stopped."
        )
        if not confirm:
            return
        entry = self.path_entries[idx]
        proc = entry['proc']
        if proc is not None and proc.poll() is None:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        entry['frame'].destroy()
        self.path_entries.pop(idx)
        self.refresh_path_layout()
        self.save_paths_to_config()

    def sort_path_entries(self):
        self.path_entries.sort(key=lambda entry_data: os.path.basename(entry_data['full_path']).lower())
        self.refresh_path_layout()
        self.save_paths_to_config()

    def refresh_path_layout(self):
        for i, entry_data in enumerate(self.path_entries):
            col = i % 2
            row = i // 2
            entry_data['frame'].grid(row=row, column=col, padx=1, pady=1, sticky="nsew")
            entry_data['remove_btn'].configure(command=lambda idx=i: self.remove_path_entry(idx))

    def add_path_entry(self, path="", save_immediately=True):
        if len(self.path_entries) >= self.max_entries:
            messagebox.showwarning("Limit Reached", f"Maximum of {self.max_entries} entries reached.")
            return
        row_idx = len(self.path_entries)
        col = row_idx % 2
        row = row_idx // 2

        entry_frame = ctk.CTkFrame(self.scrollable_frame, fg_color=self.parser["colors"]["frame5_bg"])
        entry_frame.grid(row=row, column=col, padx=1, pady=1, sticky="nsew")
        entry_frame.grid_columnconfigure(0, weight=1)
        entry_frame.grid_columnconfigure(1, weight=0)
        entry_frame.grid_columnconfigure(2, weight=0)
        entry_frame.grid_columnconfigure(3, weight=0)
        entry_frame.grid_columnconfigure(4, weight=0)

        script_name = os.path.basename(path) if path else "[New]"

        name_label = ctk.CTkLabel(
            entry_frame, text=script_name,
            fg_color=self.parser["colors"]["frame5_bg"],
            text_color=self.parser["colors"]["frame5_fg"],
            anchor="w", font=("Courier", self.LIST_FONT_SIZE, "bold")
        )
        name_label.grid(row=0, column=0, sticky="ew", padx=1, pady=1)

        entry_data = {
            'frame': entry_frame,
            'full_path': path,
            'label': name_label,
            'browse_btn': None,
            'run_btn': None,
            'stop_btn': None,
            'remove_btn': None,
            'proc': None,
        }

        def open_browser():
            filename = filedialog.askopenfilename(
                title="Select Script or Binary",
                filetypes=[("Python Scripts", "*.py"), ("Executables", "*.exe"),
                           ("Shell Scripts", "*.sh"), ("All Files", "*.*")]
            )
            if filename:
                entry_data['full_path'] = filename
                name_label.configure(text=os.path.basename(filename))
                self.save_paths_to_config()

        browse_btn = ctk.CTkButton(entry_frame, text="Br", width=self.BUTTON_WIDTH, command=open_browser)
        browse_btn.grid(row=0, column=1, padx=1, pady=1)

        run_btn = ctk.CTkButton(
            entry_frame, text="R", width=self.BUTTON_WIDTH,
            fg_color="green", text_color="white",
            command=lambda: self.run_path(entry_data)
        )
        run_btn.grid(row=0, column=2, padx=1, pady=1)

        stop_btn = ctk.CTkButton(
            entry_frame, text="S", width=self.BUTTON_WIDTH,
            fg_color="darkred", text_color="white",
            command=lambda: self.stop_path(entry_data)
        )
        stop_btn.grid(row=0, column=3, padx=1, pady=1)

        remove_btn = ctk.CTkButton(
            entry_frame, text="X", width=self.BUTTON_WIDTH,
            fg_color="red", text_color="white",
            command=lambda idx=row_idx: self.remove_path_entry(idx)
        )
        remove_btn.grid(row=0, column=4, padx=1, pady=1)

        entry_data['browse_btn'] = browse_btn
        entry_data['run_btn'] = run_btn
        entry_data['stop_btn'] = stop_btn
        entry_data['remove_btn'] = remove_btn

        self.path_entries.append(entry_data)

        if save_immediately:
            self.save_paths_to_config()

    def save_paths_to_config(self):
        if self.parser.has_section("paths"):
            self.parser.remove_section("paths")
        self.parser.add_section("paths")

        for i, entry_data in enumerate(self.path_entries):
            path = entry_data['full_path'].strip()
            if path:
                self.parser["paths"][f"path_{i}"] = path

        self.save()

    def load_paths_from_config(self):
        if "paths" in self.parser:
            paths_dict = dict(self.parser["paths"])
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
        for entry in self.path_entries:
            proc = entry['proc']
            if proc is not None and proc.poll() is None:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        geom = self.root.geometry()
        # geometry format: WxH+X+Y
        pos_part = geom.split("+", 1)[1]
        x, y = pos_part.split("+")
        self.parser["window"]["x"] = x
        self.parser["window"]["y"] = y
        self.save()
        self.root.destroy()

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    SystemConfig()
    app = AIMenuGUI()
    app.run()