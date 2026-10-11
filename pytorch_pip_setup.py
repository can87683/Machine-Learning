#!/usr/bin/env python3
# pytorch_pip_setup.py
# Copyright Su Nie | BSD-3C License | https://github.com/can87683


from pathlib import Path
import configparser
import re
import subprocess
import sys
import threading
import time
import tkinter as tk
from tkinter import messagebox, filedialog
import customtkinter as ctk

CU = "cu129"


class CuSetup:
    """Headless logic: system info + pip download/install with streaming output."""

    def __init__(self, log_callback=None, progress_callback=None):
        self.log_callback = log_callback or (lambda s: print(s, end=""))
        self.progress_callback = progress_callback or (lambda *a, **k: None)
        self.dest = None
        self.abi = None

    def log(self, msg, end="\n"):
        self.log_callback(f"{msg}{end}")

    def python_abi(self):
        return f"cp{sys.version_info.major}{sys.version_info.minor}"

    def get_system_info(self, cu):
        self.abi = self.python_abi()
        self.dest = Path(f"{cu}-{self.abi}")
        in_venv = sys.prefix != sys.base_prefix
        return [
            f"Python       : {sys.version.split()[0]}",
            f"ABI          : {self.abi}",
            f"Executable   : {sys.executable}",
            f"Venv         : {sys.prefix}",
            f"In venv      : {in_venv}",
            f"Base prefix  : {sys.base_prefix}",
            f"Platform     : {sys.platform}",
            f"CUDA tag     : {cu}",
            f"Destination  : {self.dest}",
        ]

    @staticmethod
    def _human_size_to_bytes(num, unit):
        num = float(num)
        unit = unit.lower()
        mult = {
            "b": 1, "kb": 1024, "kib": 1024,
            "mb": 1024 ** 2, "mib": 1024 ** 2,
            "gb": 1024 ** 3, "gib": 1024 ** 3,
            "tb": 1024 ** 4, "tib": 1024 ** 4,
        }.get(unit, 1)
        return int(num * mult)

    @staticmethod
    def _newest_partial_file(directory: Path):
        try:
            candidates = []
            for p in directory.rglob("*"):
                if p.is_file():
                    candidates.append(p)
            if not candidates:
                return None
            return max(candidates, key=lambda p: p.stat().st_mtime)
        except Exception:
            return None

    def _run_streaming(self, cmd, tag, watch_dir: Path):
        self.log(f"[{tag}] $ {' '.join(map(str, cmd))}")
        self.log(f"[{tag}] cwd = {Path.cwd()}")

        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=0,
            universal_newlines=True,
        )
        assert proc.stdout is not None

        state = {
            "current_file": None,
            "current_total": None,
            "baseline_size": 0,
            "baseline_path": None,
        }
        stop_poll = threading.Event()

        def poller():
            while not stop_poll.is_set():
                try:
                    newest = self._newest_partial_file(watch_dir)
                    if newest is not None and state["current_file"] is not None:
                        if newest != state["baseline_path"]:
                            state["baseline_path"] = newest
                            state["baseline_size"] = 0

                        cur_size = newest.stat().st_size
                        done = max(0, cur_size - state["baseline_size"])
                        total = state["current_total"] or 0
                        if total > 0:
                            self.progress_callback(
                                "update",
                                {
                                    "file": state["current_file"],
                                    "done_bytes": min(done, total),
                                    "total_bytes": total,
                                },
                            )
                except Exception:
                    pass
                time.sleep(0.25)

        poll_thread = threading.Thread(target=poller, daemon=True)
        poll_thread.start()

        buf = ""
        try:
            while True:
                ch = proc.stdout.read(1)
                if not ch:
                    break
                if ch == "\r":
                    clean = self._ANSI_RE.sub("", buf).strip()
                    if clean:
                        self.log_callback(clean + "\n")
                    buf = ""
                elif ch == "\n":
                    clean = self._ANSI_RE.sub("", buf)
                    self._handle_pip_line(clean, state, watch_dir)
                    self.log_callback(clean + "\n")
                    buf = ""
                else:
                    buf += ch

            if buf:
                clean = self._ANSI_RE.sub("", buf)
                self._handle_pip_line(clean, state, watch_dir)
                self.log_callback(clean + "\n")
        finally:
            proc.wait()
            stop_poll.set()
            poll_thread.join(timeout=1.0)

        if proc.returncode != 0:
            raise RuntimeError(f"{tag} failed with code {proc.returncode}")
        self.log(f"[{tag}] Completed (rc=0)")
        self.progress_callback("reset", {})

    _ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")
    _DL_RE = re.compile(r"Downloading\s+(\S+)(?:\s+\(([\d.]+)\s*(\w+)\))?")
    _SAVED_RE = re.compile(r"Saved\s+(.+)")

    def _handle_pip_line(self, line, state, watch_dir):
        m = self._DL_RE.search(line)
        if m:
            url, size_num, size_unit = m.group(1), m.group(2), m.group(3)
            fname = url.rsplit("/", 1)[-1] or url
            total = (
                self._human_size_to_bytes(size_num, size_unit)
                if size_num and size_unit else None
            )
            state["current_file"] = fname
            state["current_total"] = total

            newest = self._newest_partial_file(watch_dir)
            state["baseline_path"] = newest
            state["baseline_size"] = newest.stat().st_size if newest else 0

            self.progress_callback(
                "start", {"file": fname, "size_bytes": total}
            )
            return

        m2 = self._SAVED_RE.search(line)
        if m2:
            self.progress_callback("done", {"file": state["current_file"]})
            state["current_file"] = None
            state["current_total"] = None
            state["baseline_path"] = None
            state["baseline_size"] = 0

    def download(self, cu):
        self.abi = self.python_abi()
        self.dest = Path(f"{cu}-{self.abi}")
        self.dest.mkdir(parents=True, exist_ok=True)

        cmd = [
            sys.executable,
            "-m", "pip", "download",
            "torch",
            "torchvision",
            "torchaudio",
            "--index-url",
            f"https://download.pytorch.org/whl/{cu}",
            "--dest",
            str(self.dest),
        ]
        self._run_streaming(cmd, "DOWNLOAD", watch_dir=self.dest)

    def install(self, cu, folder=None):
        """Install wheels from ``folder`` if given, otherwise from the
        default download destination for ``cu``."""
        if folder is not None:
            self.dest = Path(folder)
        else:
            if self.dest is None:
                self.abi = self.python_abi()
                self.dest = Path(f"{cu}-{self.abi}")

        wheels = sorted(self.dest.glob("*.whl"))
        if not wheels:
            raise RuntimeError(
                f"No wheels found in {self.dest}. Run Download first "
                "or choose a folder that contains .whl files."
            )

        cmd = [sys.executable, "-m", "pip", "install"] + [str(w) for w in wheels]
        watch = Path(sys.prefix) / "Lib" / "site-packages"
        if not watch.exists():
            watch = Path(sys.prefix) / "lib"
        self._run_streaming(cmd, "INSTALL", watch_dir=watch)


class CuSetupGUI(ctk.CTk):
    BG = "#1c1c1c"
    PAD = 2
    WIN_W = 640
    WIN_H = 900

    # ---------- nested stream tee ----------
    class _StreamTee:
        def __init__(self, original, sink):
            self._original = original
            self._sink = sink

        def write(self, data):
            if not data:
                return 0
            try:
                self._original.write(data)
            except Exception:
                pass
            try:
                self._sink(data)
            except Exception:
                pass
            return len(data)

        def flush(self):
            try:
                self._original.flush()
            except Exception:
                pass

        def isatty(self):
            return False

    def __init__(self):
        super().__init__()
        ctk.set_appearance_mode("dark")

        self.withdraw()

        self.title("PyTorch Stack Setup")
        self.resizable(False, False)
        self.configure(fg_color=self.BG)

        self._log_queue = []
        self._log_lock = threading.Lock()
        self._progress_queue = []
        self._progress_lock = threading.Lock()

        self.setup = CuSetup(
            log_callback=self.queue_log,
            progress_callback=self.queue_progress,
        )

        self._build_ui()
        self.protocol("WM_DELETE_WINDOW", self._on_close)

        self.after(80, self._drain_log_queue)
        self.after(80, self._drain_progress_queue)

        sys.stdout = self._StreamTee(sys.stdout, lambda s: self.queue_log(s))
        sys.stderr = self._StreamTee(sys.stderr, lambda s: self.queue_log(s))

        self._show_system_info()

        self._apply_initial_geometry()
        self.deiconify()
        self.lift()
        self.focus_force()

    # ---------- geometry ----------
    def _apply_initial_geometry(self):
        self.update_idletasks()
        x, y = self._load_geometry()

        if x is None or y is None or not self._is_on_screen(x, y, self.WIN_W, self.WIN_H):
            sw = self.winfo_screenwidth()
            sh = self.winfo_screenheight()
            x = max(0, (sw - self.WIN_W) // 2)
            y = max(0, (sh - self.WIN_H) // 2)

        # Set size + outer-frame origin in one shot while withdrawn.
        self.geometry(f"{self.WIN_W}x{self.WIN_H}+{x}+{y}")
        self.update_idletasks()

    def _persist_geometry(self):
        try:
            self.update_idletasks()
            # winfo_x/y is the outer-frame origin in root coords -- same
            # coordinate space that geometry("+X+Y") uses on restore.
            x = self.winfo_x()
            y = self.winfo_y()
            self._save_geometry(x, y)
        except Exception:
            pass

    def _is_on_screen(self, x, y, w, h):
        sw = self.winfo_screenwidth()
        sh = self.winfo_screenheight()
        return (-w + 80 < x < sw - 80) and (0 <= y < sh - 40)

    def _load_geometry(self):
        try:
            cp = configparser.ConfigParser()
            cp.read(self._ini_file(), encoding="utf-8")
            x = cp.getint("geometry", "x", fallback=None)
            y = cp.getint("geometry", "y", fallback=None)
            return x, y
        except Exception:
            return None, None

    def _save_geometry(self, x, y):
        try:
            cp = configparser.ConfigParser()
            cp.read(self._ini_file(), encoding="utf-8")
            if not cp.has_section("geometry"):
                cp.add_section("geometry")
            cp.set("geometry", "x", str(int(x)))
            cp.set("geometry", "y", str(int(y)))
            with open(self._ini_file(), "w", encoding="utf-8") as fh:
                cp.write(fh)
        except Exception:
            pass

    def _ini_file(self):
        # whatever-script-is-named.ini, next to the script (or exe).
        stem = Path(sys.argv[0]).stem or "pytorch_download_console"
        base_dir = Path(sys.argv[0]).resolve().parent if sys.argv[0] else Path.cwd()
        return base_dir / f"{stem}.ini"

    # ---------- layout ----------
    def _build_ui(self):
        p = self.PAD

        self.version_frame = ctk.CTkFrame(self, fg_color=self.BG, border_width=2)
        self.version_frame.pack(fill="x", padx=p, pady=p)

        self.version_title = ctk.CTkLabel(
            self.version_frame,
            text="Copyright Su Nie | BSD-3C License | https://github.com/can87683",
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color="#ffff00",
            anchor="w",
        )
        self.version_title.pack(fill="x", padx=p, pady=p, anchor="w")

        self.version_var = tk.StringVar(value=CU)
        self.version_entry = ctk.CTkEntry(
            self.version_frame,
            textvariable=self.version_var,
            font=ctk.CTkFont(size=12),
            text_color="#ffffff",
        )
        self.version_entry.pack(fill="x", padx=p, pady=p)

        self.button_frame = ctk.CTkFrame(self, fg_color=self.BG, border_width=2)
        self.button_frame.pack(fill="x", padx=p, pady=p)

        self.button_frame.grid_columnconfigure(0, weight=1, uniform="btn")
        self.button_frame.grid_columnconfigure(1, weight=1, uniform="btn")

        self.download_btn = ctk.CTkButton(
            self.button_frame,
            text="Download",
            font=ctk.CTkFont(size=12),
            text_color="#ffffff",
            command=self._on_download,
        )
        self.download_btn.grid(row=0, column=0, sticky="ew", padx=p, pady=p)

        self.install_btn = ctk.CTkButton(
            self.button_frame,
            text="Install",
            font=ctk.CTkFont(size=12),
            text_color="#ffffff",
            command=self._on_install,
        )
        self.install_btn.grid(row=0, column=1, sticky="ew", padx=p, pady=p)

        self.progress_frame = ctk.CTkFrame(self, fg_color=self.BG, border_width=2)
        self.progress_frame.pack(fill="x", padx=p, pady=p)

        self.progress_title = ctk.CTkLabel(
            self.progress_frame,
            text="Progress",
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color="#ffff00",
            anchor="w",
        )
        self.progress_title.pack(fill="x", padx=p, pady=p, anchor="w")

        self.progress_file_label = ctk.CTkLabel(
            self.progress_frame,
            text="Idle.",
            font=ctk.CTkFont(size=12),
            text_color="#ffffff",
            anchor="w",
        )
        self.progress_file_label.pack(fill="x", padx=p, pady=p, anchor="w")

        self.progress_bar = ctk.CTkProgressBar(
            self.progress_frame,
            height=14,
            progress_color="#3a86ff",
            mode="determinate",
        )
        self.progress_bar.set(0.0)
        self.progress_bar.pack(fill="x", padx=p, pady=p)

        self.log_frame = ctk.CTkFrame(self, fg_color=self.BG, border_width=2)
        self.log_frame.pack(fill="both", expand=True, padx=p, pady=p)

        self.log_title = ctk.CTkLabel(
            self.log_frame,
            text="Log",
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color="#ffff00",
            anchor="w",
        )
        self.log_title.pack(fill="x", padx=p, pady=p, anchor="w")

        self.log_text = tk.Text(
            self.log_frame,
            bg="#101010",
            fg="#ffffff",
            insertbackground="#ffffff",
            font=("Courier New", 10),
            wrap="none",
            borderwidth=2,
            relief="solid",
            highlightthickness=0,
        )
        self.log_text.pack(fill="both", expand=True, padx=p, pady=p)

        self.log_text.bind("<Control-c>", self._copy_selected)
        self.log_text.bind("<Control-a>", self._select_all)
        self.log_text.bind("<Button-3>", self._popup_menu)
        self.log_text.bind("<Button-2>", self._popup_menu)

        self._menu = tk.Menu(self, tearoff=0)
        self._menu.add_command(label="Copy Selected", command=self._copy_selected)
        self._menu.add_command(label="Select All", command=self._select_all)
        self._menu.add_command(label="Copy All", command=self._copy_all)

    # ---------- window helpers ----------
    def _on_close(self):
        self._persist_geometry()

        try:
            if isinstance(sys.stdout, self._StreamTee):
                sys.stdout = sys.stdout._original
            if isinstance(sys.stderr, self._StreamTee):
                sys.stderr = sys.stderr._original
        except Exception:
            pass

        if messagebox.askyesno("Exit", "Confirm exit?"):
            self.destroy()

    # ---------- log queue ----------
    def queue_log(self, msg):
        with self._log_lock:
            self._log_queue.append(str(msg))

    def _drain_log_queue(self):
        with self._log_lock:
            items = self._log_queue[:]
            self._log_queue.clear()
        if items:
            for item in items:
                self.log_text.insert("end", item)
            self.log_text.see("end")
        self.after(80, self._drain_log_queue)

    # ---------- progress queue ----------
    def queue_progress(self, event, payload):
        with self._progress_lock:
            self._progress_queue.append((event, payload))

    def _drain_progress_queue(self):
        with self._progress_lock:
            items = self._progress_queue[:]
            self._progress_queue.clear()
        for event, payload in items:
            self._apply_progress(event, payload)
        self.after(80, self._drain_progress_queue)

    def _apply_progress(self, event, payload):
        if event == "start":
            fname = payload.get("file") or "unknown"
            total = payload.get("size_bytes")
            self.progress_bar.set(0.0)
            if total:
                self.progress_file_label.configure(
                    text=f"Downloading {fname}  ({self._fmt(total)})"
                )
            else:
                self.progress_file_label.configure(text=f"Downloading {fname}…")

        elif event == "update":
            fname = payload.get("file") or "unknown"
            done = payload.get("done_bytes", 0)
            total = payload.get("total_bytes", 0)
            if total > 0:
                self.progress_bar.set(min(1.0, done / total))
                self.progress_file_label.configure(
                    text=f"{fname}   {self._fmt(done)} / {self._fmt(total)}"
                )

        elif event == "done":
            fname = payload.get("file") or "unknown"
            self.progress_bar.set(1.0)
            self.progress_file_label.configure(text=f"Saved {fname}")

        elif event == "reset":
            self.progress_bar.set(0.0)
            self.progress_file_label.configure(text="Idle.")

    def _fmt(self, n):
        n = float(n)
        for unit in ("B", "KB", "MB", "GB", "TB"):
            if n < 1024:
                return f"{n:.1f} {unit}"
            n /= 1024
        return f"{n:.1f} PB"

    # ---------- log interactions ----------
    def _show_system_info(self):
        self.queue_log("=== System / Venv Info ===\n")
        for line in self.setup.get_system_info(self.version_var.get()):
            self.queue_log(line + "\n")
        self.queue_log("==========================\n")

    def _copy_selected(self, event=None):
        try:
            sel = self.log_text.get("sel.first", "sel.last")
            self.clipboard_clear()
            self.clipboard_append(sel)
        except tk.TclError:
            pass
        return "break"

    def _select_all(self, event=None):
        self.log_text.tag_add("sel", "1.0", "end-1c")
        return "break"

    def _copy_all(self):
        data = self.log_text.get("1.0", "end-1c")
        self.clipboard_clear()
        self.clipboard_append(data)

    def _popup_menu(self, event):
        try:
            self._menu.tk_popup(event.x_root, event.y_root)
        finally:
            self._menu.grab_release()
        return "break"

    # ---------- actions ----------
    def _set_busy(self, busy):
        state = "disabled" if busy else "normal"
        self.download_btn.configure(state=state)
        self.install_btn.configure(state=state)
        self.version_entry.configure(state=state)
        if not busy:
            self.progress_bar.set(0.0)
            self.progress_file_label.configure(text="Idle.")

    def _on_download(self):
        cu = self.version_var.get().strip()
        if not cu:
            messagebox.showwarning(
                "Missing", "Please enter a CUDA version tag (e.g. cu129)."
            )
            return
        self._set_busy(True)
        self.queue_log(f"\n--- Download start ({cu}) ---\n")

        def worker():
            try:
                self.setup.download(cu)
            except Exception as e:
                self.queue_log(f"[ERROR] {e}\n")
            finally:
                self._set_busy(False)

        threading.Thread(target=worker, daemon=True).start()

    def _on_install(self):
        cu = self.version_var.get().strip()
        if not cu:
            messagebox.showwarning(
                "Missing", "Please enter a CUDA version tag (e.g. cu129)."
            )
            return

        folder = filedialog.askdirectory(
            title="Choose folder containing .whl files",
            mustexist=True,
        )
        if not folder:
            return

        self._set_busy(True)
        self.queue_log(f"\n--- Install start ({cu}) ---\n")
        self.queue_log(f"Wheel folder: {folder}\n")

        def worker():
            try:
                self.setup.install(cu, folder=folder)
            except Exception as e:
                self.queue_log(f"[ERROR] {e}\n")
            finally:
                self._set_busy(False)

        threading.Thread(target=worker, daemon=True).start()


def main():
    app = CuSetupGUI()
    app.mainloop()


if __name__ == "__main__":
    main()