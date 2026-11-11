#!/usr/bin/env python3
# nuitka_convert.py
# Copyright Su Nie | BSD-3C License | https://github.com/can87683


"""
requirements: nuitka
"""

import configparser
import pathlib
import shutil
import subprocess
import sys
import threading


class NuitkaConvert:
    # ------------------------------------------------------------------
    # Editable constants
    # ------------------------------------------------------------------
    OUTPUT_DIRECTORY = "_nuitka_cpp"
    INI_FILENAME = "nuitka_convert.ini"
    PYTHON_EXECUTABLE = sys.executable
    NUITKA_MODULE = "nuitka"
    PYTHON_SUFFIX = ".py"
    CPP_SUFFIX = ".cpp"
    C_SUFFIX = ".c"
    CACHE_DIRECTORY_NAME = "__pycache__"
    NUITKA_CACHE_DIRECTORY_NAME = ".nuitka-cache"

    def __init__(self):
        self.script_path = pathlib.Path(sys.argv[1]).resolve()
        self.project_root = self.script_path.parent
        self.ini_path = self.project_root / self.INI_FILENAME
        self.output_root = self.project_root / self.OUTPUT_DIRECTORY
        self.lock = threading.Lock()

        self.state = configparser.ConfigParser()
        if self.ini_path.is_file():
            self.state.read(self.ini_path, encoding="utf-8")

        if "state" not in self.state:
            self.state["state"] = {}

        self.state["state"]["last_target"] = str(self.script_path)
        self.state["state"]["project_root"] = str(self.project_root)
        self.state["state"]["output_directory"] = str(self.output_root)

        with self.ini_path.open("w", encoding="utf-8") as file:
            self.state.write(file)

    def audit_project(self):
        excluded_directories = {
            self.CACHE_DIRECTORY_NAME,
            self.NUITKA_CACHE_DIRECTORY_NAME,
            self.OUTPUT_DIRECTORY,
            ".git",
            ".hg",
            ".svn",
        }

        python_files = []

        for path in self.project_root.rglob(self.PYTHON_SUFFIX):
            relative_parts = path.relative_to(self.project_root).parts

            if any(part in excluded_directories for part in relative_parts):
                continue

            if path.is_file():
                python_files.append(path)

        python_files.sort()

        if not python_files:
            raise RuntimeError("No Python source files were found.")

        return python_files

    def convert_file(self, source_path):
        relative_source = source_path.relative_to(self.project_root)
        relative_directory = relative_source.parent

        destination_directory = self.output_root / relative_directory
        destination_directory.mkdir(parents=True, exist_ok=True)

        temporary_directory = destination_directory / f".{source_path.stem}.nuitka"
        temporary_directory.mkdir(parents=True, exist_ok=True)

        command = [
            self.PYTHON_EXECUTABLE,
            "-m",
            self.NUITKA_MODULE,
            "--generate-c-only",
            "--nofollow-imports",
            f"--output-dir={temporary_directory}",
            str(source_path),
        ]

        print()
        print(f"[Nuitka] {relative_source}")

        completed = subprocess.run(
            command,
            cwd=self.project_root,
            check=True,
        )

        if completed.returncode != 0:
            raise RuntimeError(
                f"Nuitka failed for {source_path}"
            )

        generated_c = temporary_directory / f"{source_path.stem}{self.C_SUFFIX}"
        generated_cpp = destination_directory / f"{source_path.stem}{self.CPP_SUFFIX}"

        if not generated_c.is_file():
            generated_candidates = sorted(
                temporary_directory.glob(f"*{self.C_SUFFIX}")
            )

            if len(generated_candidates) != 1:
                raise FileNotFoundError(
                    f"Unable to identify generated source for {source_path}"
                )

            generated_c = generated_candidates[0]

        shutil.copy2(generated_c, generated_cpp)

        shutil.rmtree(temporary_directory)

        print(f"[Created] {generated_cpp.relative_to(self.project_root)}")

    def run(self):
        if len(sys.argv) != 2:
            raise SystemExit(
                "Usage: python nuitka_convert.py target_entry.py"
            )

        if not self.script_path.is_file():
            raise FileNotFoundError(
                f"Target entry script does not exist: {self.script_path}"
            )

        if self.script_path.suffix != self.PYTHON_SUFFIX:
            raise ValueError(
                "The target entry must be a .py file."
            )

        python_files = self.audit_project()

        self.output_root.mkdir(parents=True, exist_ok=True)

        print(f"Project : {self.project_root}")
        print(f"Target  : {self.script_path.name}")
        print(f"Python files found: {len(python_files)}")
        print(f"Output  : {self.output_root}")

        for source_path in python_files:
            self.convert_file(source_path)

        print()
        print("Conversion completed.")
        print(f"Generated files are in: {self.output_root}")


if __name__ == "__main__":
    NuitkaConvert().run()
```