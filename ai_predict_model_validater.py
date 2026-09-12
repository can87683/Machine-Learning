#!/usr/bin/env python3
# ai_predict_model_validater.py

import setproctitle
setproctitle.setproctitle("ai_predict_model_validater")

import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, filedialog
import threading
import os
import sys
import json
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import traceback
from typing import Dict, List, Tuple, Optional, Any, Union
import configparser
import pyperclip
import hashlib
import time
import shutil

import catboost
from catboost import CatBoostClassifier, CatBoostRegressor
import xgboost as xgb
import lightgbm as lgb
import torch
import torch.nn as nn
import joblib
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor

# ===== CONFIGURATION =====
CONFIG_FILE = "ai_predict_model_validater.ini"

class ConfigManager:
    """Manages GUI state persistence"""

    @staticmethod
    def load_config():
        config = configparser.ConfigParser()
        if Path(CONFIG_FILE).exists():
            config.read(CONFIG_FILE)
        return config

    @staticmethod
    def save_config(config):
        with open(CONFIG_FILE, 'w') as f:
            config.write(f)

    @staticmethod
    def save_window_position(root):
        config = configparser.ConfigParser()
        config.read(CONFIG_FILE)

        if 'Window' not in config:
            config['Window'] = {}

        config['Window']['x'] = str(root.winfo_x())
        config['Window']['y'] = str(root.winfo_y())
        config['Window']['width'] = str(root.winfo_width())
        config['Window']['height'] = str(root.winfo_height())

        ConfigManager.save_config(config)

    @staticmethod
    def restore_window_position(root, default_width=640, default_height=1020):
        config = ConfigManager.load_config()

        if 'Window' in config:
            x = int(config['Window'].get('x', '100'))
            y = int(config['Window'].get('y', '100'))
            width = int(config['Window'].get('width', str(default_width)))
            height = int(config['Window'].get('height', str(default_height)))

            screen_width = root.winfo_screenwidth()
            screen_height = root.winfo_screenheight()

            if x + width > screen_width:
                x = screen_width - width
            if y + height > screen_height:
                y = screen_height - height
            if x < 0:
                x = 0
            if y < 0:
                y = 0

            root.geometry(f"{width}x{height}+{x}+{y}")
        else:
            root.geometry(f"{default_width}x{default_height}+100+100")

class GUISettings:
    WINDOW_WIDTH = 640
    WINDOW_HEIGHT = 1020

    COLOR_BG = "#300a24"
    COLOR_FG = "#ffffff"
    COLOR_FG_DIM = "#aaaaaa"
    COLOR_BUTTON = "#5d1049"
    COLOR_BUTTON_ACTIVE = "#4caf50"
    COLOR_BUTTON_STOP = "#f44336"
    COLOR_BUTTON_DISABLED = "#444444"
    COLOR_LOG_BG = "black"
    COLOR_LOG_FG = "white"
    COLOR_FRAME_BG = "#2d2d2d"
    COLOR_TABLE_HEADER = "#444444"
    COLOR_VALID = "#1B5E20"
    COLOR_INVALID = "#B71C1C"
    COLOR_WARNING = "#F57C00"
    COLOR_DUPLICATE = "#0D47A1"
    COLOR_REMOVE = "#d32f2f"

    FONT_DEFAULT = ("Ubuntu", 10)
    FONT_MONO = ("Ubuntu Mono", 10)
    FONT_HEADER = ("Ubuntu", 10, "bold")
    FONT_TITLE = ("Ubuntu", 12, "bold")

class ModelValidator:
    """Validates trained models of various formats"""

    def __init__(self, log_callback=None):
        self.log_callback = log_callback or print
        self.validation_results = {}
        self.model_formats = [
            '.cbm', '.json', '.txt', '.pt', '.pth', '.pkl', '.joblib',
            '.model', '.h5', '.hdf5', '.keras', '.sav', '.dat'
        ]

        self.supported_models = [
            'catboost', 'cnn', 'finrl', 'gru', 'lightgbm',
            'lstm', 'patchtst', 'tcn', 'tft', 'xgboost'
        ]

    def log(self, message: str, level: str = "info"):
        if self.log_callback:
            timestamp = datetime.now().strftime("%H:%M:%S")
            log_msg = f"[{timestamp}] {message}"
            self.log_callback(log_msg, level)

    def validate_model(self, model_path: Path) -> Dict[str, Any]:
        results = {
            'valid': True,
            'status': 'VALID',
            'errors': [],
            'warnings': [],
            'info': {
                'path': str(model_path),
                'filename': model_path.name,
                'extension': model_path.suffix.lower(),
                'size_bytes': model_path.stat().st_size,
                'size_mb': model_path.stat().st_size / (1024 * 1024),
                'modified': datetime.fromtimestamp(model_path.stat().st_mtime),
                'created': datetime.fromtimestamp(model_path.stat().st_ctime),
                'model_type': 'unknown',
                'symbol': 'unknown',
                'timeframe': 'unknown',
                'trained_date': 'unknown'
            }
        }

        self._extract_model_info(model_path, results)

        file_size = results['info']['size_bytes']
        if file_size < 100:
            results['warnings'].append(f"File size very small: {file_size} bytes")
            results['status'] = 'WARNING'

        if file_size > 100 * 1024 * 1024:
            results['warnings'].append(f"File size very large: {results['info']['size_mb']:.1f} MB")
            results['status'] = 'WARNING'

        metadata_path = model_path.with_suffix('.json')
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            results['info']['metadata'] = metadata
            results['info']['model_type'] = metadata.get('model_type', results['info']['model_type'])
            results['info']['symbol'] = metadata.get('symbol', results['info']['symbol'])
            results['info']['timeframe'] = metadata.get('timeframe', results['info']['timeframe'])
            results['info']['trained_date'] = metadata.get('created_at', results['info']['trained_date'])
        else:
            results['warnings'].append("No metadata file found")
            results['status'] = 'WARNING'

        extension = results['info']['extension']
        model_type = results['info']['model_type'].lower()

        if model_type not in self.supported_models and model_type != 'unknown':
            results['warnings'].append(f"Model type '{model_type}' not in supported list")
            results['status'] = 'WARNING'

        if extension == '.cbm':
            results.update(self._validate_catboost(model_path))
        elif extension == '.json':
            if 'xgboost' in model_path.stem.lower():
                results.update(self._validate_xgboost(model_path))
            elif 'lightgbm' in model_path.stem.lower():
                results.update(self._validate_lightgbm_json(model_path))
            else:
                results.update(self._validate_json_generic(model_path))
        elif extension == '.txt':
            results.update(self._validate_lightgbm_txt(model_path))
        elif extension in ['.pt', '.pth']:
            results.update(self._validate_pytorch(model_path))
        elif extension in ['.pkl', '.joblib', '.sav', '.dat']:
            results.update(self._validate_pickle(model_path))
        elif extension in ['.h5', '.hdf5', '.keras']:
            results.update(self._validate_keras(model_path))
        else:
            results.update(self._validate_generic(model_path))

        if results['errors']:
            results['valid'] = False
            results['status'] = 'INVALID'
        elif results['warnings'] and results['status'] == 'VALID':
            results['status'] = 'WARNING'

        results['summary'] = self._create_summary(results)
        return results

    def _extract_model_info(self, model_path: Path, results: Dict[str, Any]):
        filename = model_path.stem.lower()
        timeframes = ['m1', 'm5', 'm15', 'm30', 'h1', 'h4', 'd1', 'w1', 'mn1']

        for tf in timeframes:
            if f"_{tf}" in filename or filename.endswith(f"_{tf}"):
                results['info']['timeframe'] = tf.upper()
                parts = filename.split(f"_{tf}")
                if parts[0]:
                    symbol_part = parts[0]
                    for model in self.supported_models:
                        if symbol_part.endswith(f"_{model}"):
                            symbol_part = symbol_part[:-(len(model) + 1)]
                            break
                    results['info']['symbol'] = symbol_part.upper()
                break

        if results['info']['symbol'] == 'unknown':
            common_symbols = ['eurusd', 'gbpusd', 'usdjpy', 'audusd', 'usdchf', 'xauusd', 'nd100', 'sp500', 'hk50']
            for symbol in common_symbols:
                if symbol in filename:
                    results['info']['symbol'] = symbol.upper()
                    break

        model_keywords = {
            'catboost': ['catboost', 'cbm'],
            'xgboost': ['xgboost', 'xgb'],
            'lightgbm': ['lightgbm', 'lgb'],
            'lstm': ['lstm'],
            'gru': ['gru'],
            'tft': ['tft', 'transformer'],
            'cnn': ['cnn', 'conv'],
            'finrl': ['finrl'],
            'patchtst': ['patchtst', 'patch'],
            'tcn': ['tcn', 'temporal']
        }

        for model_type, keywords in model_keywords.items():
            if any(keyword in filename for keyword in keywords):
                results['info']['model_type'] = model_type
                break

    def _validate_catboost(self, model_path: Path) -> Dict[str, Any]:
        results = {'details': {}}
        model = CatBoostClassifier()
        model.load_model(str(model_path))
        results['details']['type'] = 'classifier'

        if hasattr(model, 'tree_count_'):
            results['details']['tree_count'] = model.tree_count_

        n_features = model.n_features_in_ if hasattr(model, 'n_features_in_') else 10
        dummy_data = np.random.rand(1, n_features).astype(np.float32)
        pred = model.predict(dummy_data)
        results['details']['prediction_test'] = 'passed'
        results['details']['prediction_shape'] = str(pred.shape)
        return results

    def _validate_xgboost(self, model_path: Path) -> Dict[str, Any]:
        results = {'details': {}}
        model = xgb.Booster()
        model.load_model(str(model_path))
        results['details']['type'] = 'booster'

        if hasattr(model, 'num_boosted_rounds'):
            results['details']['num_boosted_rounds'] = model.num_boosted_rounds()

        n_features = model.n_features_in_ if hasattr(model, 'n_features_in_') else 10
        dummy_data = xgb.DMatrix(np.random.rand(1, n_features).astype(np.float32))
        pred = model.predict(dummy_data)
        results['details']['prediction_test'] = 'passed'
        results['details']['prediction_shape'] = str(pred.shape)
        return results

    def _validate_lightgbm_txt(self, model_path: Path) -> Dict[str, Any]:
        results = {'details': {}}
        model = lgb.Booster(model_file=str(model_path))

        if hasattr(model, 'num_trees'):
            results['details']['num_trees'] = model.num_trees()

        n_features = model.num_features() if hasattr(model, 'num_features') else 10
        dummy_data = np.random.rand(1, n_features).astype(np.float32)
        pred = model.predict(dummy_data)
        results['details']['prediction_test'] = 'passed'
        results['details']['prediction_shape'] = str(pred.shape)
        return results

    def _validate_lightgbm_json(self, model_path: Path) -> Dict[str, Any]:
        results = {'details': {}}
        model = lgb.Booster(model_file=str(model_path))
        results['details']['type'] = 'lightgbm_json'

        if hasattr(model, 'num_trees'):
            results['details']['num_trees'] = model.num_trees()

        n_features = model.num_features() if hasattr(model, 'num_features') else 10
        dummy_data = np.random.rand(1, n_features).astype(np.float32)
        pred = model.predict(dummy_data)
        results['details']['prediction_test'] = 'passed'
        return results

    def _validate_json_generic(self, model_path: Path) -> Dict[str, Any]:
        results = {'details': {}}
        with open(model_path, 'r') as f:
            content = json.load(f)
        results['details']['json_keys'] = list(content.keys())
        results['details']['json_size'] = len(str(content))
        return results

    def _validate_pytorch(self, model_path: Path) -> Dict[str, Any]:
        results = {'details': {}}
        checkpoint = torch.load(str(model_path), map_location='cpu', weights_only=False)

        if isinstance(checkpoint, nn.Module):
            model = checkpoint
            model.eval()
            results['details']['format'] = 'full_model'
            results['details']['model_class'] = model.__class__.__name__

            model_str = str(model).lower()
            if 'lstm' in model_str:
                results['info']['model_type'] = 'lstm'
            elif 'gru' in model_str:
                results['info']['model_type'] = 'gru'
            elif 'transformer' in model_str or 'attention' in model_str:
                results['info']['model_type'] = 'tft'
            elif 'conv' in model_str:
                results['info']['model_type'] = 'cnn'
            elif 'patch' in model_str:
                results['info']['model_type'] = 'patchtst'
            elif 'temporal' in model_str:
                results['info']['model_type'] = 'tcn'
            elif 'linear' in model_str or 'dense' in model_str:
                results['info']['model_type'] = 'finrl'

            input_size = 10
            for param in model.parameters():
                if len(param.shape) >= 2:
                    input_size = param.shape[1]
                    break

            dummy_input = torch.randn(1, input_size)
            with torch.no_grad():
                output = model(dummy_input)
            results['details']['forward_pass'] = 'passed'
            results['details']['output_shape'] = str(output.shape)

        elif isinstance(checkpoint, dict):
            results['details']['format'] = 'state_dict'
            results['details']['keys'] = list(checkpoint.keys())

        return results

    def _validate_pickle(self, model_path: Path) -> Dict[str, Any]:
        results = {'details': {}}
        model = joblib.load(str(model_path))
        results['details']['loader'] = 'joblib'
        results['details']['model_class'] = model.__class__.__name__

        n_features = 10
        if hasattr(model, 'n_features_in_'):
            n_features = model.n_features_in_

        dummy_data = np.random.rand(1, n_features).astype(np.float32)
        pred = model.predict(dummy_data)
        results['details']['prediction_test'] = 'passed'
        results['details']['prediction_shape'] = str(pred.shape)
        return results

    def _validate_keras(self, model_path: Path) -> Dict[str, Any]:
        results = {'details': {}}
        from tensorflow import keras
        model = keras.models.load_model(str(model_path))

        results['details']['model_class'] = model.__class__.__name__
        results['details']['num_layers'] = len(model.layers)
        results['details']['total_parameters'] = model.count_params()

        input_shape = model.input_shape
        if input_shape and len(input_shape) >= 2:
            batch_size = 1
            input_dims = input_shape[1:]
            dummy_input = np.random.rand(batch_size, *input_dims).astype(np.float32)
            pred = model.predict(dummy_input)
            results['details']['prediction_test'] = 'passed'
        return results

    def _validate_generic(self, model_path: Path) -> Dict[str, Any]:
        results = {'details': {}}
        with open(model_path, 'rb') as f:
            data = f.read(100)
        results['details']['file_readable'] = True
        results['details']['first_bytes'] = len(data)
        return results

    def _create_summary(self, results: Dict[str, Any]) -> str:
        if not results['valid']:
            return f"INVALID: {', '.join(results['errors'][:2])}"

        info = results['info']
        details = results.get('details', {})
        summary_parts = []

        if info['model_type'] != 'unknown':
            summary_parts.append(f"{info['model_type'].upper()}")

        if info['symbol'] != 'unknown' and info['timeframe'] != 'unknown':
            summary_parts.append(f"{info['symbol']} {info['timeframe']}")

        summary_parts.append(f"{info['size_mb']:.1f}MB")

        if details.get('prediction_test') == 'passed':
            summary_parts.append("✓ Predict")
        elif details.get('forward_pass') == 'passed':
            summary_parts.append("✓ Forward")

        if results['warnings']:
            summary_parts.append(f"{len(results['warnings'])} warnings")

        return " | ".join(summary_parts)

    def scan_and_validate(self, root_dir: Path, recursive: bool = True) -> Dict[str, Dict[str, Any]]:
        self.log(f"📂 Scanning directory: {root_dir}", "info")

        model_files = []
        if recursive:
            for ext in self.model_formats:
                model_files.extend(root_dir.rglob(f"*{ext}"))
        else:
            for ext in self.model_formats:
                model_files.extend(root_dir.glob(f"*{ext}"))

        self.log(f"📊 Found {len(model_files)} model files", "info")

        self.validation_results = {}
        total = len(model_files)

        for i, model_file in enumerate(model_files, 1):
            self.log(f"🔍 [{i}/{total}] Validating {model_file.name}...", "info")
            result = self.validate_model(model_file)
            self.validation_results[str(model_file)] = result

            if result['valid'] and result['status'] == 'VALID':
                self.log(f"   ✅ {model_file.name} - VALID", "success")
            elif result['valid'] and result['status'] == 'WARNING':
                self.log(f"   ⚠️ {model_file.name} - WARNING: {len(result['warnings'])} warnings", "warning")
            else:
                self.log(f"   ❌ {model_file.name} - INVALID: {result['errors'][0][:50]}", "error")

        valid_count = sum(1 for r in self.validation_results.values() if r['valid'] and r['status'] == 'VALID')
        warning_count = sum(1 for r in self.validation_results.values() if r['status'] == 'WARNING')
        invalid_count = sum(1 for r in self.validation_results.values() if not r['valid'] or r['status'] == 'INVALID')
        total_processed = len(self.validation_results)

        self.log(f"\n📊 Validation Summary:", "info")
        self.log(f"   Total models: {total_processed}", "info")
        self.log(f"   ✅ Valid: {valid_count} ({valid_count/total_processed*100:.1f}%)", "success")
        self.log(f"   ⚠️  Warnings: {warning_count} ({warning_count/total_processed*100:.1f}%)", "warning")
        self.log(f"   ❌ Invalid: {invalid_count} ({invalid_count/total_processed*100:.1f}%)", "error")

        return self.validation_results

    def find_duplicates(self, root_dir: Path, recursive: bool = True) -> Dict[str, List[str]]:
        self.log("🔍 Scanning for duplicates...", "info")

        model_files = []
        if recursive:
            for ext in self.model_formats:
                model_files.extend(root_dir.rglob(f"*{ext}"))
        else:
            for ext in self.model_formats:
                model_files.extend(root_dir.glob(f"*{ext}"))

        file_hashes = {}
        duplicates = {}

        for file_path in model_files:
            sha256 = hashlib.sha256()
            with open(file_path, 'rb') as f:
                for chunk in iter(lambda: f.read(8192), b''):
                    sha256.update(chunk)
            file_hash = sha256.hexdigest()

            if file_hash in file_hashes:
                file_hashes[file_hash].append(str(file_path))
            else:
                file_hashes[file_hash] = [str(file_path)]

        duplicates = {h: paths for h, paths in file_hashes.items() if len(paths) > 1}
        self.log(f"📊 Found {len(duplicates)} duplicate groups", "info")
        return duplicates

    def remove_non_valid_models(self, results: Dict[str, Dict[str, Any]], backup: bool = True) -> Tuple[int, List[str]]:
        removed_files = []
        kept_files = []

        for path_str, result in results.items():
            if result['valid'] and result['status'] == 'VALID':
                kept_files.append(path_str)
                continue

            file_path = Path(path_str)
            if backup:
                backup_dir = file_path.parent / "removed_models_backup"
                backup_dir.mkdir(exist_ok=True)
                backup_path = backup_dir / file_path.name
                counter = 1
                while backup_path.exists():
                    stem = file_path.stem
                    suffix = file_path.suffix
                    backup_path = backup_dir / f"{stem}_{counter}{suffix}"
                    counter += 1
                shutil.move(str(file_path), str(backup_path))
                removed_files.append(str(backup_path))
                status = "WARNING" if result['valid'] else "INVALID"
                self.log(f"📦 Moved {status} model to backup: {file_path.name} -> {backup_path.name}", "warning")
            else:
                file_path.unlink()
                removed_files.append(str(file_path))
                status = "WARNING" if result['valid'] else "INVALID"
                self.log(f"🗑️ Deleted {status} model: {file_path.name}", "warning")

        self.log(f"✅ Kept {len(kept_files)} valid models", "success")
        return len(removed_files), removed_files

    def get_problematic_models(self) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        warnings = []
        invalid = []

        for path, result in self.validation_results.items():
            if result['status'] == 'WARNING':
                warnings.append({
                    'path': path,
                    'filename': Path(path).name,
                    'model_type': result['info'].get('model_type', 'unknown'),
                    'symbol': result['info'].get('symbol', 'unknown'),
                    'timeframe': result['info'].get('timeframe', 'unknown'),
                    'trained_date': result['info'].get('trained_date', result['info']['modified']),
                    'error': f"{len(result['warnings'])} warnings: {result['warnings'][0][:50] if result['warnings'] else 'Unknown'}",
                    'size_mb': result['info'].get('size_mb', 0),
                    'warnings': result['warnings']
                })
            elif not result['valid'] or result['status'] == 'INVALID':
                invalid.append({
                    'path': path,
                    'filename': Path(path).name,
                    'model_type': result['info'].get('model_type', 'unknown'),
                    'symbol': result['info'].get('symbol', 'unknown'),
                    'timeframe': result['info'].get('timeframe', 'unknown'),
                    'trained_date': result['info'].get('trained_date', result['info']['modified']),
                    'error': result['errors'][0] if result['errors'] else 'Unknown error',
                    'size_mb': result['info'].get('size_mb', 0)
                })

        return warnings, invalid

class ModelValidatorGUI:
    """GUI for model validation"""

    def __init__(self, root):
        self.root = root
        self.root.title("🤖 AI Predict Model Validator")

        ConfigManager.restore_window_position(root, GUISettings.WINDOW_WIDTH, GUISettings.WINDOW_HEIGHT)
        self.root.configure(bg=GUISettings.COLOR_BG)
        self.root.resizable(False, False)

        self.root.bind('<Configure>', self._on_window_configure)
        self.root.protocol("WM_DELETE_WINDOW", self._on_closing)

        self.validator = ModelValidator(log_callback=self.log)
        self.validation_results = {}
        self.warning_models = []
        self.invalid_models = []
        self.duplicates = {}

        self.selected_folder = tk.StringVar()
        self.scan_recursive = tk.BooleanVar(value=True)
        self.validation_running = False
        self.scan_mode = tk.StringVar(value="validate")

        self.create_widgets()
        self.load_default_folder()

        self.log("🚀 AI Predict Model Validator Started", "info")
        self.log("📁 Select a folder containing ML models and click 'Start Scan'", "info")
        self.log(f"🔍 Supports 10 AI models: {', '.join(self.validator.supported_models)}", "info")
        self.log("🔍 New: Only valid models (green) are kept. Warnings and Invalid are removed.", "warning")

    def _on_window_configure(self, event):
        if event.widget == self.root:
            ConfigManager.save_window_position(self.root)

    def _on_closing(self):
        if messagebox.askokcancel("Exit", "Do you want to exit the application?"):
            ConfigManager.save_window_position(self.root)
            self.root.destroy()

    def log(self, message: str, level: str = "info"):
        if hasattr(self, 'log_text'):
            timestamp = datetime.now().strftime("%H:%M:%S")
            log_msg = f"[{timestamp}] {message}\n"
            self.log_text.insert(tk.END, log_msg)
            self.log_text.see(tk.END)

            if level == "error":
                self.log_text.tag_add("error", "end-2l", "end-1l")
            elif level == "warning":
                self.log_text.tag_add("warning", "end-2l", "end-1l")
            elif level == "success":
                self.log_text.tag_add("success", "end-2l", "end-1l")
            elif level == "duplicate":
                self.log_text.tag_add("duplicate", "end-2l", "end-1l")
            elif level == "info":
                self.log_text.tag_add("info", "end-2l", "end-1l")

            self.log_text.update()

    def create_widgets(self):
        main_container = tk.Frame(self.root, bg=GUISettings.COLOR_BG)
        main_container.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)

        header_frame = tk.Frame(main_container, bg=GUISettings.COLOR_FRAME_BG, relief=tk.RAISED, borderwidth=2)
        header_frame.pack(fill=tk.X, pady=2)

        tk.Label(
            header_frame,
            text="Copyright Su Nie | BSD-3C License | https://github.com/can87683",
            font=("Ubuntu", 14),
            bg=GUISettings.COLOR_FRAME_BG,
            fg="yellow"
        ).pack(pady=2)

        tk.Label(
            header_frame,
            text="🤖 AI PREDICT MODEL VALIDATOR",
            font=GUISettings.FONT_TITLE,
            bg=GUISettings.COLOR_FRAME_BG,
            fg=GUISettings.COLOR_FG
        ).pack(pady=2)

        tk.Label(
            header_frame,
            text="Supports: CatBoost, CNN, FinRL, GRU, LightGBM, LSTM, PatchTST, TCN, TFT, XGBoost",
            font=GUISettings.FONT_DEFAULT,
            bg=GUISettings.COLOR_FRAME_BG,
            fg=GUISettings.COLOR_FG_DIM
        ).pack(pady=2)

        folder_frame = tk.Frame(main_container, bg=GUISettings.COLOR_FRAME_BG, relief=tk.RAISED, borderwidth=2)
        folder_frame.pack(fill=tk.X, pady=2)

        folder_row = tk.Frame(folder_frame, bg=GUISettings.COLOR_FRAME_BG)
        folder_row.pack(fill=tk.X, padx=2, pady=2)

        tk.Label(
            folder_row,
            text="Models Folder:",
            bg=GUISettings.COLOR_FRAME_BG,
            fg=GUISettings.COLOR_FG,
            font=GUISettings.FONT_DEFAULT
        ).pack(side=tk.LEFT, padx=2)

        self.folder_entry = tk.Entry(
            folder_row,
            textvariable=self.selected_folder,
            bg="#444444",
            fg=GUISettings.COLOR_FG,
            font=GUISettings.FONT_DEFAULT,
            width=40
        )
        self.folder_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)

        tk.Button(
            folder_row,
            text="📁 Browse",
            command=self.browse_folder,
            bg=GUISettings.COLOR_BUTTON,
            fg=GUISettings.COLOR_FG,
            font=GUISettings.FONT_DEFAULT,
            width=8,
            bd=2
        ).pack(side=tk.LEFT, padx=2)

        options_row = tk.Frame(folder_frame, bg=GUISettings.COLOR_FRAME_BG)
        options_row.pack(fill=tk.X, padx=2, pady=2)

        tk.Checkbutton(
            options_row,
            text="Scan Subfolders",
            variable=self.scan_recursive,
            bg=GUISettings.COLOR_FRAME_BG,
            fg=GUISettings.COLOR_FG,
            selectcolor="#444444",
            font=GUISettings.FONT_DEFAULT
        ).pack(side=tk.LEFT, padx=2)

        self.auto_scan = tk.BooleanVar(value=True)
        tk.Checkbutton(
            options_row,
            text="Auto-scan on folder change",
            variable=self.auto_scan,
            bg=GUISettings.COLOR_FRAME_BG,
            fg=GUISettings.COLOR_FG,
            selectcolor="#444444",
            font=GUISettings.FONT_DEFAULT
        ).pack(side=tk.LEFT, padx=2)

        control_frame = tk.Frame(main_container, bg=GUISettings.COLOR_FRAME_BG, relief=tk.RAISED, borderwidth=2)
        control_frame.pack(fill=tk.X, pady=2)

        mode_row = tk.Frame(control_frame, bg=GUISettings.COLOR_FRAME_BG)
        mode_row.pack(padx=2, pady=2)

        tk.Label(
            mode_row,
            text="Scan Mode:",
            bg=GUISettings.COLOR_FRAME_BG,
            fg=GUISettings.COLOR_FG,
            font=GUISettings.FONT_DEFAULT
        ).pack(side=tk.LEFT, padx=2)

        modes = [
            ("🔍 Validate Models", "validate"),
            ("🔄 Find Duplicates", "duplicates")
        ]

        for text, mode in modes:
            tk.Radiobutton(
                mode_row,
                text=text,
                variable=self.scan_mode,
                value=mode,
                bg=GUISettings.COLOR_FRAME_BG,
                fg=GUISettings.COLOR_FG,
                selectcolor="#444444",
                font=GUISettings.FONT_DEFAULT
            ).pack(side=tk.LEFT, padx=2)

        button_row = tk.Frame(control_frame, bg=GUISettings.COLOR_FRAME_BG)
        button_row.pack(padx=2, pady=2)

        self.scan_button = tk.Button(
            button_row,
            text="🔍 Start Scan",
            command=self.start_scan,
            bg=GUISettings.COLOR_BUTTON_ACTIVE,
            fg=GUISettings.COLOR_FG,
            font=GUISettings.FONT_DEFAULT,
            width=15,
            bd=2
        )
        self.scan_button.pack(side=tk.LEFT, padx=2)

        self.stop_button = tk.Button(
            button_row,
            text="⏹️ Stop",
            command=self.stop_validation,
            bg=GUISettings.COLOR_BUTTON_STOP,
            fg=GUISettings.COLOR_FG,
            font=GUISettings.FONT_DEFAULT,
            width=10,
            state="disabled",
            bd=2
        )
        self.stop_button.pack(side=tk.LEFT, padx=2)

        self.remove_button = tk.Button(
            button_row,
            text="🗑️ Remove Warnings & Invalid",
            command=self.remove_non_valid_models,
            bg=GUISettings.COLOR_REMOVE,
            fg=GUISettings.COLOR_FG,
            font=GUISettings.FONT_DEFAULT,
            width=25,
            state="disabled",
            bd=2
        )
        self.remove_button.pack(side=tk.LEFT, padx=2)

        table_frame = tk.Frame(main_container, bg=GUISettings.COLOR_FRAME_BG, relief=tk.RAISED, borderwidth=2)
        table_frame.pack(fill=tk.BOTH, expand=True, pady=2)

        header_row = tk.Frame(table_frame, bg=GUISettings.COLOR_TABLE_HEADER)
        header_row.pack(fill=tk.X, padx=2, pady=2)

        self.table_headers = [
            ("Model", 200),
            ("Symbol/TF", 100),
            ("Trained Date", 120),
            ("Issues", 200)
        ]

        for col, (text, width) in enumerate(self.table_headers):
            header = tk.Label(
                header_row,
                text=text,
                bg=GUISettings.COLOR_TABLE_HEADER,
                fg=GUISettings.COLOR_FG,
                font=GUISettings.FONT_HEADER,
                relief=tk.RAISED,
                bd=2,
                width=width // 8
            )
            header.pack(side=tk.LEFT, fill=tk.X, expand=(col == len(self.table_headers) - 1), padx=2)

        table_container = tk.Frame(table_frame, bg=GUISettings.COLOR_FRAME_BG)
        table_container.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)

        canvas = tk.Canvas(table_container, bg=GUISettings.COLOR_FRAME_BG, highlightthickness=0)
        scrollbar = ttk.Scrollbar(table_container, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas, bg=GUISettings.COLOR_FRAME_BG)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        self.table_frame = scrollable_frame

        log_frame = tk.Frame(main_container, bg=GUISettings.COLOR_FRAME_BG, relief=tk.RAISED, borderwidth=2)
        log_frame.pack(fill=tk.BOTH, expand=True, pady=2)

        log_header = tk.Frame(log_frame, bg=GUISettings.COLOR_FRAME_BG)
        log_header.pack(fill=tk.X, padx=2, pady=2)

        tk.Label(
            log_header,
            text="📝 Validation Log",
            bg=GUISettings.COLOR_FRAME_BG,
            fg=GUISettings.COLOR_FG,
            font=GUISettings.FONT_HEADER
        ).pack(side=tk.LEFT, padx=2)

        log_buttons = tk.Frame(log_header, bg=GUISettings.COLOR_FRAME_BG)
        log_buttons.pack(side=tk.RIGHT)

        tk.Button(
            log_buttons,
            text="📋 Copy Selected",
            command=self.copy_selected_text,
            bg="#444444",
            fg=GUISettings.COLOR_FG,
            font=GUISettings.FONT_DEFAULT,
            width=12,
            bd=2
        ).pack(side=tk.LEFT, padx=2)

        tk.Button(
            log_buttons,
            text="📋 Copy All",
            command=self.copy_log_all,
            bg="#444444",
            fg=GUISettings.COLOR_FG,
            font=GUISettings.FONT_DEFAULT,
            width=8,
            bd=2
        ).pack(side=tk.LEFT, padx=2)

        tk.Button(
            log_buttons,
            text="🗑️ Clear",
            command=self.clear_log,
            bg="#444444",
            fg=GUISettings.COLOR_FG,
            font=GUISettings.FONT_DEFAULT,
            width=6,
            bd=2
        ).pack(side=tk.LEFT, padx=2)

        self.log_text = scrolledtext.ScrolledText(
            log_frame,
            height=10,
            bg=GUISettings.COLOR_LOG_BG,
            fg=GUISettings.COLOR_LOG_FG,
            font=GUISettings.FONT_MONO,
            wrap=tk.WORD,
            exportselection=True,
            selectbackground="#4a4a4a",
            selectforeground="white"
        )
        self.log_text.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)

        self.log_text.bind("<Button-1>", self._start_selection)
        self.log_text.bind("<B1-Motion>", self._update_selection)
        self.log_text.bind("<Control-c>", self.copy_selected_text_shortcut)
        self.log_text.bind("<Control-a>", self.select_all_shortcut)

        self.log_context_menu = tk.Menu(self.log_text, tearoff=0)
        self.log_context_menu.add_command(label="Copy", command=self.copy_selected_text)
        self.log_context_menu.add_command(label="Copy All", command=self.copy_log_all)
        self.log_context_menu.add_separator()
        self.log_context_menu.add_command(label="Select All", command=self.select_all_log)

        self.log_text.bind("<Button-3>", self._show_log_context_menu)

        self.log_text.tag_config("error", foreground="#ff4444")
        self.log_text.tag_config("warning", foreground="#ffaa00")
        self.log_text.tag_config("success", foreground="#00ff00")
        self.log_text.tag_config("info", foreground="#44aaff")
        self.log_text.tag_config("duplicate", foreground="#aa88ff")

        status_frame = tk.Frame(main_container, bg=GUISettings.COLOR_FRAME_BG)
        status_frame.pack(fill=tk.X, pady=2)

        self.status_label = tk.Label(
            status_frame,
            text="Ready. Select a folder and click 'Start Scan'",
            bg=GUISettings.COLOR_FRAME_BG,
            fg=GUISettings.COLOR_FG_DIM,
            font=GUISettings.FONT_DEFAULT
        )
        self.status_label.pack(side=tk.LEFT, padx=2, pady=2)

        self.progress_label = tk.Label(
            status_frame,
            text="",
            bg=GUISettings.COLOR_FRAME_BG,
            fg=GUISettings.COLOR_FG,
            font=GUISettings.FONT_DEFAULT
        )
        self.progress_label.pack(side=tk.RIGHT, padx=2, pady=2)

    def _start_selection(self, event):
        self.log_text.focus_set()
        self.log_text.tag_remove("sel", "1.0", "end")
        return None

    def _update_selection(self, event):
        pass

    def _show_log_context_menu(self, event):
        self.log_context_menu.tk_popup(event.x_root, event.y_root)
        self.log_context_menu.grab_release()

    def copy_selected_text_shortcut(self, event):
        self.copy_selected_text()
        return "break"

    def select_all_shortcut(self, event):
        self.select_all_log()
        return "break"

    def copy_selected_text(self):
        selected_text = self.log_text.get("sel.first", "sel.last")
        self.root.clipboard_clear()
        self.root.clipboard_append(selected_text)
        self.log("📋 Selected text copied to clipboard", "info")

    def copy_log_all(self):
        all_text = self.log_text.get("1.0", "end-1c")
        self.root.clipboard_clear()
        self.root.clipboard_append(all_text)
        self.log("📋 All log text copied to clipboard", "info")

    def select_all_log(self):
        self.log_text.tag_add("sel", "1.0", "end")
        self.log_text.mark_set("insert", "1.0")
        self.log_text.see("1.0")

    def scan_folder(self):
        folder_path = Path(self.selected_folder.get())

        model_count = 0
        for ext in self.validator.model_formats:
            if self.scan_recursive.get():
                model_count += len(list(folder_path.rglob(f"*{ext}")))
            else:
                model_count += len(list(folder_path.glob(f"*{ext}")))

        self.status_label.config(text=f"📊 Found {model_count} model files in {folder_path.name}")
        if model_count > 0:
            self.log(f"📊 Found {model_count} model files in folder", "info")
        else:
            self.log(f"⚠️ No model files found in folder", "warning")

    def load_default_folder(self):
        config = ConfigManager.load_config()

        if 'Settings' in config and 'last_folder' in config['Settings']:
            last_folder = config['Settings']['last_folder']
            if Path(last_folder).exists():
                self.selected_folder.set(last_folder)
                self.log(f"📁 Loaded last folder from config: {last_folder}", "info")
                if self.auto_scan.get():
                    self.scan_folder()
                return

        default_folders = [
            Path.home() / "ml_models",
            Path.home() / "models",
            Path.home() / "ai_predict_models",
            Path.cwd() / "models"
        ]

        for folder in default_folders:
            if folder.exists() and folder.is_dir():
                self.selected_folder.set(str(folder))
                self.log(f"📁 Loaded default folder: {folder}", "info")
                if self.auto_scan.get():
                    self.scan_folder()
                break

    def browse_folder(self):
        folder = filedialog.askdirectory(title="Select Models Folder", mustexist=True)
        if folder:
            self.selected_folder.set(folder)
            self.log(f"📁 Selected folder: {folder}", "info")

            config = ConfigManager.load_config()
            if 'Settings' not in config:
                config['Settings'] = {}
            config['Settings']['last_folder'] = folder
            ConfigManager.save_config(config)

            if self.auto_scan.get():
                self.scan_folder()

    def start_scan(self):
        if self.validation_running:
            self.log("⚠️ Scan already running", "warning")
            return

        folder_path = Path(self.selected_folder.get())

        self._clear_table()
        self.warning_models = []
        self.invalid_models = []
        self.duplicates = {}

        mode = self.scan_mode.get()
        mode_names = {
            "validate": "Validating models",
            "duplicates": "Finding duplicates"
        }

        self.log(f"🚀 Starting scan: {mode_names.get(mode, mode)}", "info")

        self.validation_running = True
        self.scan_button.config(state="disabled", bg=GUISettings.COLOR_BUTTON_DISABLED)
        self.stop_button.config(state="normal")
        self.remove_button.config(state="disabled")

        thread = threading.Thread(target=self._scan_thread, args=(folder_path, mode), daemon=True)
        thread.start()

    def _scan_thread(self, folder_path: Path, mode: str):
        self.root.after(0, lambda: self.status_label.config(text=f"🔍 Scanning {folder_path.name}..."))

        if mode == "validate":
            results = self.validator.scan_and_validate(folder_path, recursive=self.scan_recursive.get())
            self.validation_results = results
            self.warning_models, self.invalid_models = self.validator.get_problematic_models()
            self.root.after(0, lambda: self._update_validation_results())
        elif mode == "duplicates":
            self.duplicates = self.validator.find_duplicates(folder_path, recursive=self.scan_recursive.get())
            self.root.after(0, lambda: self._update_duplicate_results())

        self.root.after(0, self._scan_complete)

    def _update_validation_results(self):
        for model_info in self.warning_models:
            self._add_model_to_table(model_info, bg_color=GUISettings.COLOR_WARNING)

        for model_info in self.invalid_models:
            self._add_model_to_table(model_info, bg_color=GUISettings.COLOR_INVALID)

        total = len(self.validation_results)
        valid = total - len(self.warning_models) - len(self.invalid_models)
        problematic = len(self.warning_models) + len(self.invalid_models)

        self.status_label.config(text=f"✅ Validation complete: {valid} valid, {len(self.warning_models)} warnings, {len(self.invalid_models)} invalid")

        self.log(f"📊 Validation Results:", "info")
        self.log(f"   Total models: {total}", "info")
        self.log(f"   ✅ Valid (kept): {valid} ({valid/total*100:.1f}%)", "success")
        self.log(f"   ⚠️  Warnings (to remove): {len(self.warning_models)} ({len(self.warning_models)/total*100:.1f}%)", "warning")
        self.log(f"   ❌ Invalid (to remove): {len(self.invalid_models)} ({len(self.invalid_models)/total*100:.1f}%)", "error")

        if problematic > 0:
            self.log(f"🗑️ Found {problematic} models to remove (warnings + invalid)", "warning")
            self.remove_button.config(state="normal", bg=GUISettings.COLOR_REMOVE)
        else:
            self.log(f"✅ All models are valid - nothing to remove", "success")
            self.remove_button.config(state="disabled", bg=GUISettings.COLOR_BUTTON_DISABLED)

    def _update_duplicate_results(self):
        for file_hash, paths in self.duplicates.items():
            for path in paths:
                model_info = {
                    'filename': Path(path).name,
                    'symbol': 'DUPLICATE',
                    'timeframe': '',
                    'trained_date': '',
                    'error': f'Hash: {file_hash[:8]}... ({len(paths)} copies)',
                    'path': path
                }
                self._add_model_to_table(model_info, bg_color=GUISettings.COLOR_DUPLICATE)

        total_duplicates = sum(len(paths) for paths in self.duplicates.values())
        duplicate_groups = len(self.duplicates)

        self.status_label.config(text=f"🔄 Found {duplicate_groups} duplicate groups ({total_duplicates} files)")

        self.log(f"📊 Duplicate Detection Results:", "duplicate")
        self.log(f"   Duplicate groups: {duplicate_groups}", "duplicate")
        self.log(f"   Total duplicate files: {total_duplicates}", "duplicate")

        for hash_val, paths in self.duplicates.items():
            self.log(f"   🔑 Hash {hash_val[:8]}...: {len(paths)} copies", "duplicate")
            for path in paths[:3]:
                self.log(f"      📄 {Path(path).name}", "duplicate")
            if len(paths) > 3:
                self.log(f"      ... and {len(paths) - 3} more", "duplicate")

        self.remove_button.config(state="disabled")

    def _add_model_to_table(self, model_info: Dict[str, Any], bg_color: str = GUISettings.COLOR_INVALID):
        row_frame = tk.Frame(self.table_frame, bg=GUISettings.COLOR_FRAME_BG)
        row_frame.pack(fill=tk.X, padx=2, pady=2)

        model_name = model_info['filename']
        if len(model_name) > 30:
            model_name = model_name[:27] + "..."

        tk.Label(
            row_frame,
            text=model_name,
            bg=bg_color,
            fg="white",
            font=GUISettings.FONT_MONO,
            relief=tk.SUNKEN,
            bd=2,
            anchor="w",
            padx=2
        ).pack(side=tk.LEFT, fill=tk.Y, padx=2)

        symbol_tf = f"{model_info.get('symbol', '')}/{model_info.get('timeframe', '')}"
        tk.Label(
            row_frame,
            text=symbol_tf,
            bg=bg_color,
            fg="white",
            font=GUISettings.FONT_DEFAULT,
            relief=tk.SUNKEN,
            bd=2,
            anchor="center",
            padx=2
        ).pack(side=tk.LEFT, fill=tk.Y, padx=2)

        trained_date = model_info.get('trained_date', '')
        if isinstance(trained_date, datetime):
            date_str = trained_date.strftime("%Y-%m-%d")
        else:
            date_str = str(trained_date)[:10] if trained_date else "Unknown"

        tk.Label(
            row_frame,
            text=date_str,
            bg=bg_color,
            fg="white",
            font=GUISettings.FONT_DEFAULT,
            relief=tk.SUNKEN,
            bd=2,
            anchor="center",
            padx=2
        ).pack(side=tk.LEFT, fill=tk.Y, padx=2)

        error = model_info.get('error', 'Unknown')
        if len(error) > 50:
            error = error[:47] + "..."

        tk.Label(
            row_frame,
            text=error,
            bg=bg_color,
            fg="white",
            font=GUISettings.FONT_DEFAULT,
            relief=tk.SUNKEN,
            bd=2,
            anchor="w",
            padx=2,
            wraplength=200
        ).pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        row_frame.model_info = model_info
        row_frame.bind("<Double-Button-1>", lambda e, info=model_info: self.show_model_details(info))

    def _clear_table(self):
        for widget in self.table_frame.winfo_children():
            widget.destroy()

    def show_model_details(self, model_info: Dict[str, Any]):
        details_window = tk.Toplevel(self.root)
        details_window.title(f"Model Details: {model_info['filename']}")
        details_window.geometry("500x400")
        details_window.configure(bg=GUISettings.COLOR_BG)

        notebook = ttk.Notebook(details_window)
        notebook.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)

        info_frame = tk.Frame(notebook, bg=GUISettings.COLOR_FRAME_BG)
        notebook.add(info_frame, text="📋 Info")

        info_canvas = tk.Canvas(info_frame, bg=GUISettings.COLOR_FRAME_BG, highlightthickness=0)
        info_scrollbar = ttk.Scrollbar(info_frame, orient="vertical", command=info_canvas.yview)
        info_content = tk.Frame(info_canvas, bg=GUISettings.COLOR_FRAME_BG)

        info_content.bind(
            "<Configure>",
            lambda e: info_canvas.configure(scrollregion=info_canvas.bbox("all"))
        )

        info_canvas.create_window((0, 0), window=info_content, anchor="nw")
        info_canvas.configure(yscrollcommand=info_scrollbar.set)

        info_canvas.pack(side="left", fill="both", expand=True)
        info_scrollbar.pack(side="right", fill="y")

        row = 0
        fields = [
            ("Filename", model_info['filename']),
            ("Path", model_info.get('path', 'Unknown')),
            ("Model Type", model_info.get('model_type', 'Unknown')),
            ("Symbol", model_info.get('symbol', 'Unknown')),
            ("Timeframe", model_info.get('timeframe', 'Unknown')),
            ("Trained Date", str(model_info.get('trained_date', 'Unknown'))),
            ("Size", f"{model_info.get('size_mb', 0):.2f} MB"),
            ("Issue", model_info.get('error', 'Unknown'))
        ]

        for field_name, field_value in fields:
            tk.Label(
                info_content,
                text=f"{field_name}:",
                bg=GUISettings.COLOR_FRAME_BG,
                fg=GUISettings.COLOR_FG,
                font=GUISettings.FONT_HEADER,
                anchor="w"
            ).grid(row=row, column=0, sticky="w", padx=2, pady=2)

            value_label = tk.Label(
                info_content,
                text=field_value,
                bg=GUISettings.COLOR_FRAME_BG,
                fg=GUISettings.COLOR_FG_DIM,
                font=GUISettings.FONT_MONO,
                anchor="w",
                wraplength=400,
                justify="left"
            )
            value_label.grid(row=row, column=1, sticky="w", padx=2, pady=2)
            row += 1

        buttons_frame = tk.Frame(details_window, bg=GUISettings.COLOR_BG)
        buttons_frame.pack(fill=tk.X, padx=2, pady=2)

        tk.Button(
            buttons_frame,
            text="Close",
            command=details_window.destroy,
            bg="#444444",
            fg="white",
            width=10,
            bd=2
        ).pack(pady=2)

    def remove_non_valid_models(self):
        if not self.validation_results:
            messagebox.showwarning("No Data", "No validation results. Run validation first.")
            return

        problematic_count = len(self.warning_models) + len(self.invalid_models)
        if problematic_count == 0:
            messagebox.showinfo("No Models to Remove", "No warnings or invalid models found to remove.")
            return

        result = messagebox.askyesnocancel(
            "Remove Non-Valid Models",
            f"Found {problematic_count} models to remove:\n"
            f"   ⚠️  Warnings: {len(self.warning_models)}\n"
            f"   ❌ Invalid: {len(self.invalid_models)}\n\n"
            f"Only valid models (green) will be kept.\n\n"
            "🗑️  YES: Move to backup folder (recommended)\n"
            "❌  NO: Permanently delete\n"
            "🔙  CANCEL: Abort operation\n\n"
            "Backup folder will be created as 'removed_models_backup' in each directory."
        )

        if result is None:
            self.log("❌ Removal cancelled by user", "warning")
            return

        backup = result
        self.log(f"🚀 {'Moving' if backup else 'Deleting'} non-valid models...", "info")
        count, removed = self.validator.remove_non_valid_models(self.validation_results, backup=backup)

        if backup:
            self.log(f"📦 Successfully moved {count} non-valid models to backup folders", "success")
            if removed:
                backup_path = Path(removed[0]).parent.parent / "removed_models_backup"
                self.log(f"📁 Backup folders created in each directory", "info")
        else:
            self.log(f"🗑️ Successfully deleted {count} non-valid models", "success")

        self._clear_table()
        self.warning_models = []
        self.invalid_models = []
        self.validation_results = {}
        self.remove_button.config(state="disabled", bg=GUISettings.COLOR_BUTTON_DISABLED)

    def stop_validation(self):
        self.validation_running = False
        self.log("⏹️ Scan stopped by user", "warning")
        self._scan_complete()

    def _scan_complete(self):
        self.validation_running = False
        self.scan_button.config(state="normal", bg=GUISettings.COLOR_BUTTON_ACTIVE)
        self.stop_button.config(state="disabled")

    def clear_log(self):
        self.log_text.delete(1.0, tk.END)
        self.log("🗑️ Log cleared", "info")

if __name__ == "__main__":
    root = tk.Tk()
    app = ModelValidatorGUI(root)
    root.protocol("WM_DELETE_WINDOW", app._on_closing)
    root.mainloop()