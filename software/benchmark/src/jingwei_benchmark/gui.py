"""Native Qt desktop interface for the shared proteomics benchmark backend."""

import os
from pathlib import Path
import sys
import threading

import numpy as np
import pandas as pd
from PySide6.QtCore import Qt, QObject, Signal, QRunnable, QThreadPool, QTimer, QUrl, QLockFile, QAbstractTableModel, QStandardPaths
from PySide6.QtGui import QColor, QDesktopServices, QFontDatabase, QFont
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QFormLayout, QLabel,
    QComboBox, QLineEdit, QSpinBox, QDoubleSpinBox, QCheckBox, QListWidget,
    QListWidgetItem, QTableWidgetItem, QPushButton, QFileDialog, QMessageBox, QDialog,
    QTableView, QHeaderView, QDialogButtonBox,
)

from .gui_jobs import JobStore, TERMINAL, inspect_input
from .gui_parameters import LABELS, PARAMETERS, NEURAL
from .gui_layout import WorkbenchLayout
from .gui_i18n import translate
from .gui_session import SessionState

STATE_LABELS = {"running": "运行中", "cancelling": "正在停止", "completed": "已完成",
                "incomplete": "部分完成", "failed": "失败", "cancelled": "已停止", "interrupted": "已中断",
                "skipped": "跳过", "timeout": "超时"}

NUMERIC_COLUMNS = {"repeat", "epochs_trained", "best_epoch", "MSE", "RMSE", "MAE",
                   "NRMSE", "Pearson", "Spearman", "W2", "W2_n_samples",
                   "W2_n_features", "W2_n_evaluated", "time_s"}


def evaluation_scale_name(config):
    scale = config.get("scale", "raw")
    transform = config.get("transform", "auto")
    return "log2" if scale == "log2" or transform == "log2" or (scale == "raw" and transform == "auto") else "raw"


def text(label, style=None, wrap=True):
    widget = QLabel(label)
    widget.setTextFormat(Qt.PlainText)
    widget.setWordWrap(wrap)
    if style:
        widget.setObjectName(style)
    return widget


def combo(items):
    widget = QComboBox()
    for label, data in items:
        widget.addItem(label, data)
    return widget


def control(spec):
    label, default, low, high = spec
    if isinstance(default, list):
        return QLineEdit(", ".join(map(str, default)))
    widget = QSpinBox() if isinstance(default, int) else QDoubleSpinBox()
    widget.setRange(low, high)
    if isinstance(widget, QDoubleSpinBox):
        widget.setDecimals(6 if low is not None and 0 < low < 0.0001 else 4)
        widget.setSingleStep(0.01 if default <= 1 else 0.1)
    widget.setValue(default)
    return widget


def value(widget):
    if isinstance(widget, QComboBox):
        return widget.currentData()
    if isinstance(widget, QCheckBox):
        return widget.isChecked()
    if isinstance(widget, (QSpinBox, QDoubleSpinBox)):
        return widget.value()
    return widget.text().strip()


def set_value(widget, data):
    previous = widget.blockSignals(True)
    if isinstance(widget, QComboBox):
        index = widget.findData(data)
        if index >= 0:
            widget.setCurrentIndex(index)
    elif isinstance(widget, QCheckBox):
        widget.setChecked(bool(data))
    elif isinstance(widget, (QSpinBox, QDoubleSpinBox)):
        widget.setValue(data)
    else:
        widget.setText(", ".join(map(str, data)) if isinstance(data, list) else str(data or ""))
    widget.blockSignals(previous)


def dimensions(widget):
    try:
        result = [int(part.strip()) for part in widget.text().split(",")]
        if not result or any(part <= 0 for part in result):
            raise ValueError()
        return result
    except ValueError:
        raise ValueError("隐藏层维度请输入用逗号分隔的正整数，例如 256, 128")


class FractionControl(QDoubleSpinBox):
    """Display a percentage while the controller continues to use fractions."""
    def __init__(self, default, low, high):
        super().__init__()
        self.setDecimals(2)
        self.setRange(low * 100, high * 100)
        self.setSuffix(" %")
        self.setSingleStep(1)
        self.setValue(default)

    def value(self):
        return super().value() / 100

    def setValue(self, fraction):
        super().setValue(fraction * 100)


class TaskSignals(QObject):
    done = Signal(object)
    error = Signal(str)


class BackgroundTask(QRunnable):
    def __init__(self, function, *args):
        super().__init__()
        self.function, self.args = function, args
        self.signals = TaskSignals()

    def run(self):
        try:
            self.signals.done.emit(self.function(*self.args))
        except Exception as exc:
            self.signals.error.emit(str(exc))


class ParameterDialog(QDialog):
    def __init__(self, name, overrides, parent):
        super().__init__(parent)
        self.t = parent.t
        self.setWindowTitle(self.t(LABELS.get(name, name)) + " · " + self.t("参数"))
        self.resize(450, 320)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 14, 16, 14)
        layout.setSpacing(12)
        self.fields = {}
        form = QFormLayout()
        form.setVerticalSpacing(8)
        form.setHorizontalSpacing(12)
        from .gui_parameters import valid_overrides
        overrides = valid_overrides(name, overrides)
        for key, spec in PARAMETERS.get(name, {}).items():
            widget = control(spec)
            if key in overrides:
                set_value(widget, overrides[key])
            self.fields[key] = widget
            form.addRow(self.t(spec[0]), widget)
        if not self.fields:
            layout.addWidget(text(self.t("此方法没有可调整的参数。")))
        layout.addLayout(form)
        layout.addStretch()
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel | QDialogButtonBox.Reset)
        for flag, label in ((QDialogButtonBox.Ok, "确定"), (QDialogButtonBox.Cancel, "取消"), (QDialogButtonBox.Reset, "重置")):
            buttons.button(flag).setText(self.t(label))
        buttons.accepted.connect(self._accept)
        buttons.rejected.connect(self.reject)
        buttons.button(QDialogButtonBox.Reset).clicked.connect(lambda: self._reset(name))
        layout.addWidget(buttons)

    def _reset(self, name):
        for key, widget in self.fields.items():
            set_value(widget, PARAMETERS[name][key][1])

    def values(self):
        return {key: dimensions(widget) if key == "hidden_dims" else value(widget)
                for key, widget in self.fields.items()}

    def _accept(self):
        try:
            self.values()
            self.accept()
        except ValueError as exc:
            QMessageBox.warning(self, self.t("参数无效"), self.t(str(exc)))


class RunTableModel(QAbstractTableModel):
    def __init__(self, translator=lambda source: source):
        super().__init__()
        self.t = translator
        self.rows, self.columns = [], []

    def replace(self, rows, columns):
        self.beginResetModel()
        self.rows, self.columns = rows, columns
        self.endResetModel()

    def rowCount(self, parent=None):
        return 0 if parent is not None and parent.isValid() else len(self.rows)

    def columnCount(self, parent=None):
        return 0 if parent is not None and parent.isValid() else len(self.columns)

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if role == Qt.DisplayRole:
            if orientation != Qt.Horizontal:
                return str(section + 1)
            key = self.columns[section]
            labels = {"repeat": "重复", "method": "方法名称", "status": "状态", "time_s": "耗时（秒）",
                      "convergence_status": "收敛状态", "epochs_trained": "实际轮数", "best_epoch": "最佳轮数",
                      "W2_n_samples": "W2 样本数", "W2_n_features": "W2 蛋白数", "W2_n_evaluated": "W2 评测数"}
            return self.t(labels.get(key, key))
        if role == Qt.TextAlignmentRole and orientation == Qt.Horizontal:
            key = self.columns[section]
            return int(Qt.AlignRight | Qt.AlignVCenter) if key in NUMERIC_COLUMNS else int(Qt.AlignLeft | Qt.AlignVCenter)

    def data(self, index, role=Qt.DisplayRole):
        if not index.isValid():
            return None
        row, key = self.rows[index.row()], self.columns[index.column()]
        cell = row.get(key)
        if role == Qt.DisplayRole:
            if key == "status":
                return self.t(STATE_LABELS.get(cell, cell or "—"))
            if key == "convergence_status":
                return self.t({"validation_plateau": "验证已稳定", "max_epochs": "达到训练上限",
                    "fixed_budget": "固定预算", "insufficient_validation": "验证数据不足",
                    "not_applicable": "不适用", "not_assessed": "未评估"}.get(cell, cell or "—"))
            if cell is None:
                return "—"
            if isinstance(cell, float):
                return f"{cell:.2f}" if key == "time_s" else f"{cell:.4f}"
            return str(cell)
        if role == Qt.TextAlignmentRole:
            return int(Qt.AlignRight | Qt.AlignVCenter) if key in NUMERIC_COLUMNS else int(Qt.AlignLeft | Qt.AlignVCenter)
        if role == Qt.ToolTipRole:
            notes = []
            if row.get("error"):
                notes.append(str(row["error"]))
            if row.get("W2_note"):
                notes.append(str(row["W2_note"]))
            if row.get("epochs_trained") is not None:
                notes.append(f"{self.t('实际轮数')}: {row['epochs_trained']}")
            if row.get("best_epoch") is not None:
                notes.append(f"{self.t('最佳轮数')}: {row['best_epoch']}")
            if row.get("W2_n_samples") is not None and row.get("W2_n_features") is not None:
                notes.append(f"{self.t('W2 样本数')}: {row['W2_n_samples']} · {self.t('W2 蛋白数')}: {row['W2_n_features']}")
            return "\n".join(notes)
        if role == Qt.ForegroundRole and key == "status":
            if cell == "completed":
                return QColor("#15803D")
            if cell in ("running", "cancelling"):
                return QColor("#1D5FA7")
            return QColor("#B91C1C")


class RunTable(QTableView):
    def rowCount(self):
        return self.model().rowCount()

    def currentRow(self):
        return self.currentIndex().row()

    def selectedItems(self):
        return self.selectionModel().selectedIndexes()


class MetricPlot(QWidget):
    def __init__(self, translator=lambda source: source):
        super().__init__()
        self.t = translator
        from matplotlib import rcParams
        rcParams["font.sans-serif"] = ["Microsoft YaHei UI", "Microsoft YaHei", "DejaVu Sans"]
        rcParams["axes.unicode_minus"] = False
        # Select the native Qt binding, independent of PIMMS worker plotting.
        os.environ["QT_API"] = "pyside6"
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
        self.figure = Figure(figsize=(8, 4), dpi=100, layout="constrained", facecolor="white")
        self.canvas = FigureCanvasQTAgg(self.figure)
        plot_layout = QVBoxLayout(self)
        plot_layout.setContentsMargins(0, 0, 0, 0)
        plot_layout.addWidget(self.canvas)
        self.source = pd.DataFrame()
        self.provenance = {}
        self.empty(self.t("暂无可比较数据"))

    def empty(self, message):
        self.setFixedHeight(280)
        self.figure.clear()
        axes = self.figure.add_subplot()
        axes.axis("off")
        # Qt label carries localized explanatory text; plot keeps portable fonts.
        axes.text(0.5, 0.5, self.t("暂无可比较数据"), ha="center", va="center", color="#64748B", fontsize=11)
        self.canvas.draw_idle()
        self.setToolTip(message)
        self.source = pd.DataFrame()
        self.provenance = {}

    def render(self, frame, metric, scale, paired, job_id):
        if frame.empty or metric not in frame:
            self.empty(self.t("暂无可比较数据"))
            return 0
        frame = frame[frame.status == "completed"].copy()
        frame[metric] = pd.to_numeric(frame[metric], errors="coerce")
        frame = frame[np.isfinite(frame[metric])]
        if frame.empty:
            self.empty(self.t("W2 比较样本的多维蛋白分布；仅使用所选蛋白上原始完整的参考样本。未定义值留空。"))
            return 0
        stats = frame.groupby("method")[metric].agg(["mean", "std", "count"])
        stats = stats.sort_values("mean", ascending=metric not in ("Pearson", "Spearman"))
        self.setFixedHeight(max(280, len(stats) * 32 + 120))
        self.figure.clear()
        axes = self.figure.add_subplot()
        positions = np.arange(len(stats))
        colors = ["#1D5FA7" if name.startswith("jingwei") else "#64748B" for name in stats.index]
        bars = axes.barh(positions, stats["mean"], color=colors, height=0.58)
        multi_repeats = bool((stats["count"] > 1).any())
        show_counts = multi_repeats or stats["count"].nunique() > 1
        for i, row in enumerate(stats.itertuples()):
            if np.isfinite(row.std):
                axes.errorbar(row.mean, i, xerr=row.std, fmt="none", color="#0F172A", capsize=3, lw=1)
        tick_labels = [f"{name}  (n={int(count)})" if show_counts else str(name)
                       for name, count in zip(stats.index, stats["count"])]
        axes.set_yticks(positions, tick_labels, fontsize=9.5, color="#1E293B")
        axes.invert_yaxis()
        axes.axvline(0, color="#CBD5E1", linewidth=0.8)
        axes.set_axisbelow(True)
        axes.grid(axis="x", color="#F1F5F9", linewidth=0.8)
        for side in ("top", "right", "left"):
            axes.spines[side].set_visible(False)
        axes.spines["bottom"].set_color("#CBD5E1")
        axes.tick_params(axis="x", colors="#475569", labelsize=9)
        axes.tick_params(axis="y", length=0)
        max_val = float((stats["mean"] + stats["std"].fillna(0)).max())
        min_val = float((stats["mean"] - stats["std"].fillna(0)).min())
        if metric not in ("Pearson", "Spearman"):
            axes.set_xlim(left=0, right=max_val * 1.14 if max_val > 0 else 1.0)
        else:
            span = max(abs(max_val), abs(min_val), 0.1)
            axes.set_xlim(left=min(0.0, min_val - 0.08 * span), right=max_val + 0.14 * span)
        if not multi_repeats:
            axes.bar_label(bars, fmt="%.4g", padding=5, fontsize=8.5, color="#334155")
        scale_display = self.t("原始丰度") if scale == "raw" else (scale if scale and scale != "working" else "log2")
        units = (self.t("秒") if metric == "time_s"
                 else f"{scale_display}²" if metric == "MSE"
                 else scale_display if metric in ("MAE", "RMSE", "W2")
                 else self.t("无量纲"))
        axes.set_xlabel(f"{metric} ({units})", fontsize=9.5, color="#334155")
        axes.set_title(self.t("共同完整重复" if paired else "各方法成功运行"), loc="left", fontsize=11, color="#0F172A", pad=8)
        if multi_repeats:
            self.figure.supxlabel(self.t("均值 ± 标准差（n ≥ 2）"), fontsize=8, color="#64748B")
        self.canvas.draw_idle()
        self.source = frame
        self.provenance = {"job_id": job_id, "metric": metric, "scale": scale,
                           "aggregation": "mean across repeat runs", "uncertainty": "sample SD; absent when n=1",
                           "paired_complete_repeats_only": paired, "nonfinite_metric_rows_excluded": True}
        return len(stats)


class BenchmarkWindow(SessionState, WorkbenchLayout, QMainWindow):
    def __init__(self, workspace=None, *, load_default=True, startup=False):
        super().__init__()
        self.store = JobStore(workspace)
        self.default_output_root = self.store.root / "outputs" if workspace is not None else Path(QStandardPaths.writableLocation(QStandardPaths.DocumentsLocation)) / "jingwei"
        self.pool = QThreadPool.globalInstance()
        self._tasks = set()
        self.preview_data = None
        self.preview_signature = None
        self.feature_selection = None
        self.method_parameters = {}
        self.backend_states = {}
        self.selected_job = None
        self._last_result_digest = None
        self._inspect_generation = 0
        self._submitting = False
        self._closed = False
        self._pending_features = None
        self._environment_generation = 0
        self._environment_cancel = threading.Event()
        self._environment_cache_hit = False
        self._environment_probe_count = 0
        self._environment_retry_used = False
        self._startup_pending = startup
        self._environment_error = None
        self._session_jobs = set()
        self._status_timer = QTimer(self)
        self._status_timer.setSingleShot(True)
        self._status_timer.timeout.connect(lambda: self.status_notice.clear())
        self.setup_ui()
        restored_input = self.restore_session(load_input=load_default)
        self.timer = QTimer(self)
        self.timer.setInterval(900)
        self.timer.timeout.connect(self.refresh_jobs)
        self.timer.start()
        self.refresh_jobs()
        self.refresh_environment()
        if load_default and not restored_input:
            from .runtime import default_input_path
            if path := default_input_path():
                self.set_input(path)
        self.connect_session()

    def flash_status(self, message, duration_ms=5000):
        self.status_notice.setText(message)
        self._status_timer.start(duration_ms)

    def background(self, function, done, *args, error=None):
        task = BackgroundTask(function, *args)
        self._tasks.add(task)
        def success(result):
            self._tasks.discard(task)
            if not self._closed:
                done(result)
        def failure(message):
            self._tasks.discard(task)
            if not self._closed:
                (error or self.show_error)(message)
        task.signals.done.connect(success)
        task.signals.error.connect(failure)
        self.pool.start(task)

    def show_error(self, message):
        box = QMessageBox(QMessageBox.Warning, self.t("无法完成操作"), self.t(message), QMessageBox.Ok, self)
        box.button(QMessageBox.Ok).setText(self.t("确定"))
        box.exec()

    def choose_backend(self, entry, directory):
        path = QFileDialog.getExistingDirectory(self, self.t("选择目录")) if directory else QFileDialog.getOpenFileName(self, self.t("选择 Rscript"))[0]
        if path:
            entry.setText(path)

    def refresh_environment(self, *, force=False):
        from .gui_cache import environment_identity, load_environment, save_environment
        self._environment_cancel.set()
        self._environment_cancel = threading.Event()
        self._environment_error = None
        cancel = self._environment_cancel
        self._environment_generation += 1
        generation = self._environment_generation
        options = {key: value(self.fields[key]) or None for key in ("rscript", "r_library", "naguider_source", "dreamai_source", "mice_backend")}
        identity = environment_identity(options)
        cache = self.store.root / "environment.json"
        result = None if force else load_environment(cache, identity)
        self._environment_cache_hit = result is not None
        if result is not None:
            self.apply_environment(result)
            return
        if force:
            try:
                cache.unlink(missing_ok=True)
            except OSError:
                pass
        self.backend_states = {}
        self._environment_checking = True
        self.update_environment_labels()
        self.update_run_state()
        self._environment_probe_count += 1
        def inspect():
            # Probe torch and R in an isolated process, never in a Qt thread.
            from .gui_environment import inspect_environment
            return inspect_environment(options, cancel_event=cancel)
        def done(result):
            if generation != self._environment_generation:
                return
            try:
                save_environment(cache, identity, result)
            except OSError as exc:
                self.flash_status(str(exc))
            self.apply_environment(result)
        def failed(message):
            if generation != self._environment_generation:
                return
            self._environment_checking = False
            self._environment_error = message
            self.update_environment_labels()
            if not self._startup_pending:
                self.show_error(message)
        self.background(inspect, done, error=failed)

    def apply_environment(self, result):
        self._environment_base_result = result
        states, gpu, self._gpu_name = result
        self.backend_states = {name: dict(state) for name, state in states.items()}
        self._environment_checking = False
        prev_tree = self.method_tree.blockSignals(True)
        for name, box in self.method_checks.items():
            available = self.backend_states[name]["available"]
            box.setEnabled(available or box.isChecked())
        self.method_tree.blockSignals(prev_tree)
        self.update_method_count()
        self.update_environment_labels()
        if "gpu_id" in self.fields:
            index = self.fields["device"].findData("cuda")
            self.fields["device"].model().item(index).setEnabled(gpu)
            if not gpu and value(self.fields["device"]) == "cuda":
                set_value(self.fields["device"], "cpu")
            self.device_changed()
        self.update_run_state()

    def update_run_state(self):
        busy = self.configuration_locked()
        self.configuration_panel.setEnabled(not busy)
        for index in range(self.advanced_tabs.count()):
            self.advanced_tabs.widget(index).setEnabled(not busy)
        self.clone_button.setEnabled(bool(self.selected_job) and not busy)
        self.start_button.setEnabled(self.preview_data is not None and bool(self.backend_states) and not self._environment_checking and not self._inspecting and not busy)

    def configuration_locked(self):
        return self._submitting or bool(self.store.active())

    def device_changed(self):
        if "gpu_id" in self.fields:
            self.fields["gpu_id"].setEnabled(value(self.fields["device"]) != "cpu")

    def training_policy_changed(self):
        converge = value(self.fields["training_policy"]) == "convergence"
        for key in ("patience", "min_epochs", "min_delta", "validation_fraction"):
            if key in self.fields:
                self.fields[key].setEnabled(converge)

    def mode_changed(self):
        benchmark = value(self.fields["command"]) == "benchmark"
        set_value(self.fields["output_scale"], "working" if benchmark else "input")
        self.update_mask_controls()

    def invalidate_preview(self, *args):
        self._pending_features = None
        self._inspect_generation += 1
        self._inspecting = False
        self.preview_data = None
        self.preview_signature = None
        self.feature_selection = None
        self.start_button.setEnabled(False)
        self.update_data_labels()

    def choose_input(self):
        path = QFileDialog.getOpenFileName(self, self.t("导入蛋白质组数据"), "", self.t("蛋白质组数据 (*.csv *.h5ad)"))[0]
        if path:
            self.set_input(path)

    def choose_output(self):
        path = QFileDialog.getExistingDirectory(self, self.t("选择输出目录"), self.output_root.text())
        if path:
            self.output_root.setText(path)

    def choose_task(self):
        path = QFileDialog.getExistingDirectory(self, self.t("打开任务目录"), self.output_root.text())
        if path:
            self.open_task(path)

    def open_task(self, path):
        if self.store.active() or self._submitting:
            self.show_error(self.t("请先完成或停止当前任务"))
            return
        try:
            job = self.store.attach(path)
        except (ValueError, OSError, KeyError, TypeError) as exc:
            self.show_error(str(exc))
            return
        self.selected_job = job
        self._last_result_digest = None
        self.refresh_jobs()

    def set_input(self, path, layer_name=None):
        self.invalidate_preview()
        layer = self.fields["layer"]
        layer.blockSignals(True)
        layer.clear()
        layer.addItem(self.t("X（默认矩阵）"), None)
        if layer_name:
            layer.addItem(layer_name, layer_name)
            layer.setCurrentIndex(1)
        layer.blockSignals(False)
        self.input_path.setText(str(path))
        self.inspect_data()

    def load_demo(self):
        path = self.store.root / "demo.csv"
        rng = np.random.default_rng(42)
        data = rng.lognormal(8, 0.65, (16, 80))
        data[rng.random(data.shape) < 0.025] = np.nan
        # Preserve several completely observed samples for sample-distribution W2.
        data[:4] = rng.lognormal(8, 0.65, (4, 80))
        pd.DataFrame(data, index=[f"Sample_{i+1:02d}" for i in range(16)], columns=[f"Protein_{i+1:03d}" for i in range(80)]).to_csv(path)
        self.set_input(path)

    def load_example(self):
        from .runtime import default_input_path
        if path := default_input_path():
            for key, setting in {"orientation": "samples-by-features", "header": "auto",
                                 "index_column": "auto", "missing_values": "nan", "scale": "raw", "transform": "auto"}.items():
                set_value(self.fields[key], setting)
            self.set_input(path)
        else:
            self.load_demo()

    def input_options(self):
        keys = ("orientation", "layer", "index_column", "header", "missing_values", "scale", "transform", "max_dense_mb")
        return {key: value(self.fields[key]) for key in keys}

    def signature(self):
        raw = self.input_path.text().strip()
        if not raw:
            raise FileNotFoundError(self.t("请选择有效的 CSV 或 h5ad 文件"))
        path = Path(raw).resolve()
        if not path.is_file():
            raise FileNotFoundError(self.t("请选择有效的 CSV 或 h5ad 文件"))
        stat = path.stat()
        return str(path), stat.st_size, stat.st_mtime_ns, self.input_options()

    def inspect_data(self):
        try:
            signature = self.signature()
        except OSError as exc:
            self.show_error(str(exc))
            return
        self._inspect_generation += 1
        generation = self._inspect_generation
        self._inspecting = True
        self.update_run_state()
        self.update_data_labels()
        def done(data):
            if generation != self._inspect_generation:
                return
            self.preview_data = data
            self.preview_signature = signature
            if self._pending_features and self._pending_features[0] == signature:
                available = set(data["feature_ids"])
                selected = [name for name in self._pending_features[1] if name in available]
                self.feature_selection = None if (not selected or len(selected) == len(data["feature_ids"])) else selected
                self._pending_features = None
            self._inspecting = False
            self.update_run_state()
            self.update_data_labels()
            preview = data["preview"]
            self.preview_table.setRowCount(len(preview))
            self.preview_table.setColumnCount(preview.shape[1])
            self.preview_table.setHorizontalHeaderLabels(list(map(str, preview.columns)))
            self.preview_table.setVerticalHeaderLabels(list(map(str, preview.index)))
            for i in range(len(preview)):
                for j in range(preview.shape[1]):
                    cell = preview.iloc[i, j]
                    item = QTableWidgetItem("—" if not np.isfinite(cell) else f"{cell:.4g}")
                    if not np.isfinite(cell):
                        item.setBackground(QColor("#FFF0E6"))
                    self.preview_table.setItem(i, j, item)
            layer = self.fields["layer"]
            current = layer.currentData()
            layer.blockSignals(True)
            layer.clear()
            layer.addItem(self.t("X（默认矩阵）"), None)
            for name in data["layers"]:
                layer.addItem(name, name)
            set_value(layer, current)
            layer.blockSignals(False)
        def failure(message):
            if generation != self._inspect_generation:
                return
            self._inspecting = False
            self.preview_data = None
            self.preview_signature = None
            self.update_run_state()
            self.data_summary.setText(self.t("数据检查失败：{error}", error=message))
            self.show_error(message)
        self.background(inspect_input, done, signature[0], signature[3], error=failure)

    def select_methods(self, names):
        names = set(names if names is not None else [name for name, state in self.backend_states.items() if state["available"]])
        prev_tree = self.method_tree.blockSignals(True)
        for name, box in self.method_checks.items():
            available = self.backend_states.get(name, {}).get("available", box.isEnabled())
            box.setChecked(name in names and available)
            box.setEnabled(available)
        self.method_tree.blockSignals(prev_tree)
        self.update_method_count()
        if hasattr(self, "_session_timer"):
            self.remember_session()

    def edit_parameters(self, name):
        if self.configuration_locked():
            return
        try:
            common = {key: dimensions(self.fields[key]) if key == "hidden_dims" else value(self.fields[key])
                      for key in PARAMETERS.get(name, {}) if key in NEURAL}
        except ValueError as exc:
            self.show_error(str(exc))
            return
        from .presets import preset_parameters
        preset = preset_parameters(value(self.fields["jingwei_preset"]), name)
        dialog = ParameterDialog(name, {**common, **preset, **self.method_parameters.get(name, {})}, self)
        if dialog.exec() == QDialog.Accepted:
            self.method_parameters[name] = dialog.values()
            self.remember_session()

    def choose_features(self):
        if self.configuration_locked():
            return
        if not self.preview_data:
            self.show_error(self.t("请先检查数据"))
            return
        dialog = QDialog(self)
        dialog.setWindowTitle(self.t("W2 的共同参考蛋白"))
        dialog.resize(440, 560)
        layout = QVBoxLayout(dialog)
        layout.addWidget(text(self.t("选择共同蛋白子集；仅在该子集上原始完整的样本进入 W2。"), "muted"))
        top_bar = QHBoxLayout()
        search = QLineEdit()
        search.setPlaceholderText(self.t("搜索蛋白名称"))
        top_bar.addWidget(search, 1)
        listing = QListWidget()
        all_btn = QPushButton(self.t("全选"))
        none_btn = QPushButton(self.t("清空"))
        all_btn.clicked.connect(lambda: [listing.item(i).setCheckState(Qt.Checked) for i in range(listing.count()) if not listing.item(i).isHidden()])
        none_btn.clicked.connect(lambda: [listing.item(i).setCheckState(Qt.Unchecked) for i in range(listing.count()) if not listing.item(i).isHidden()])
        top_bar.addWidget(all_btn)
        top_bar.addWidget(none_btn)
        layout.addLayout(top_bar)
        feature_ids = self.preview_data["feature_ids"]
        chosen = set(self.feature_selection) if self.feature_selection else set(feature_ids)
        for name in feature_ids:
            item = QListWidgetItem(str(name))
            item.setData(Qt.UserRole, name)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked if name in chosen else Qt.Unchecked)
            listing.addItem(item)
        search.textChanged.connect(lambda query: [listing.item(i).setHidden(query.lower() not in listing.item(i).text().lower()) for i in range(listing.count())])
        layout.addWidget(listing, 1)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.button(QDialogButtonBox.Ok).setText(self.t("确定"))
        buttons.button(QDialogButtonBox.Cancel).setText(self.t("取消"))
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        if dialog.exec() == QDialog.Accepted:
            selection = [listing.item(i).data(Qt.UserRole) for i in range(listing.count()) if listing.item(i).checkState() == Qt.Checked]
            if not selection:
                self.show_error(self.t("至少选择一个蛋白"))
                return
            self.feature_selection = None if len(selection) == len(feature_ids) else selection
            self.update_data_labels()
            self.remember_session()

    def configuration(self):
        config = {key: value(widget) for key, widget in self.fields.items()}
        config["hidden_dims"] = dimensions(self.fields["hidden_dims"])
        config["methods"] = [name for name, box in self.method_checks.items() if box.isChecked()]
        for name in config["methods"]:
            if not self.backend_states.get(name, {}).get("available"):
                raise ValueError(self.t("方法 {name} 不可用，请检查运行环境", name=name))
        if "gpu_id" in config and (config["gpu_id"] == -1 or config["device"] == "cpu"):
            config["gpu_id"] = None
        for key in ("rscript", "r_library", "naguider_source", "dreamai_source"):
            config[key] = config[key] or None
        config["repeats"] = 1
        config["evaluation_protocol"] = "fair"
        config["seed"] = 0
        config["w2_features"] = self.feature_selection
        if not config["methods"]:
            raise ValueError(self.t("至少选择一种方法"))
        return config

    def start_job(self):
        if self.configuration_locked():
            return
        try:
            if not self.preview_data or self.signature() != self.preview_signature:
                raise ValueError(self.t("数据或读取设置已改变，请重新检查数据"))
            config = self.configuration()
            if not self.output_root.text().strip():
                raise ValueError(self.t("请选择输出目录"))
        except (ValueError, OSError) as exc:
            self.show_error(str(exc))
            return
        if self._submitting:
            return
        self._submitting = True
        self.update_run_state()
        output_root = self.output_root.text().strip()
        parameters = {} if config["evaluation_protocol"] == "original-notebook" else {
            name: settings for name, settings in self.method_parameters.items() if name in config["methods"]}
        def done(job_id):
            self._session_jobs.add(job_id)
            self.selected_job = job_id
            self._last_result_digest = None
            self._submitting = False
            self.result_tabs.setCurrentIndex(2)
            self.refresh_jobs()
            self.update_run_state()
        def failure(message):
            self._submitting = False
            self.update_run_state()
            self.show_error(message)
        self.background(lambda *a: self.store.submit(*a, output_root=output_root), done, self.input_path.text().strip(), config,
                        parameters, self.job_name.text(), error=failure)

    def _style_status_pill(self, state):
        palette = {
            "completed": ("#DCFCE7", "#15803D"),
            "running": ("#E8F1FB", "#1D5FA7"),
            "cancelling": ("#FEF3C7", "#B45309"),
        }
        bg, fg = palette.get(state, ("#FEE2E2", "#B91C1C"))
        self.progress_label.setStyleSheet(
            f"background:{bg};color:{fg};border-radius:4px;padding:2px 8px;font-weight:600;font-size:11px;"
        )

    def refresh_jobs(self):
        self.update_run_state()
        for button in self.job_actions:
            button.setEnabled(bool(self.selected_job) and (button is not self.clone_button or not self.configuration_locked()))
        if not self.selected_job:
            self.cancel_button.setEnabled(False)
            self.progress.hide()
            return
        try:
            info = self.store.info(self.selected_job)
        except (ValueError, OSError, KeyError, TypeError):
            self.selected_job = None
            self._last_result_digest = None
            self.current_task_label.setText(self.t("尚无任务"))
            self.current_task_label.setObjectName("muted")
            self.progress_label.setText(self.t("任务目录无法读取，请重新打开任务。"))
            self._style_status_pill("failed")
            self.progress_label.show()
            self.progress.hide()
            self.log_view.setPlainText(self.t("任务目录无法读取，请重新打开任务。"))
            self.run_model.replace([], self.run_model.columns)
            self.plot.empty(self.t("暂无可比较数据"))
            self.result_notice.clear()
            self.result_notice.hide()
            self.cancel_button.setEnabled(False)
            for button in self.job_actions:
                button.setEnabled(False)
            return
        self.current_task_label.setText(info["name"])
        self.current_task_label.setObjectName("taskTitle")
        self.current_task_label.style().unpolish(self.current_task_label)
        self.current_task_label.style().polish(self.current_task_label)
        self.current_task_label.setToolTip(info["folder"])
        if info["id"] in self._session_jobs and not self._environment_retry_used:
            environment_errors = ("cuda requested but unavailable", "rscript not found", "missing r packages", "r dependency check failed",
                                  "no module named", "dll load failed", "cannot load shared object")
            if any(any(message in (row.get("error") or "").lower() for message in environment_errors)
                   for row in info["runs"] if row.get("status") in ("failed", "skipped")):
                self._environment_retry_used = True
                self.refresh_environment(force=True)
        self.progress.setRange(0, max(1, info["expected"]))
        self.progress.setValue(info["done"])
        self.progress.setVisible(info["state"] in ("running", "cancelling"))
        self._style_status_pill(info["state"])
        self.progress_label.setText(self.t("{state} · {done}/{total}", state=self.t(STATE_LABELS[info["state"]]), done=info["done"], total=info["expected"]))
        self.progress_label.show()
        self.cancel_button.setEnabled(info["state"] == "running" and info["id"] in self.store._processes)
        digest = (info["id"], info["done"], info["state"])
        if digest != self._last_result_digest:
            self._last_result_digest = digest
            self.render_results()
        self.show_method_log()

    def render_results(self):
        if not self.selected_job:
            return
        info = self.store.info(self.selected_job)
        runs = pd.DataFrame(info["runs"])
        benchmark = info["config"].get("command") == "benchmark"
        if benchmark:
            show_best = (info["config"].get("training_policy") == "convergence"
                         or any(row.get("best_epoch") is not None for row in info["runs"]))
            columns = ["repeat", "method", "status", "MSE", "RMSE", "MAE", "NRMSE",
                       "Pearson", "Spearman", "W2", *(["best_epoch"] if show_best else []), "time_s"]
        else:
            columns = ["repeat", "method", "status", "epochs_trained", "time_s"]
        self.run_model.replace(info["runs"], columns)
        self.result_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.result_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        multiple = info["config"].get("repeats", 1) > 1
        self.paired.setVisible(benchmark and multiple)
        self.paired.setEnabled(benchmark and multiple)
        self.result_table.setColumnHidden(0, not multiple)
        paired = self.paired.isChecked() and benchmark and multiple
        if paired and not runs.empty:
            total = len(info["config"]["methods"])
            groups = runs.groupby("repeat")
            repeats = [repeat for repeat, group in groups if len(group) == total and (group.status == "completed").all()]
            scores = runs[runs.repeat.isin(repeats)]
        else:
            scores = runs
        metric = self.metric_choice.currentData()
        eval_scale = info["manifest"].get("evaluation_scale")
        if not eval_scale or eval_scale == "working":
            eval_scale = evaluation_scale_name(info["config"])
        count = self.plot.render(scores, metric, eval_scale, paired, info["id"])
        if count:
            self.plot.provenance.update(input_sha256=info["manifest"].get("input_sha256"),
                language=self.language,
                configuration=info["config"], dependencies=info["manifest"].get("versions"),
                candidate_rows=len(scores), included_rows=len(self.plot.source),
                W2_definition="multivariate empirical sample distribution on common complete reference features")
        failed = sum(row.get("status") != "completed" for row in info["runs"])
        if failed:
            message = self.t("{count} 种方法 · {failed} 条未完成/失败记录",
                             count=count if benchmark else len(info["runs"]), failed=failed)
        elif benchmark and not count:
            message = self.t("暂无可比较数据")
        else:
            message = ""
        self.result_notice.setText(message)
        self.result_notice.setVisible(bool(message))
        self.show_method_log()

    def show_method_log(self, *args):
        if not self.selected_job:
            return
        info = self.store.info(self.selected_job)
        folder = Path(info["folder"])
        results = Path(info.get("results_folder", folder / "results"))
        selected = self.result_table.currentRow()
        sections = [self.t("{state} · {done}/{total}", state=self.t(STATE_LABELS[info["state"]]), done=info["done"], total=info["expected"])]
        paths = [("任务", folder / "launch.log")]
        if 0 <= selected < len(info["runs"]):
            row = info["runs"][selected]
            path = results / f"repeat_{row.get('repeat', 0):03d}/{row['method']}/run.log"
            if row.get("error"):
                sections.append(str(row["error"]))
        else:
            logs = list(results.glob("repeat_*/*/run.log"))
            path = max(logs, key=lambda item: item.stat().st_mtime_ns) if logs else None
        if path:
            paths.append((path.parent.name, path))
        for label, path in paths:
            try:
                with path.open("rb") as log:
                    log.seek(max(0, path.stat().st_size - 32768))
                    content = log.read().decode("utf-8", errors="replace").strip()
                if content:
                    sections.append(f"[{self.t(label)}]\n{content}")
            except OSError:
                pass
        if len(sections) == 1 and info["state"] not in TERMINAL:
            sections.append(self.t("正在启动计算进程…"))
        content = "\n\n".join(sections)
        if self.log_view.toPlainText() != content:
            scroll = self.log_view.verticalScrollBar()
            follow = scroll.value() >= scroll.maximum() - 2
            position = scroll.value()
            self.log_view.setPlainText(content)
            scroll.setValue(scroll.maximum() if follow else position)

    def cancel_job(self):
        if self.selected_job:
            self.store.cancel(self.selected_job)
            self.refresh_jobs()

    def open_results(self):
        if self.selected_job:
            info = self.store.info(self.selected_job)
            folder = Path(info["folder"])
            if info["config"].get("command") == "impute":
                imputed = Path(info.get("results_folder", folder / "results")) / "imputed"
                if imputed.is_dir():
                    folder = imputed
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(folder)))

    def export_job(self):
        if not self.selected_job:
            return
        path = QFileDialog.getSaveFileName(self, self.t("导出完整任务"), "JINGWEI-results.zip", "ZIP (*.zip)")[0]
        if path:
            self.background(self.store.export, lambda result: self.flash_status(self.t("完整任务已导出")), self.selected_job, path)

    def export_plot(self):
        if self.plot.source.empty:
            self.show_error(self.t("暂无可导出的图表数据"))
            return
        path = QFileDialog.getSaveFileName(self, self.t("导出图表及源数据"), "JINGWEI-comparison.png", "PNG (*.png);;PDF (*.pdf);;SVG (*.svg)")[0]
        if not path:
            return
        try:
            self.plot.figure.savefig(path, dpi=300, facecolor="white")
            self.plot.source.to_csv(str(Path(path).with_suffix(".source.csv")), index=False)
            from .gui_jobs import write_json
            write_json(Path(path).with_suffix(".provenance.json"), self.plot.provenance)
            self.flash_status(self.t("图表、源数据和计算说明已导出"))
        except Exception as exc:
            self.show_error(str(exc))

    def clone_job(self):
        if not self.selected_job or self.configuration_locked():
            return
        from copy import deepcopy
        from .gui_jobs import read_json
        from .methods import canonical_name
        info = self.store.info(self.selected_job)
        parameters = deepcopy(info.get("parameters", {}))
        methods = [canonical_name(name) for name in info["config"]["methods"]]
        results = Path(info.get("results_folder", Path(info["folder"]) / "results"))
        # CLI manifests do not carry GUI overrides. The submitted worker jobs
        # record the effective common + preset + per-method parameter values.
        for name in methods:
            payload = read_json(results / f"repeat_000/{name}/job.json")
            if isinstance(payload, dict) and isinstance(payload.get("parameters"), dict):
                editable = {key: deepcopy(setting) for key, setting in payload["parameters"].items()
                            if key in PARAMETERS.get(name, {})}
                parameters[name] = {**parameters.get(name, {}), **editable}
        if "mice_backend" not in info["config"] and any(
                name in ("mice", "mice-pmm") for name in info["config"]["methods"]):
            set_value(self.fields["mice_backend"], "r")
        for key, data in info["config"].items():
            if key in self.fields:
                set_value(self.fields[key], -1 if key == "gpu_id" and data is None else data)
        self.method_parameters = parameters
        # Keep the recorded selection even if a newly selected runtime has not
        # finished probing availability yet.
        for name, box in self.method_checks.items():
            checked = name in methods
            box.setChecked(checked)
            if checked:
                box.setEnabled(True)
        self.mode_changed()
        self.device_changed()
        self.training_policy_changed()
        set_value(self.fields["output_scale"], info["config"].get("output_scale", "working"))
        self.output_root.setText(str(Path(info["folder"]).parent))
        self.job_name.setText(info["name"] + " · " + self.t("副本"))
        self.input_path.setFocus()
        source = next(Path(info["folder"]).glob("input.*"), None)
        if source is None and info["config"].get("input"):
            candidate = Path(info["config"]["input"])
            source = candidate if candidate.is_file() else None
        if source:
            self.set_input(source, info["config"].get("layer"))
            features = deepcopy(info["config"].get("w2_features"))
            if features:
                self._pending_features = self.signature(), features
        else:
            self.input_path.clear()
            self.invalidate_preview()
        self.refresh_environment()

    def closeEvent(self, event):
        if self._submitting:
            # Let submission publish its process identity before normal close
            # handling decides whether a running job must be stopped.
            if not getattr(self, "_close_after_submit", False):
                self._close_after_submit = True
                def retry():
                    self._close_after_submit = False
                    self.close()
                QTimer.singleShot(100, retry)
            event.ignore()
            return
        active = self.store.active()
        if active:
            question = QMessageBox(QMessageBox.Question, self.t("任务仍在运行"), self.t("停止运行中的任务并退出？"), QMessageBox.Yes | QMessageBox.No, self)
            question.button(QMessageBox.Yes).setText(self.t("是"))
            question.button(QMessageBox.No).setText(self.t("否"))
            question.setDefaultButton(QMessageBox.No)
            if question.exec() != QMessageBox.Yes:
                event.ignore()
                return
        for job in active:
            self.store.cancel(job["id"], force=True)
        self.save_session()
        self._closed = True
        self._environment_cancel.set()
        self._session_timer.stop()
        self.timer.stop()
        event.accept()


def launch(workspace=None):
    if sys.platform != "win32":
        raise ValueError("The benchmark GUI is Windows-only; use jingwei CLI")
    from .runtime import prepare_standard_streams
    prepare_standard_streams()
    from .windows_identity import set_process_app_id, apply_application_icon
    set_process_app_id()
    app = QApplication.instance() or QApplication(["jingwei"])
    if os.name == "nt":
        font_id = QFontDatabase.addApplicationFont("C:/Windows/Fonts/msyh.ttc")
        families = QFontDatabase.applicationFontFamilies(font_id)
        app.setFont(QFont(families[0] if families else "Microsoft YaHei UI", 10))
    apply_application_icon(app)
    store = JobStore(workspace)
    lock = QLockFile(str(store.root / "desktop.lock"))
    if not lock.tryLock(0):
        from .gui_language import initial_language
        language = initial_language(store.root)
        QMessageBox.information(None, "JINGWEI", translate(language, "这个工作目录已有 JINGWEI 窗口在运行。"))
        return 1
    from .gui_startup import StartupGate
    window = BenchmarkWindow(workspace, startup=True)
    gate = StartupGate(window)
    gate.start()
    code = app.exec()
    lock.unlock()
    return code
