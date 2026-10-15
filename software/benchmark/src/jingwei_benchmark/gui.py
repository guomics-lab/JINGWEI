"""Native Qt desktop interface for the shared proteomics benchmark backend."""

import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from PySide6.QtCore import Qt, QObject, Signal, QRunnable, QThreadPool, QTimer, QUrl, QLockFile, QAbstractTableModel
from PySide6.QtGui import QColor, QDesktopServices
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QFormLayout, QLabel,
    QComboBox, QLineEdit, QSpinBox, QDoubleSpinBox, QCheckBox, QListWidget,
    QListWidgetItem, QTableWidgetItem, QFileDialog, QMessageBox, QDialog,
    QTableView, QDialogButtonBox,
)

from .gui_jobs import JobStore, TERMINAL, inspect_input, read_json
from .gui_parameters import LABELS, PARAMETERS, NEURAL
from .gui_layout import WorkbenchLayout
from .gui_i18n import translate

STATE_LABELS = {"running": "运行中", "cancelling": "正在停止", "completed": "已完成",
                "incomplete": "部分完成", "failed": "失败", "cancelled": "已停止", "interrupted": "已中断",
                "skipped": "跳过", "timeout": "超时"}

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
        widget.setDecimals(8 if low is not None and 0 < low < 0.0001 else 4)
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
        self.resize(470, 340)
        layout = QVBoxLayout(self)
        self.fields = {}
        form = QFormLayout()
        for key, spec in PARAMETERS.get(name, {}).items():
            widget = control(spec)
            if key in overrides:
                set_value(widget, overrides[key])
            self.fields[key] = widget
            form.addRow(self.t(spec[0]), widget)
        if not self.fields:
            layout.addWidget(text(self.t("此方法没有可调整的参数。")))
        layout.addLayout(form)
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
                      "W2_n_samples": "W2 样本数", "W2_n_features": "W2 蛋白数", "W2_n_evaluated": "W2 评测数"}
            return self.t(labels.get(key, key))

    def data(self, index, role=Qt.DisplayRole):
        if not index.isValid():
            return None
        row, key = self.rows[index.row()], self.columns[index.column()]
        cell = row.get(key)
        if role == Qt.DisplayRole:
            if key == "status":
                return self.t(STATE_LABELS.get(cell, cell or "—"))
            return "—" if cell is None else f"{cell:.5g}" if isinstance(cell, float) else str(cell)
        if role == Qt.ToolTipRole:
            return row.get("error") or row.get("W2_note") or ""
        if role == Qt.ForegroundRole and key == "status" and cell != "completed":
            return QColor("#a44128")


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
        rcParams["font.sans-serif"] = ["Microsoft YaHei", "DejaVu Sans"]
        rcParams["axes.unicode_minus"] = False
        # Select the native Qt binding, independent of PIMMS worker plotting.
        os.environ["QT_API"] = "pyside6"
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
        self.figure = Figure(figsize=(8, 4), dpi=100, layout="constrained", facecolor="white")
        self.canvas = FigureCanvasQTAgg(self.figure)
        QVBoxLayout(self).addWidget(self.canvas)
        self.source = pd.DataFrame()
        self.provenance = {}
        self.empty(self.t("暂无可比较数据"))

    def empty(self, message):
        self.setFixedHeight(300)
        self.figure.clear()
        axes = self.figure.add_subplot()
        axes.axis("off")
        # Qt label carries localized explanatory text; plot keeps portable fonts.
        axes.text(0.5, 0.5, self.t("暂无可比较数据"), ha="center", va="center", color="#68768c")
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
        self.setFixedHeight(max(300, len(stats) * 28 + 140))
        self.figure.clear()
        axes = self.figure.add_subplot()
        positions = np.arange(len(stats))
        colors = ["#375fc7" if name.startswith("jingwei") else "#557e9b" for name in stats.index]
        axes.barh(positions, stats["mean"], color=colors, height=0.64)
        for i, row in enumerate(stats.itertuples()):
            if np.isfinite(row.std):
                axes.errorbar(row.mean, i, xerr=row.std, fmt="none", color="#17253c", capsize=3, lw=1)
        axes.set_yticks(positions, [f"{name}  (n={int(count)})" for name, count in zip(stats.index, stats["count"])], fontsize=9)
        axes.invert_yaxis()
        axes.axvline(0, color="#8693a8", linewidth=0.6)
        axes.set_axisbelow(True)
        axes.grid(axis="x", color="#e9edf4", linewidth=0.7)
        for side in ("top", "right", "left"):
            axes.spines[side].set_visible(False)
        if metric not in ("Pearson", "Spearman"):
            axes.set_xlim(left=0)
        units = self.t("秒") if metric == "time_s" else scale if metric in ("MAE", "RMSE", "W2") else self.t("无量纲")
        axes.set_xlabel(f"{metric} ({units})")
        axes.set_title(self.t("共同完整重复" if paired else "各方法成功运行"), loc="left", fontsize=11)
        self.figure.supxlabel(self.t("均值 ± 标准差（n ≥ 2）"), fontsize=8, color="#59677e")
        self.canvas.draw_idle()
        self.source = frame
        self.provenance = {"job_id": job_id, "metric": metric, "scale": scale,
                           "aggregation": "mean across repeat runs", "uncertainty": "sample SD; absent when n=1",
                           "paired_complete_repeats_only": paired, "nonfinite_metric_rows_excluded": True}
        return len(stats)


class BenchmarkWindow(WorkbenchLayout, QMainWindow):
    def __init__(self, workspace=None, *, load_default=True):
        super().__init__()
        self.store = JobStore(workspace)
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
        self.setup_ui()
        self.timer = QTimer(self)
        self.timer.setInterval(900)
        self.timer.timeout.connect(self.refresh_jobs)
        self.timer.start()
        self.refresh_jobs()
        self.refresh_environment()
        if load_default:
            from .runtime import default_input_path
            if path := default_input_path():
                self.set_input(path)

    def background(self, function, done, *args, error=None):
        task = BackgroundTask(function, *args)
        self._tasks.add(task)
        def success(result):
            self._tasks.discard(task)
            done(result)
        def failure(message):
            self._tasks.discard(task)
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

    def refresh_environment(self):
        self._environment_checking = True
        self.update_environment_labels()
        options = {key: value(self.fields[key]) or None for key in ("rscript", "r_library", "naguider_source", "dreamai_source", "mice_backend")}
        def inspect():
            # Probe torch and R in an isolated process, never in a Qt thread.
            from .gui_environment import inspect_environment
            return inspect_environment(options)
        def done(result):
            self.backend_states, gpu, self._gpu_name = result
            self._environment_checking = False
            for name, box in self.method_checks.items():
                box.setEnabled(self.backend_states[name]["available"])
            self.update_environment_labels()
            if "gpu_id" in self.fields:
                index = self.fields["device"].findData("cuda")
                self.fields["device"].model().item(index).setEnabled(gpu)
                self.device_changed()
            self.update_run_state()
        def failed(message):
            self._environment_checking = False
            self.update_environment_labels()
            self.show_error(message)
        self.background(inspect, done, error=failed)

    def update_run_state(self):
        active = any(row["state"] not in TERMINAL for row in self.store.history())
        self.start_button.setEnabled(self.preview_data is not None and bool(self.backend_states) and not active and not self._submitting)

    def device_changed(self):
        if "gpu_id" in self.fields:
            self.fields["gpu_id"].setEnabled(value(self.fields["device"]) != "cpu")

    def mode_changed(self):
        benchmark = value(self.fields["command"]) == "benchmark"
        for key in ("mask_rate", "mechanism", "repeats", "training_mechanism", "training_mask_rate", "mask_level",
                    "logistic_slope", "low_quantile", "high_quantile", "low_probability", "high_probability", "threshold_probability",
                    "paper_probabilities", "allow_empty_marginals"):
            self.fields[key].setEnabled(benchmark)
        self.fields["repeats"].setEnabled(benchmark)
        self.protocol_summary.setVisible(benchmark)
        set_value(self.fields["output_scale"], "working" if benchmark else "input")

    def invalidate_preview(self, *args):
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
        path = Path(self.input_path.text()).resolve()
        stat = path.stat()
        return str(path), stat.st_size, stat.st_mtime_ns, self.input_options()

    def inspect_data(self):
        try:
            signature = self.signature()
        except OSError as exc:
            self.show_error(str(exc))
            return
        generation = self._inspect_generation
        self.inspect_button.setEnabled(False)
        self._inspecting = True
        self.update_data_labels()
        def done(data):
            self.inspect_button.setEnabled(True)
            if generation != self._inspect_generation:
                return
            self.preview_data = data
            self.preview_signature = signature
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
                        item.setBackground(QColor("#fff0e6"))
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
            self.inspect_button.setEnabled(True)
            self._inspecting = False
            self.data_summary.setText(self.t("数据检查失败：{error}", error=message))
            self.show_error(message)
        self.background(inspect_input, done, signature[0], signature[3], error=failure)

    def select_methods(self, names):
        names = set(names if names is not None else [name for name, state in self.backend_states.items() if state["available"]])
        for name, box in self.method_checks.items():
            box.setChecked(name in names and box.isEnabled())

    def edit_parameters(self, name):
        try:
            common = {key: dimensions(self.fields[key]) if key == "hidden_dims" else value(self.fields[key])
                      for key in PARAMETERS.get(name, {}) if key in NEURAL}
        except ValueError as exc:
            self.show_error(str(exc))
            return
        dialog = ParameterDialog(name, {**common, **self.method_parameters.get(name, {})}, self)
        if dialog.exec() == QDialog.Accepted:
            self.method_parameters[name] = dialog.values()

    def choose_features(self):
        if not self.preview_data:
            self.show_error(self.t("请先检查数据"))
            return
        dialog = QDialog(self)
        dialog.setWindowTitle(self.t("W2 的共同参考蛋白"))
        dialog.resize(430, 560)
        layout = QVBoxLayout(dialog)
        layout.addWidget(text(self.t("选择共同蛋白子集；仅在该子集上原始完整的样本进入 W2。"), "muted"))
        search = QLineEdit()
        search.setPlaceholderText(self.t("搜索蛋白名称"))
        layout.addWidget(search)
        listing = QListWidget()
        chosen = set(self.feature_selection or self.preview_data["feature_ids"])
        for name in self.preview_data["feature_ids"]:
            item = QListWidgetItem(str(name))
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked if name in chosen else Qt.Unchecked)
            listing.addItem(item)
        search.textChanged.connect(lambda query: [listing.item(i).setHidden(query.lower() not in listing.item(i).text().lower()) for i in range(listing.count())])
        layout.addWidget(listing)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.button(QDialogButtonBox.Ok).setText(self.t("确定"))
        buttons.button(QDialogButtonBox.Cancel).setText(self.t("取消"))
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        if dialog.exec() == QDialog.Accepted:
            selection = [listing.item(i).text() for i in range(listing.count()) if listing.item(i).checkState() == Qt.Checked]
            if not selection:
                self.show_error(self.t("至少选择一个蛋白"))
                return
            self.feature_selection = selection
            self.update_data_labels()

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
        if config["command"] == "impute":
            config["repeats"] = 1
        config["w2_features"] = self.feature_selection
        if not config["methods"]:
            raise ValueError(self.t("至少选择一种方法"))
        return config

    def start_job(self):
        try:
            if not self.preview_data or self.signature() != self.preview_signature:
                raise ValueError(self.t("数据或读取设置已改变，请重新检查数据"))
            config = self.configuration()
        except (ValueError, OSError) as exc:
            self.show_error(str(exc))
            return
        if self._submitting:
            return
        self._submitting = True
        self.start_button.setEnabled(False)
        def done(job_id):
            self.selected_job = job_id
            self._last_result_digest = None
            self._submitting = False
            self.result_tabs.setCurrentIndex(0)
            self.refresh_jobs()
            self.update_run_state()
        def failure(message):
            self._submitting = False
            self.update_run_state()
            self.show_error(message)
        self.background(self.store.submit, done, self.input_path.text(), config,
                        {name: settings for name, settings in self.method_parameters.items() if name in config["methods"]}, self.job_name.text(), error=failure)

    def refresh_jobs(self):
        rows = self.store.history()
        selected = self.selected_job or (rows[0]["id"] if rows else None)
        existing = [self.job_combo.itemData(i) for i in range(self.job_combo.count())]
        ids = [row["id"] for row in rows]
        self.job_combo.blockSignals(True)
        if existing != ids:
            self.job_combo.clear()
            for row in rows:
                self.job_combo.addItem(row["name"], row["id"])
        for i, row in enumerate(rows):
            self.job_combo.setItemText(i, f"{row['name']} · {self.t(STATE_LABELS.get(row['state'],row['state']))} · {row['created_at'][:16].replace('T',' ')} UTC")
        if selected in ids:
            self.job_combo.setCurrentIndex(ids.index(selected))
            self.selected_job = selected
        self.job_combo.blockSignals(False)
        self.update_run_state()
        for button in self.job_actions:
            button.setEnabled(bool(self.selected_job))
        if not self.selected_job:
            self.cancel_button.setEnabled(False)
            return
        info = self.store.info(self.selected_job)
        self.progress.setRange(0, max(1, info["expected"]))
        self.progress.setValue(info["done"])
        self.progress_label.setText(self.t("{state} · {done}/{total}", state=self.t(STATE_LABELS[info["state"]]), done=info["done"], total=info["expected"]))
        self.cancel_button.setEnabled(info["state"] == "running")
        digest = (info["id"], info["done"], info["state"])
        if digest != self._last_result_digest:
            self._last_result_digest = digest
            self.render_results()
        if not self.result_table.selectedItems():
            self.show_method_log()

    def select_job(self):
        self.selected_job = self.job_combo.currentData()
        self._last_result_digest = None
        if self.selected_job:
            self.refresh_jobs()

    def render_results(self):
        if not self.selected_job:
            return
        info = self.store.info(self.selected_job)
        runs = pd.DataFrame(info["runs"])
        columns = ["repeat", "method", "status", "RMSE", "MAE", "NRMSE", "Pearson", "Spearman", "W2", "W2_n_samples", "W2_n_features", "W2_n_evaluated", "time_s"]
        self.run_model.replace(info["runs"], columns)
        benchmark = info["config"].get("command") == "benchmark"
        self.paired.setEnabled(benchmark)
        paired = self.paired.isChecked() and benchmark
        if paired and not runs.empty:
            total = len(info["config"]["methods"])
            groups = runs.groupby("repeat")
            repeats = [repeat for repeat, group in groups if len(group) == total and (group.status == "completed").all()]
            scores = runs[runs.repeat.isin(repeats)]
        else:
            scores = runs
        metric = self.metric_choice.currentData()
        count = self.plot.render(scores, metric, info["manifest"].get("evaluation_scale", "working"), paired, info["id"])
        if count:
            self.plot.provenance.update(input_sha256=info["manifest"].get("input_sha256"),
                language=self.language,
                configuration=info["config"], dependencies=info["manifest"].get("versions"),
                candidate_rows=len(scores), included_rows=len(self.plot.source),
                W2_definition="multivariate empirical sample distribution on common complete reference features")
        failed = sum(row.get("status") != "completed" for row in info["runs"])
        message = self.t("{count} 种方法 · {failed} 条未完成/失败记录", count=count, failed=failed)
        if not benchmark:
            message = self.t("插补任务仅提供运行时间；准确性指标不适用。")
        elif not count:
            message = self.t("暂无可比较数据") + " · " + message
        self.result_notice.setText(message)
        self.show_method_log()

    def show_method_log(self, *args):
        if not self.selected_job:
            return
        info = self.store.info(self.selected_job)
        folder = Path(info["folder"])
        selected = self.result_table.currentRow()
        if 0 <= selected < len(info["runs"]):
            row = info["runs"][selected]
            path = folder / f"results/repeat_{row['repeat']:03d}/{row['method']}/run.log"
        else:
            logs = list((folder / "results").glob("repeat_*/*/run.log"))
            path = max(logs, key=lambda item: item.stat().st_mtime_ns) if logs and info["state"] not in TERMINAL else folder / "launch.log"
        try:
            with path.open("rb") as log:
                log.seek(max(0, path.stat().st_size - 65536))
                content = log.read().decode("utf-8", errors="replace")
        except OSError:
            content = self.t("日志尚未生成。")
        if self.log_view.toPlainText() != content:
            self.log_view.setPlainText(content)
            self.log_view.verticalScrollBar().setValue(self.log_view.verticalScrollBar().maximum())

    def cancel_job(self):
        if self.selected_job:
            self.store.cancel(self.selected_job)
            self.refresh_jobs()

    def open_results(self):
        if self.selected_job:
            QDesktopServices.openUrl(QUrl.fromLocalFile(self.store.info(self.selected_job)["folder"]))

    def export_job(self):
        if not self.selected_job:
            return
        path = QFileDialog.getSaveFileName(self, self.t("导出完整任务"), "JINGWEI-results.zip", "ZIP (*.zip)")[0]
        if path:
            self.background(self.store.export, lambda result: self.statusBar().showMessage(self.t("完整任务已导出"), 5000), self.selected_job, path)

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
            self.statusBar().showMessage(self.t("图表、源数据和计算说明已导出"), 5000)
        except Exception as exc:
            self.show_error(str(exc))

    def clone_job(self):
        if not self.selected_job:
            return
        info = self.store.info(self.selected_job)
        if "mice_backend" not in info["config"] and any(
                name in ("mice", "mice-pmm") for name in info["config"]["methods"]):
            set_value(self.fields["mice_backend"], "r")
        for key, data in info["config"].items():
            if key in self.fields:
                set_value(self.fields[key], -1 if key == "gpu_id" and data is None else data)
        self.method_parameters = dict(info.get("parameters", {}))
        self.select_methods(info["config"]["methods"])
        self.mode_changed()
        set_value(self.fields["output_scale"], info["config"].get("output_scale", "working"))
        self.job_name.setText(info["name"] + " · " + self.t("副本"))
        self.input_path.setFocus()
        source = next(Path(info["folder"]).glob("input.*"), None)
        if source:
            self.set_input(source, info["config"].get("layer"))
            self.feature_selection = info["config"].get("w2_features")
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
        active = [job for job in self.store.history() if job["state"] not in TERMINAL]
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
        event.accept()


def launch(workspace=None):
    if sys.platform != "win32":
        raise ValueError("The benchmark GUI is Windows-only; use jingwei-benchmark CLI")
    from .runtime import prepare_standard_streams
    prepare_standard_streams()
    from .windows_identity import set_process_app_id, apply_application_icon
    set_process_app_id()
    app = QApplication.instance() or QApplication(["JINGWEI Benchmark"])
    apply_application_icon(app)
    store = JobStore(workspace)
    lock = QLockFile(str(store.root / "desktop.lock"))
    if not lock.tryLock(0):
        language = read_json(store.root / "ui.json", {}).get("language", "zh")
        QMessageBox.information(None, "JINGWEI", translate(language, "这个工作目录已有 JINGWEI 窗口在运行。"))
        return 1
    window = BenchmarkWindow(store.root)
    window.show()
    code = app.exec()
    lock.unlock()
    return code
