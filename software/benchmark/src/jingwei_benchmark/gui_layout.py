"""Compact native workbench layout and live language bindings."""

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QFormLayout, QLabel, QPushButton, QComboBox,
    QLineEdit, QSpinBox, QCheckBox, QGroupBox, QDialog, QDialogButtonBox, QTabWidget,
    QTreeWidget, QTreeWidgetItem, QTableWidget, QHeaderView, QAbstractItemView,
    QProgressBar, QPlainTextEdit, QSplitter, QScrollArea,
)

from .gui_i18n import translate
from .cli import DEFAULT_TIMEOUT_SECONDS
from .gui_jobs import read_json, write_json
from .gui_parameters import NEURAL, PARAMETERS, LABELS
from .methods import METHODS
from .platforms import cuda_supported


STYLE = """
QWidget {font-size:12px;color:#263442;}
QMainWindow {background:#f2f4f6;}
QGroupBox {background:white;border:1px solid #d3dbe3;border-radius:4px;margin-top:8px;padding:12px 9px 8px;font-weight:600;}
QGroupBox::title {subcontrol-origin:margin;left:10px;padding:0 4px;}
QPushButton {background:#fff;border:1px solid #c5cfd9;border-radius:3px;padding:5px 10px;min-height:20px;}
QPushButton:hover {background:#edf3f8;border-color:#5d8aaf;}
QPushButton:disabled {color:#9ba5af;background:#f1f3f5;border-color:#dce1e6;}
QPushButton#primary {background:#24699b;color:white;border-color:#24699b;font-weight:600;}
QPushButton#primary:hover {background:#1b567f;}
QPushButton#primary:disabled {background:#a6bccd;border-color:#a6bccd;}
QLineEdit,QComboBox,QSpinBox,QDoubleSpinBox {background:white;border:1px solid #c5cfd9;border-radius:3px;padding:3px 5px;min-height:20px;}
QLineEdit:focus,QComboBox:focus,QSpinBox:focus,QDoubleSpinBox:focus {border-color:#24699b;}
QTreeWidget,QTableView,QPlainTextEdit {background:white;border:1px solid #d3dbe3;selection-background-color:#dcebf6;selection-color:#153d5a;}
QTreeWidget::item {height:23px;}
QHeaderView::section {background:#f0f3f6;border:0;border-bottom:1px solid #d3dbe3;padding:6px;}
QTabWidget::pane {border:1px solid #d3dbe3;background:white;}
QTabBar::tab {background:#e9edf1;padding:8px 15px;border:1px solid #d3dbe3;}
QTabBar::tab:selected {background:white;border-bottom-color:white;}
QProgressBar {border:1px solid #d3dbe3;border-radius:2px;background:white;text-align:center;max-height:14px;}
QProgressBar::chunk {background:#4587b5;}
QLabel#muted {color:#647584;}
QLabel#brand {font-size:17px;font-weight:600;}
QPlainTextEdit {font-family:Consolas,monospace;font-size:12px;}
QSplitter::handle {background:#e1e6eb;}
"""


class MethodCheck:
    """Stable checked-method API backed by a native tree checkbox."""
    def __init__(self, item):
        self.item = item

    def isChecked(self):
        return self.item.checkState(0) == Qt.Checked

    def setChecked(self, checked):
        self.item.setCheckState(0, Qt.Checked if checked else Qt.Unchecked)

    def isEnabled(self):
        return not self.item.isDisabled()

    def setEnabled(self, enabled):
        self.item.setDisabled(not enabled)

    def setToolTip(self, label):
        self.item.setToolTip(0, label)


class WorkbenchLayout:
    def t(self, source, **values):
        return translate(self.language, source, **values)

    def bind(self, widget, source, setter="setText"):
        self._translations.append((widget, setter, source))
        getattr(widget, setter)(self.t(source))
        return widget

    def label(self, source, muted=False):
        label = self.bind(QLabel(), source)
        label.setTextFormat(Qt.PlainText)
        if muted:
            label.setObjectName("muted")
        return label

    def button(self, source, action):
        button = self.bind(QPushButton(), source)
        button.clicked.connect(action)
        return button

    def options(self, items):
        widget = QComboBox()
        for source, data in items:
            widget.addItem(self.t(source), data)
        self._combo_translations.append((widget, list(items)))
        return widget

    def field(self, form, key, source, widget):
        self.fields[key] = widget
        form.addRow(self.label(source), widget)
        return widget

    def group(self, source, parent_layout):
        group = self.bind(QGroupBox(), source, "setTitle")
        parent_layout.addWidget(group)
        layout = QVBoxLayout(group)
        layout.setSpacing(6)
        return layout

    def setup_ui(self):
        from .gui import RunTable, RunTableModel, MetricPlot, FractionControl, control
        from .windows_identity import apply_application_icon
        self.language = read_json(self.store.root / "ui.json", {}).get("language", "zh")
        if self.language not in ("zh", "en"):
            self.language = "zh"
        self._translations, self._combo_translations = [], []
        self.fields = {}
        self._gpu_name = None
        self._environment_checking = True
        self._inspecting = False
        self.setWindowTitle("JINGWEI Benchmark 1.0")
        app = QApplication.instance()
        # A light native widget palette keeps unchecked boxes legible even when
        # Windows uses dark mode. Fusion gives stable metrics across DPI scales.
        app.setStyle("Fusion")
        app.styleHints().setColorScheme(Qt.ColorScheme.Light)
        apply_application_icon(app, self)
        self.logo = self.windowIcon().pixmap(64, 64)
        self.resize(1260, 800)
        self.setMinimumSize(1080, 700)
        self.setStyleSheet(STYLE)
        root = QWidget()
        layout = QVBoxLayout(root)
        layout.setContentsMargins(12, 9, 12, 9)
        layout.setSpacing(9)
        toolbar = QHBoxLayout()
        badge = QLabel()
        badge.setPixmap(self.logo.scaled(30, 30, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        toolbar.addWidget(badge)
        brand = QLabel("JINGWEI Benchmark")
        brand.setObjectName("brand")
        toolbar.addWidget(brand)
        toolbar.addStretch()
        self.advanced_button = self.button("高级设置", self.show_advanced)
        toolbar.addWidget(self.advanced_button)
        toolbar.addWidget(self.button("计算环境", lambda: self.show_advanced(3)))
        self.language_choice = QComboBox()
        self.language_choice.setObjectName("languageChoice")
        self.language_choice.setAccessibleName("Language / 语言")
        self.language_choice.addItem("中文", "zh")
        self.language_choice.addItem("English", "en")
        self.language_choice.setCurrentIndex(self.language_choice.findData(self.language))
        self.language_choice.currentIndexChanged.connect(lambda: self.set_language(self.language_choice.currentData()))
        toolbar.addWidget(self.language_choice)
        layout.addLayout(toolbar)
        splitter = QSplitter(Qt.Horizontal)
        left = QWidget()
        config = QVBoxLayout(left)
        config.setContentsMargins(0, 0, 0, 0)
        config.setSpacing(8)
        data = self.group("数据", config)
        row = QHBoxLayout()
        self.input_path = self.bind(QLineEdit(), "选择 CSV 或 AnnData (.h5ad)", "setPlaceholderText")
        row.addWidget(self.input_path, 1)
        row.addWidget(self.button("选择文件", self.choose_input))
        row.addWidget(self.button("示例", self.load_example))
        data.addLayout(row)
        form = QFormLayout()
        self.field(form, "scale", "输入尺度", self.options([("原始正丰度", "raw"), ("已经 log2 转换", "log2")]))
        self.field(form, "transform", "预处理", self.options([("自动（原始丰度转为 log2）", "auto"), ("不转换", "none"), ("log2 转换", "log2")]))
        data.addLayout(form)
        row = QHBoxLayout()
        self.data_summary = self.label("尚未载入数据", True)
        self.data_summary.setWordWrap(True)
        row.addWidget(self.data_summary, 1)
        self.inspect_button = self.button("检查", self.inspect_data)
        row.addWidget(self.inspect_button)
        self.preview_button = self.button("预览", self.show_preview)
        self.preview_button.setEnabled(False)
        row.addWidget(self.preview_button)
        row.addWidget(self.button("导入设置", lambda: self.show_advanced(0)))
        data.addLayout(row)
        methods = self.group("方法", config)
        row = QHBoxLayout()
        self.method_count = QLabel()
        self.method_count.setObjectName("muted")
        row.addWidget(self.method_count, 1)
        row.addWidget(self.button("全选", lambda: self.select_methods(None)))
        row.addWidget(self.button("清空", lambda: self.select_methods([])))
        self.parameter_button = self.button("参数…", self.edit_current_method)
        self.parameter_button.setEnabled(False)
        row.addWidget(self.parameter_button)
        methods.addLayout(row)
        self.method_tree = QTreeWidget()
        self.method_tree.setHeaderHidden(True)
        self.method_tree.setMinimumHeight(145)
        self.method_tree.setRootIsDecorated(True)
        self.method_tree.setUniformRowHeights(True)
        self.method_checks = {}
        self.method_groups = {}
        for backend, source in (("jingwei", "JINGWEI"), ("pimms", "PIMMS"), ("python", "传统方法"), ("r", "NAguideR / R")):
            group = QTreeWidgetItem(self.method_tree, [self.t(source)])
            self.method_groups[backend] = group
            for name, spec in METHODS.items():
                if spec.backend != backend:
                    continue
                item = QTreeWidgetItem(group, [self.t(LABELS.get(name, name))])
                item.setData(0, Qt.UserRole, name)
                item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
                item.setCheckState(0, Qt.Checked if backend == "jingwei" else Qt.Unchecked)
                item.setDisabled(True)
                self.method_checks[name] = MethodCheck(item)
            group.setExpanded(backend == "jingwei")
        self.method_tree.itemChanged.connect(lambda *_: self.update_method_count())
        self.method_tree.currentItemChanged.connect(self.current_method_changed)
        self.method_tree.itemDoubleClicked.connect(lambda item, _: self.edit_current_method() if item.data(0, Qt.UserRole) else None)
        methods.addWidget(self.method_tree, 1)
        config.setStretch(1, 1)
        settings = self.group("运行设置", config)
        form = QFormLayout()
        self.field(form, "command", "任务", self.options([("评测", "benchmark"), ("插补", "impute")]))
        mechanisms = [("MCAR", "mcar"), ("MNAR · Logistic", "mnar-logistic"), ("MNAR · Threshold", "mnar-threshold"), ("MNAR · JINGWEI", "mnar-jingwei")]
        self.field(form, "mechanism", "缺失机制", self.options(mechanisms))
        self.field(form, "mask_rate", "评测掩蔽比例", FractionControl(0.2, 0.0001, 0.9999))
        self.bind(self.fields["mask_rate"], "训练掩蔽只从评测掩蔽后剩余的观测值中抽取。评测真值不传入方法。", "setToolTip")
        self.field(form, "repeats", "重复次数", control(("", 5, 1, 10000)))
        devices = [("自动", "auto"), ("CPU", "cpu"), ("NVIDIA CUDA", "cuda")] if cuda_supported() else [("CPU", "cpu")]
        self.field(form, "device", "计算设备", self.options(devices))
        settings.addLayout(form)
        self.protocol_summary = QLabel()
        self.protocol_summary.setObjectName("muted")
        self.bind(self.protocol_summary, "训练掩蔽只从评测掩蔽后剩余的观测值中抽取。评测真值不传入方法。", "setToolTip")
        settings.addWidget(self.protocol_summary)
        self.job_name = self.bind(QLineEdit(), "任务名称（可选）", "setPlaceholderText")
        config.addWidget(self.job_name)
        row = QHBoxLayout()
        self.start_button = self.button("开始", self.start_job)
        self.start_button.setObjectName("primary")
        self.start_button.setMinimumHeight(30)
        self.start_button.setEnabled(False)
        self.cancel_button = self.button("停止", self.cancel_job)
        self.cancel_button.setEnabled(False)
        row.addWidget(self.start_button, 2)
        row.addWidget(self.cancel_button, 1)
        config.addLayout(row)
        splitter.addWidget(left)
        right = QWidget()
        results = QVBoxLayout(right)
        results.setContentsMargins(8, 4, 0, 0)
        results.setSpacing(8)
        top = QHBoxLayout()
        top.addWidget(self.label("任务记录"))
        self.job_combo = QComboBox()
        self.job_combo.setMinimumContentsLength(8)
        self.job_combo.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.job_combo.currentIndexChanged.connect(self.select_job)
        top.addWidget(self.job_combo, 1)
        results.addLayout(top)
        actions = QHBoxLayout()
        self.progress_label = self.label("尚无任务", True)
        actions.addWidget(self.progress_label, 1)
        self.job_actions = []
        for label, action in (("打开目录", self.open_results), ("导出任务", self.export_job), ("复制配置", self.clone_job)):
            button = self.button(label, action)
            button.setEnabled(False)
            actions.addWidget(button)
            self.job_actions.append(button)
        results.addLayout(actions)
        self.progress = QProgressBar()
        self.progress.setValue(0)
        results.addWidget(self.progress)
        self.result_tabs = QTabWidget()
        self.result_table = RunTable()
        self.run_model = RunTableModel(self.t)
        self.result_table.setModel(self.run_model)
        self.run_model.replace([], ["repeat", "method", "status", "RMSE", "MAE", "NRMSE", "Pearson", "Spearman", "W2", "time_s"])
        self.result_table.setAlternatingRowColors(True)
        self.result_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.result_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.result_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.result_table.selectionModel().selectionChanged.connect(self.show_method_log)
        self.result_tabs.addTab(self.result_table, self.t("结果"))
        chart = QWidget()
        chart_layout = QVBoxLayout(chart)
        controls = QHBoxLayout()
        self.paired = self.bind(QCheckBox(), "共同重复")
        self.paired.setChecked(True)
        self.bind(self.paired, "仅比较所有所选方法均成功完成的重复；取消勾选后，各方法使用各自的成功重复。", "setToolTip")
        self.paired.stateChanged.connect(self.render_results)
        controls.addWidget(self.paired, 1)
        self.metric_choice = self.options([(name, name) for name in ("RMSE", "MAE", "NRMSE", "Pearson", "Spearman", "W2")] + [("耗时（秒）", "time_s")])
        self.metric_choice.currentIndexChanged.connect(self.render_results)
        controls.addWidget(self.metric_choice)
        controls.addWidget(self.button("导出图表", self.export_plot))
        chart_layout.addLayout(controls)
        self.plot = MetricPlot(self.t)
        chart_scroll = QScrollArea()
        chart_scroll.setWidgetResizable(True)
        chart_scroll.setFrameShape(QScrollArea.NoFrame)
        chart_scroll.setAlignment(Qt.AlignTop)
        chart_scroll.setWidget(self.plot)
        chart_layout.addWidget(chart_scroll, 1)
        self.result_tabs.addTab(chart, self.t("图表"))
        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumBlockCount(2000)
        self.result_tabs.addTab(self.log_view, self.t("日志"))
        results.addWidget(self.result_tabs, 1)
        self.result_notice = self.label("暂无结果", True)
        self.bind(self.result_notice, "图表显示均值与样本标准差；n=1 时不显示误差线。失败与未定义指标保留在结果表中。", "setToolTip")
        results.addWidget(self.result_notice)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([440, 790])
        layout.addWidget(splitter, 1)
        self.env_badge = self.label("正在检查计算环境…", True)
        layout.addWidget(self.env_badge)
        self.setCentralWidget(root)
        self.build_advanced(mechanisms)
        self.preview_dialog = QDialog(self)
        self.preview_dialog.resize(870, 480)
        self.bind(self.preview_dialog, "预览", "setWindowTitle")
        preview_layout = QVBoxLayout(self.preview_dialog)
        self.preview_details = QLabel()
        self.preview_details.setTextFormat(Qt.PlainText)
        self.preview_details.setWordWrap(True)
        preview_layout.addWidget(self.preview_details)
        self.preview_table = QTableWidget()
        self.preview_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.preview_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        preview_layout.addWidget(self.preview_table, 1)
        preview_layout.addWidget(self.label("仅预览前 12 个样本、10 个蛋白；橙色单元格表示缺失。", True))
        self.dialog_buttons(preview_layout, self.preview_dialog, QDialogButtonBox.Close)
        self.input_path.textChanged.connect(self.invalidate_preview)
        for key in ("orientation", "header", "index_column", "layer", "missing_values", "scale", "transform"):
            self.fields[key].currentIndexChanged.connect(self.invalidate_preview)
        self.fields["max_dense_mb"].valueChanged.connect(self.invalidate_preview)
        self.fields["command"].currentIndexChanged.connect(self.mode_changed)
        self.fields["device"].currentIndexChanged.connect(self.device_changed)
        for key in ("training_mask_rate", "seed"):
            self.fields[key].valueChanged.connect(self.update_protocol_summary)
        self.fields["training_mechanism"].currentIndexChanged.connect(self.update_protocol_summary)
        self.mode_changed()
        self.device_changed()
        self.update_method_count()
        self.update_protocol_summary()

    def build_advanced(self, mechanisms):
        from .gui import control, FractionControl
        self.advanced_dialog = QDialog(self)
        self.bind(self.advanced_dialog, "高级设置", "setWindowTitle")
        self.advanced_dialog.resize(700, 650)
        layout = QVBoxLayout(self.advanced_dialog)
        self.advanced_tabs = QTabWidget()
        layout.addWidget(self.advanced_tabs, 1)
        forms = []
        for source in ("导入", "训练", "掩蔽与 W2", "运行环境"):
            page = QWidget()
            content = QVBoxLayout(page)
            form = QFormLayout()
            content.addLayout(form)
            forms.append((form, content))
            self.advanced_tabs.addTab(page, self.t(source))
        form, content = forms[0]
        for key, label, items in [
            ("orientation", "矩阵方向", [("样本 × 蛋白", "samples-by-features"), ("蛋白 × 样本（CSV）", "features-by-samples")]),
            ("header", "CSV 第一行", [("自动识别", "auto"), ("是表头", "yes"), ("是数值数据", "no")]),
            ("index_column", "CSV 第一列", [("自动识别", "auto"), ("是样本/蛋白 ID", "first"), ("是数值数据", "none")]),
            ("layer", "AnnData 数据层", [("X（默认矩阵）", None)]),
            ("missing_values", "缺失值编码", [("NaN / 空值", "nan"), ("NaN 或零值", "zero"), ("NaN 或非正值", "nonpositive")]),
        ]:
            self.field(form, key, label, self.options(items))
        content.addWidget(self.label("数字型 ID 和数字型表头需要手动指定。", True))
        self.field(form, "max_dense_mb", "稠密矩阵上限（MB）", control(("", 2048, 1, 131072)))
        self.field(form, "output_scale", "输出矩阵尺度", self.options([("工作尺度", "working"), ("原始输入尺度", "input")]))
        content.addStretch()
        form, content = forms[1]
        self.field(form, "training_engine", "JINGWEI 训练引擎", self.options([("PyTorch（较快）", "torch"), ("Lightning（兼容）", "lightning")]))
        for key, spec in NEURAL.items():
            self.field(form, key, spec[0], control(spec))
        if cuda_supported():
            gpu = QSpinBox()
            gpu.setRange(-1, 128)
            gpu.setValue(-1)
            self.bind(gpu, "默认设备", "setSpecialValueText")
            self.field(form, "gpu_id", "GPU 编号", gpu)
        content.addWidget(self.label("单方法参数优先于通用训练设置。传统方法使用 CPU。", True))
        content.addStretch()
        form, content = forms[2]
        self.field(form, "training_mechanism", "训练掩蔽机制", self.options([("与评测相同", None), *mechanisms]))
        self.field(form, "training_mask_rate", "训练掩蔽比例", FractionControl(0.2, 0.0, 1.0))
        self.field(form, "seed", "随机种子", control(("", 42, 0, 2147483647)))
        self.field(form, "mask_level", "缺失概率的丰度水平", self.options([("每个矩阵条目", "entry"), ("每个蛋白的均值", "feature-mean")]))
        for key, spec in {
            "logistic_slope": ("Logistic 斜率（非正）", -1., -100., 0.),
            "low_quantile": ("低丰度分位数", .2, 0., .9999), "high_quantile": ("高丰度分位数", .6, .0001, 1.),
            "low_probability": ("低区间缺失概率", .9, 0., 1.), "high_probability": ("高区间缺失概率", .2, 0., 1.),
            "threshold_probability": ("阈值以外缺失概率", .05, 0., 1.),
        }.items():
            self.field(form, key, spec[0], control(spec))
        for key, source in (("paper_probabilities", "使用未校准概率"), ("allow_empty_marginals", "允许全缺失样本或蛋白")):
            box = self.bind(QCheckBox(), source)
            self.fields[key] = box
            form.addRow(box)
        self.bind(self.fields["paper_probabilities"], "实际掩蔽率可能与指定比例不同。", "setToolTip")
        content.addWidget(self.button("W2 参考蛋白…", self.choose_features))
        self.w2_note = self.label("W2：所有蛋白", True)
        self.bind(self.w2_note, "W2 比较样本的多维蛋白分布；仅使用所选蛋白上原始完整的参考样本。未定义值留空。", "setToolTip")
        content.addWidget(self.w2_note)
        content.addStretch()
        form, content = forms[3]
        self.field(form, "baseline_engine", "对比方法执行方式", self.options([("优化执行", "optimized"), ("原生执行（复现）", "native")]))
        self.field(form, "mice_backend", "MICE 实现", self.options([("Python · 线性 PMM", "python"), ("R · 原版 mice", "r")]))
        self.fields["mice_backend"].currentIndexChanged.connect(lambda *_: self.refresh_environment())
        self.field(form, "threads", "CPU 线程数", control(("", 4, 1, 256)))
        self.field(form, "timeout", "单方法超时（秒）", control(("", DEFAULT_TIMEOUT_SECONDS, 1, 864000)))
        for key, source, directory in (("rscript", "Rscript 可执行文件", False), ("r_library", "R 包目录", True), ("naguider_source", "NAguideR 源码目录", True), ("dreamai_source", "DreamAI 源码目录", True)):
            row = QHBoxLayout()
            entry = self.bind(QLineEdit(), "自动查找", "setPlaceholderText")
            row.addWidget(entry, 1)
            row.addWidget(self.button("选择", lambda checked=False, e=entry, d=directory: self.choose_backend(e, d)))
            self.fields[key] = entry
            form.addRow(self.label(source), row)
        self.environment_text = self.label("正在检查计算环境…", True)
        content.addWidget(self.environment_text)
        content.addWidget(self.button("重新检查", self.refresh_environment))
        self.environment_table = QTableWidget(0, 3)
        self.environment_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.environment_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)
        content.addWidget(self.environment_table, 1)
        location = QLabel(str(self.store.root))
        location.setTextFormat(Qt.PlainText)
        location.setWordWrap(True)
        location.setTextInteractionFlags(Qt.TextSelectableByMouse)
        content.addWidget(location)
        self.dialog_buttons(layout, self.advanced_dialog, QDialogButtonBox.Close)

    def dialog_buttons(self, layout, dialog, flags):
        buttons = QDialogButtonBox(flags)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        for flag, source in ((QDialogButtonBox.Ok, "确定"), (QDialogButtonBox.Cancel, "取消"), (QDialogButtonBox.Close, "关闭"), (QDialogButtonBox.Reset, "重置")):
            button = buttons.button(flag)
            if button:
                self.bind(button, source)
        layout.addWidget(buttons)
        return buttons

    def show_advanced(self, index=1):
        from .gui import value
        if isinstance(index, bool):
            index = 1
        keys = ("rscript", "r_library", "naguider_source", "dreamai_source")
        before = [value(self.fields[key]) for key in keys]
        self.advanced_tabs.setCurrentIndex(index)
        self.advanced_dialog.exec()
        if before != [value(self.fields[key]) for key in keys]:
            self.refresh_environment()
        if self.input_path.text() and self.preview_data is None:
            self.inspect_data()

    def show_preview(self):
        if self.preview_data:
            self.preview_dialog.exec()

    def current_method_changed(self, item, previous=None):
        name = item.data(0, Qt.UserRole) if item else None
        self.parameter_button.setEnabled(bool(PARAMETERS.get(name)))

    def edit_current_method(self):
        item = self.method_tree.currentItem()
        name = item.data(0, Qt.UserRole) if item else None
        if name and PARAMETERS.get(name):
            self.edit_parameters(name)

    def update_method_count(self):
        self.method_count.setText(self.t("已选 {count} 种", count=sum(box.isChecked() for box in self.method_checks.values())))

    def update_protocol_summary(self):
        from .gui import value
        rate = value(self.fields["training_mask_rate"])
        seed = value(self.fields["seed"])
        mechanism = value(self.fields["training_mechanism"])
        self.protocol_summary.setText(self.t("训练掩蔽 {rate:.0%} · 种子 {seed}", rate=rate, seed=seed) + (" · " + str(mechanism) if mechanism else ""))

    def update_data_labels(self):
        if self.preview_data:
            data = self.preview_data
            self.data_summary.setText(self.t("{samples:,} 样本 × {features:,} 蛋白 · 缺失 {missing:.1%}", samples=data["samples"], features=data["features"], missing=data["missing_rate"]))
            details = self.t("尺度：{scale} · 完整参考样本：{complete}", scale=data["working_scale"], complete=data["complete_rows"])
            detection = data["csv_detection"]
            if detection:
                details += "\n" + self.t("表头：{header} · ID 列：{index}", header=self.t("是" if detection["has_header"] else "否"), index=self.t("是" if detection["has_id_column"] else "否"))
            self.preview_details.setText(details)
            self.data_summary.setToolTip(details)
        else:
            self.data_summary.setText(self.t("正在读取并检查数据…" if self._inspecting else "数据或读取设置已更改，请检查数据。" if self.input_path.text() else "尚未载入数据"))
        self.preview_button.setEnabled(self.preview_data is not None)
        self.w2_note.setText(self.t("W2：{count} 个蛋白", count=len(self.feature_selection)) if self.feature_selection else self.t("W2：所有蛋白"))

    def update_environment_labels(self):
        if self._environment_checking:
            label = self.t("正在检查计算环境…")
        else:
            count = sum(state["available"] for state in self.backend_states.values())
            label = self.t("{count}/{total} 种方法可用 · {device}", count=count, total=len(METHODS), device=self._gpu_name or "CPU")
        self.env_badge.setText(label)
        self.environment_text.setText(label)
        self.environment_table.setHorizontalHeaderLabels([self.t(source) for source in ("方法", "后端", "状态")])
        from PySide6.QtWidgets import QTableWidgetItem
        self.environment_table.setRowCount(len(self.backend_states))
        for i, (name, state) in enumerate(self.backend_states.items()):
            label = self.t("可用") if state["available"] else str(state["reason"])
            self.method_checks[name].setToolTip(name + " · " + label)
            for j, display in enumerate((name, state["backend"], label)):
                self.environment_table.setItem(i, j, QTableWidgetItem(str(display)))

    def set_language(self, language):
        if language not in ("zh", "en"):
            return
        selected_row = self.result_table.currentRow()
        self.language = language
        previous = self.language_choice.blockSignals(True)
        self.language_choice.setCurrentIndex(self.language_choice.findData(language))
        self.language_choice.blockSignals(previous)
        for widget, setter, source in self._translations:
            getattr(widget, setter)(self.t(source))
        for widget, items in self._combo_translations:
            # AnnData layers are discovered dynamically; keep their data and names.
            for source, data in items:
                index = widget.findData(data)
                if index >= 0:
                    widget.setItemText(index, self.t(source))
        for name, box in self.method_checks.items():
            box.item.setText(0, self.t(LABELS.get(name, name)))
        self.method_groups["python"].setText(0, self.t("传统方法"))
        for i, source in enumerate(("结果", "图表", "日志")):
            self.result_tabs.setTabText(i, self.t(source))
        for i, source in enumerate(("导入", "训练", "掩蔽与 W2", "运行环境")):
            self.advanced_tabs.setTabText(i, self.t(source))
        self.update_data_labels()
        self.update_environment_labels()
        self.update_method_count()
        self.update_protocol_summary()
        self._last_result_digest = None
        self.refresh_jobs()
        if selected_row >= 0:
            self.result_table.selectRow(selected_row)
        if not self.selected_job:
            self.plot.empty(self.t("暂无可比较数据"))
            self.progress_label.setText(self.t("尚无任务"))
        write_json(self.store.root / "ui.json", {"language": language})
