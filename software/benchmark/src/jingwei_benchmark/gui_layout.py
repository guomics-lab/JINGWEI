"""Clean native workbench layout and live language bindings."""

from pathlib import Path
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QFormLayout, QLabel, QPushButton, QComboBox,
    QLineEdit, QSpinBox, QCheckBox, QGroupBox, QDialog, QDialogButtonBox, QTabWidget,
    QTreeWidget, QTreeWidgetItem, QTableWidget, QHeaderView, QAbstractItemView,
    QProgressBar, QPlainTextEdit, QSplitter, QScrollArea, QStackedWidget, QFrame, QStyle,
)

from .gui_i18n import translate
from .cli import DEFAULT_TIMEOUT_SECONDS
from .gui_parameters import NEURAL, PARAMETERS, LABELS
from .methods import METHODS
from .platforms import cuda_supported
from .protocol import ORIGINAL_MECHANISM, JINGWEI_MECHANISM, uses_mask_rate


STYLE = """
QWidget {
    font-family: "Microsoft YaHei UI", "Segoe UI", sans-serif;
    font-size: 12px;
    color: #1E293B;
}
QMainWindow, QDialog {
    background: #F5F7FA;
}
QGroupBox {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 6px;
    margin-top: 9px;
    padding: 13px 11px 10px 11px;
    font-weight: 600;
    color: #0F172A;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 5px;
    color: #334155;
}
QPushButton {
    background: #FFFFFF;
    border: 1px solid #CBD5E1;
    border-radius: 5px;
    padding: 4px 11px;
    min-height: 20px;
    color: #1E293B;
}
QPushButton:hover {
    background: #F1F5F9;
    border-color: #94A3B8;
}
QPushButton:pressed {
    background: #E2E8F0;
}
QPushButton:disabled {
    color: #94A3B8;
    background: #F8FAFC;
    border-color: #E2E8F0;
}
QPushButton#subtle {
    padding: 2px 9px;
    min-height: 18px;
    color: #334155;
}
QPushButton#primary {
    background: #1D5FA7;
    color: #FFFFFF;
    border-color: #1D5FA7;
    font-weight: 600;
    font-size: 13px;
}
QPushButton#primary:hover {
    background: #164B87;
    border-color: #164B87;
}
QPushButton#primary:disabled {
    background: #9BB5D3;
    border-color: #9BB5D3;
    color: #F8FAFC;
}
QLineEdit {
    background: #FFFFFF;
    border: 1px solid #CBD5E1;
    border-radius: 5px;
    padding: 4px 7px;
    min-height: 20px;
}
QComboBox {
    background: #FFFFFF;
    min-height: 24px;
    padding: 2px 6px;
}
QSpinBox, QDoubleSpinBox {
    background: #FFFFFF;
    min-height: 24px;
    padding-left: 5px;
}
QLineEdit:focus {
    border-color: #1D5FA7;
}
QLineEdit:disabled, QComboBox:disabled, QSpinBox:disabled, QDoubleSpinBox:disabled {
    color: #94A3B8;
    background: #F8FAFC;
}
QCheckBox:disabled {
    color: #94A3B8;
    background: transparent;
}
QTreeWidget, QTableView, QTableWidget, QPlainTextEdit, QListWidget {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 5px;
    selection-background-color: #E8F1FB;
    selection-color: #163A63;
    gridline-color: #F1F5F9;
    alternate-background-color: #FAFBFD;
}
QTreeWidget::item {
    height: 24px;
    padding: 1px 2px;
}
QHeaderView::section {
    background: #F8FAFC;
    color: #475569;
    font-weight: 600;
    border: 0;
    border-bottom: 1px solid #E2E8F0;
    padding: 6px 12px;
}
QTabWidget::pane {
    border: 1px solid #E2E8F0;
    border-radius: 6px;
    background: #FFFFFF;
    top: -1px;
}
QTabBar::tab {
    background: #EDF1F6;
    color: #64748B;
    padding: 7px 16px;
    border: 1px solid #E2E8F0;
    border-bottom: none;
    border-top-left-radius: 5px;
    border-top-right-radius: 5px;
    margin-right: 3px;
    font-weight: 600;
}
QTabBar::tab:selected {
    background: #FFFFFF;
    color: #1D5FA7;
}
QTabBar::tab:hover:!selected {
    background: #F8FAFC;
    color: #334155;
}
QProgressBar {
    border: none;
    border-radius: 4px;
    background: #E2E8F0;
    max-height: 7px;
    min-height: 7px;
}
QProgressBar::chunk {
    background: #1D5FA7;
    border-radius: 4px;
}
QLabel#muted {
    color: #64748B;
}
QLabel#brand {
    font-size: 17px;
    font-weight: 700;
    color: #0F172A;
    letter-spacing: 0.2px;
}
QLabel#taskTitle {
    font-size: 13px;
    font-weight: 600;
    color: #1E293B;
}
QLabel#statusPill {
    background: #E8F1FB;
    color: #1D5FA7;
    border-radius: 4px;
    padding: 2px 8px;
    font-weight: 600;
    font-size: 11px;
}
QFrame#divider {
    background: #EDF2F7;
    max-height: 1px;
    min-height: 1px;
    border: none;
}
QPlainTextEdit {
    font-family: Consolas, "Cascadia Mono", monospace;
    font-size: 12px;
    color: #1E293B;
    padding: 6px;
}
QSplitter::handle {
    background: transparent;
    width: 8px;
}
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

    def button(self, source, action, subtle=False):
        button = self.bind(QPushButton(), source)
        if subtle:
            button.setObjectName("subtle")
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
        if not hasattr(self, "field_labels"):
            self.field_labels = {}
        label = self.label(source)
        self.field_labels[key] = label
        form.addRow(label, widget)
        return widget

    def group(self, source, parent_layout):
        group = self.bind(QGroupBox(), source, "setTitle")
        parent_layout.addWidget(group)
        layout = QVBoxLayout(group)
        layout.setSpacing(7)
        return layout

    def setup_ui(self):
        from .gui import RunTable, RunTableModel, MetricPlot, FractionControl
        from .windows_identity import apply_application_icon
        from .gui_language import initial_language
        self.language = initial_language(self.store.root)
        self._translations, self._combo_translations = [], []
        self.fields = {}
        self._gpu_name = None
        self._environment_checking = True
        self._inspecting = False
        from . import __version__
        self.setWindowTitle("jingwei " + __version__)
        app = QApplication.instance()
        # A light native widget palette keeps unchecked boxes legible even when
        # Windows uses dark mode. Fusion gives stable metrics across DPI scales.
        app.setStyle("Fusion")
        app.styleHints().setColorScheme(Qt.ColorScheme.Light)
        apply_application_icon(app, self)
        self.logo = self.windowIcon().pixmap(64, 64)
        self.resize(1280, 800)
        self.setMinimumSize(1080, 700)
        self.setStyleSheet(STYLE)

        root = QWidget()
        layout = QVBoxLayout(root)
        layout.setContentsMargins(14, 10, 14, 8)
        layout.setSpacing(8)

        # Clean top bar: Brand on left, Advanced + Language on right (no duplicate Runtime button)
        toolbar = QHBoxLayout()
        toolbar.setSpacing(8)
        badge = QLabel()
        badge.setPixmap(self.logo.scaled(28, 28, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        toolbar.addWidget(badge)
        brand = QLabel("jingwei")
        self.brand_label = brand
        brand.setObjectName("brand")
        toolbar.addWidget(brand)
        toolbar.addStretch()
        self.advanced_button = self.button("高级设置", self.show_advanced)
        toolbar.addWidget(self.advanced_button)
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
        self.main_splitter = splitter

        # Left configuration sidebar
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 2, 0)
        left_layout.setSpacing(8)
        self.configuration_panel = QWidget()
        config = QVBoxLayout(self.configuration_panel)
        config.setContentsMargins(0, 0, 0, 0)
        config.setSpacing(8)

        # 1. Data card (streamlined: Browse + Preview; automatic inspection on change)
        data = self.group("数据", config)
        row = QHBoxLayout()
        row.setSpacing(6)
        self.input_path = self.bind(QLineEdit(), "选择 CSV 或 AnnData (.h5ad)", "setPlaceholderText")
        self.recheck_action = self.input_path.addAction(
            self.style().standardIcon(QStyle.SP_BrowserReload),
            QLineEdit.TrailingPosition,
        )
        self.bind(self.recheck_action, "重新检查数据", "setToolTip")
        self.recheck_action.triggered.connect(self.recheck_input)
        row.addWidget(self.input_path, 1)
        row.addWidget(self.button("选择文件", self.choose_input))
        data.addLayout(row)
        form = QFormLayout()
        form.setVerticalSpacing(6)
        form.setHorizontalSpacing(10)
        self.field(form, "scale", "输入尺度", self.options([("原始正丰度", "raw"), ("已经 log2 转换", "log2")]))
        self.field(form, "transform", "预处理", self.options([("自动（原始丰度转为 log2）", "auto"), ("不转换", "none"), ("log2 转换", "log2")]))
        data.addLayout(form)
        row = QHBoxLayout()
        row.setSpacing(6)
        self.data_summary = self.label("尚未载入数据", True)
        self.data_summary.setWordWrap(True)
        row.addWidget(self.data_summary, 1)
        self.preview_button = self.button("预览", self.show_preview, subtle=True)
        self.preview_button.setEnabled(False)
        row.addWidget(self.preview_button)
        data.addLayout(row)

        # 2. Methods card
        methods = self.group("方法", config)
        row = QHBoxLayout()
        row.setSpacing(5)
        self.method_count = QLabel()
        self.method_count.setObjectName("muted")
        row.addWidget(self.method_count, 1)
        row.addWidget(self.button("全选", lambda: self.select_methods(None), subtle=True))
        row.addWidget(self.button("清空", lambda: self.select_methods([]), subtle=True))
        self.parameter_button = self.button("参数…", self.edit_current_method, subtle=True)
        self.parameter_button.setEnabled(False)
        row.addWidget(self.parameter_button)
        methods.addLayout(row)
        self.method_tree = QTreeWidget()
        self.method_tree.setHeaderHidden(True)
        self.method_tree.setMinimumHeight(150)
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
        self.method_tree.itemChanged.connect(self._on_method_item_changed)
        self.method_tree.currentItemChanged.connect(self.current_method_changed)
        self.method_tree.itemDoubleClicked.connect(lambda item, _: self.edit_current_method() if item.data(0, Qt.UserRole) else None)
        if "jingwei-dmf" in self.method_checks:
            self.method_tree.setCurrentItem(self.method_checks["jingwei-dmf"].item)
        methods.addWidget(self.method_tree, 1)
        config.setStretch(1, 1)

        # 3. Run settings card (3 core protocol rows + integrated output folder)
        settings = self.group("运行设置", config)
        form = QFormLayout()
        form.setVerticalSpacing(6)
        form.setHorizontalSpacing(10)
        self.run_settings_form = form
        self.field(form, "command", "任务", self.options([("评测", "benchmark"), ("插补", "impute")]))
        mechanisms = [("MCAR", "mcar"), ("MNAR · JINGWEI", JINGWEI_MECHANISM), ("MNAR · Logistic", "mnar-logistic"), ("MNAR · Threshold", "mnar-threshold"), ("MNAR · 分位数（自定义）", "mnar-jingwei"), ("历史规则（兼容旧任务）", ORIGINAL_MECHANISM)]
        self.field(form, "mechanism", "缺失机制", self.options(mechanisms))
        self.bind(self.fields["mechanism"], "JINGWEI：按每个蛋白的丰度构造缺失概率，低丰度值更易缺失，并校准到所选比例。", "setToolTip")
        self.mask_rate_stack = QStackedWidget()
        self.fields["mask_rate"] = FractionControl(0.2, 0.0001, 0.9999)
        self.mask_rate_stack.addWidget(self.fields["mask_rate"])
        self.automatic_mask_rate = self.label("固定概率")
        self.bind(self.automatic_mask_rate, "使用 get_mask() 的固定概率 0.9 / 0.5 / 0.3，不使用自定义掩蔽比例；实际比例写入运行日志。", "setToolTip")
        self.mask_rate_stack.addWidget(self.automatic_mask_rate)
        self.field_labels["mask_rate"] = self.label("掩蔽比例")
        form.addRow(self.field_labels["mask_rate"], self.mask_rate_stack)
        settings.addLayout(form)

        divider = QFrame()
        divider.setObjectName("divider")
        settings.addWidget(divider)

        output = QHBoxLayout()
        output.setSpacing(6)
        output.addWidget(self.label("输出目录"))
        self.output_root = QLineEdit(str(self.default_output_root))
        self.bind(self.output_root, "每次运行会在所选目录创建独立的任务文件夹。", "setToolTip")
        output.addWidget(self.output_root, 1)
        output.addWidget(self.button("选择", self.choose_output, subtle=True))
        settings.addLayout(output)

        left_layout.addWidget(self.configuration_panel, 1)

        # Primary execution buttons
        row = QHBoxLayout()
        row.setSpacing(8)
        self.start_button = self.button("开始", self.start_job)
        self.start_button.setObjectName("primary")
        self.start_button.setMinimumHeight(32)
        self.start_button.setEnabled(False)
        self.cancel_button = self.button("停止", self.cancel_job)
        self.cancel_button.setMinimumHeight(32)
        self.cancel_button.setEnabled(False)
        row.addWidget(self.start_button, 3)
        row.addWidget(self.cancel_button, 1)
        left_layout.addLayout(row)
        splitter.addWidget(left)

        # Right results workspace
        right = QWidget()
        results = QVBoxLayout(right)
        results.setContentsMargins(6, 4, 0, 0)
        results.setSpacing(7)

        # Consolidated single-row task header (no stacked duplicate "尚无任务")
        header = QHBoxLayout()
        header.setSpacing(8)
        self.current_task_label = self.label("尚无任务", True)
        self.current_task_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        header.addWidget(self.current_task_label)
        self.progress_label = self.label("尚无任务")
        self.progress_label.setObjectName("statusPill")
        self.progress_label.hide()
        header.addWidget(self.progress_label)
        header.addStretch(1)
        self.job_actions = []
        for label, action in (("打开目录", self.open_results), ("导出任务", self.export_job), ("复制配置", self.clone_job)):
            button = self.button(label, action, subtle=True)
            button.setEnabled(False)
            header.addWidget(button)
            self.job_actions.append(button)
            if label == "导出任务":
                self.export_job_button = button
            if label == "复制配置":
                self.clone_button = button
        results.addLayout(header)

        self.progress = QProgressBar()
        self.progress.setTextVisible(False)
        self.progress.setValue(0)
        self.progress.hide()
        results.addWidget(self.progress)

        self.result_tabs = QTabWidget()
        self.result_table = RunTable()
        self.run_model = RunTableModel(self.t)
        self.result_table.setModel(self.run_model)
        self.run_model.replace([], ["repeat", "method", "status", "MSE", "RMSE", "MAE", "NRMSE", "Pearson", "Spearman", "W2", "time_s"])
        self.result_table.setColumnHidden(0, True)
        self.result_table.setAlternatingRowColors(True)
        self.result_table.setShowGrid(False)
        self.result_table.verticalHeader().setVisible(False)
        self.result_table.verticalHeader().setDefaultSectionSize(30)
        self.result_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.result_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.result_table.horizontalHeader().setStretchLastSection(False)
        self.result_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.result_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.result_table.selectionModel().selectionChanged.connect(self.show_method_log)
        self.result_tabs.addTab(self.result_table, self.t("结果"))

        chart = QWidget()
        chart_layout = QVBoxLayout(chart)
        chart_layout.setContentsMargins(12, 10, 12, 10)
        chart_layout.setSpacing(8)
        controls = QHBoxLayout()
        controls.setSpacing(8)
        self.metric_choice = self.options([(name, name) for name in ("MSE", "RMSE", "MAE", "NRMSE", "Pearson", "Spearman", "W2")] + [("耗时（秒）", "time_s")])
        self.metric_choice.setMinimumWidth(165)
        self.metric_choice.currentIndexChanged.connect(self._on_metric_choice_changed)
        controls.addWidget(self.metric_choice)
        self.paired = self.bind(QCheckBox(), "共同重复")
        self.paired.setChecked(True)
        self.paired.hide()
        self.bind(self.paired, "仅比较所有所选方法均成功完成的重复；取消勾选后，各方法使用各自的成功重复。", "setToolTip")
        self.paired.stateChanged.connect(self.render_results)
        controls.addWidget(self.paired)
        controls.addStretch(1)
        controls.addWidget(self.button("导出图表", self.export_plot))
        chart_layout.addLayout(controls)
        self.plot = MetricPlot(self.t)
        chart_scroll = QScrollArea()
        chart_scroll.setWidgetResizable(True)
        chart_scroll.setFrameShape(QScrollArea.NoFrame)
        chart_scroll.setStyleSheet("QScrollArea, QScrollArea > QWidget > QWidget { background: #FFFFFF; border: none; }")
        chart_scroll.setAlignment(Qt.AlignTop)
        chart_scroll.setWidget(self.plot)
        chart_layout.addWidget(chart_scroll, 1)
        self.result_tabs.addTab(chart, self.t("图表"))

        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setFrameShape(QFrame.NoFrame)
        self.log_view.setMaximumBlockCount(2000)
        self.result_tabs.addTab(self.log_view, self.t("日志"))
        results.addWidget(self.result_tabs, 1)

        self.result_notice = QLabel("")
        self.result_notice.setObjectName("muted")
        self.result_notice.setTextFormat(Qt.PlainText)
        self.bind(self.result_notice, "图表显示均值与样本标准差；n=1 时不显示误差线。失败与未定义指标保留在结果表中。", "setToolTip")
        self.result_notice.hide()
        results.addWidget(self.result_notice)

        splitter.addWidget(right)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([415, 835])
        layout.addWidget(splitter, 1)

        # Clean single bottom status bar (replaces lazy QMainWindow.statusBar() popups)
        footer = QHBoxLayout()
        footer.setContentsMargins(2, 0, 2, 0)
        self.env_badge = self.label("正在检查计算环境…", True)
        footer.addWidget(self.env_badge)
        footer.addStretch(1)
        self.status_notice = QLabel("")
        self.status_notice.setObjectName("muted")
        self.status_notice.setTextFormat(Qt.PlainText)
        footer.addWidget(self.status_notice)
        layout.addLayout(footer)

        self.setCentralWidget(root)
        self.build_advanced(mechanisms)

        self.preview_dialog = QDialog(self)
        self.preview_dialog.resize(860, 480)
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
        self.input_path.returnPressed.connect(self.recheck_input)
        self.input_path.editingFinished.connect(self.on_input_editing_finished)
        for key in ("scale", "transform"):
            self.fields[key].currentIndexChanged.connect(self.on_input_setting_changed)
        for key in ("orientation", "header", "index_column", "layer", "missing_values"):
            self.fields[key].currentIndexChanged.connect(self.invalidate_preview)
        self.fields["max_dense_mb"].valueChanged.connect(self.invalidate_preview)
        self.fields["command"].currentIndexChanged.connect(self.mode_changed)
        self.fields["command"].currentIndexChanged.connect(self.protocol_changed)
        self.fields["device"].currentIndexChanged.connect(self.device_changed)
        self.fields["training_policy"].currentIndexChanged.connect(self.training_policy_changed)
        self.fields["mechanism"].currentIndexChanged.connect(self.update_mask_controls)
        self.fields["paper_probabilities"].toggled.connect(self.update_mask_controls)
        self.mode_changed()
        self.device_changed()
        self.training_policy_changed()
        self.update_method_count()

    def _sync_method_tree_selection(self, checked_names):
        checked_set = set(checked_names)
        for group in self.method_groups.values():
            if any(group.child(i).data(0, Qt.UserRole) in checked_set for i in range(group.childCount())):
                group.setExpanded(True)
        current = self.method_tree.currentItem()
        current_name = current.data(0, Qt.UserRole) if current else None
        if checked_names and current_name not in checked_set:
            first = next((n for n in checked_names if n in self.method_checks), None)
            if first:
                self.method_tree.setCurrentItem(self.method_checks[first].item)
        self.current_method_changed(self.method_tree.currentItem())

    def _on_method_item_changed(self, item, column=0):
        name = item.data(0, Qt.UserRole) if item else None
        if (name and name in self.backend_states
                and not self.backend_states[name]["available"]
                and item.checkState(0) == Qt.Unchecked
                and not item.isDisabled()):
            previous = self.method_tree.blockSignals(True)
            item.setDisabled(True)
            self.method_tree.blockSignals(previous)
        if name and item.checkState(0) == Qt.Checked and self.method_tree.currentItem() is not item:
            self.method_tree.setCurrentItem(item)
        else:
            self.current_method_changed(self.method_tree.currentItem())
        self.update_method_count()

    def _is_existing_file(self, raw):
        from .gui import clean_path_text
        cleaned = clean_path_text(raw)
        if not cleaned:
            return False
        try:
            return Path(cleaned).is_file()
        except OSError:
            return False

    def recheck_input(self):
        from .gui import clean_path_text
        if self.configuration_locked() or self._inspecting:
            return
        cleaned = clean_path_text(self.input_path.text())
        if cleaned != self.input_path.text():
            previous = self.input_path.blockSignals(True)
            self.input_path.setText(cleaned)
            self.input_path.blockSignals(previous)
        self.input_path.setModified(False)
        self.invalidate_preview()
        self.inspect_data()

    def on_input_editing_finished(self):
        from .gui import clean_path_text
        cleaned = clean_path_text(self.input_path.text())
        if cleaned != self.input_path.text():
            previous = self.input_path.blockSignals(True)
            self.input_path.setText(cleaned)
            self.input_path.blockSignals(previous)
        self.input_path.setModified(False)
        if cleaned and self.preview_data is None and not self._inspecting:
            if self._is_existing_file(cleaned):
                self.inspect_data()
            else:
                self.update_data_labels()

    def on_input_setting_changed(self, *args):
        from .gui import clean_path_text
        self.invalidate_preview()
        cleaned = clean_path_text(self.input_path.text())
        if cleaned and self._is_existing_file(cleaned):
            self.inspect_data()

    def build_advanced(self, mechanisms):
        from .gui import control
        self.advanced_dialog = QDialog(self)
        self.bind(self.advanced_dialog, "高级设置", "setWindowTitle")
        self.advanced_dialog.resize(680, 620)
        layout = QVBoxLayout(self.advanced_dialog)
        self.advanced_tabs = QTabWidget()
        layout.addWidget(self.advanced_tabs, 1)
        forms = []
        for source in ("导入", "训练", "掩蔽与 W2", "运行环境"):
            page = QWidget()
            content = QVBoxLayout(page)
            content.setContentsMargins(14, 12, 14, 12)
            content.setSpacing(10)
            form = QFormLayout()
            form.setVerticalSpacing(8)
            form.setHorizontalSpacing(12)
            content.addLayout(form)
            forms.append((form, content))
            self.advanced_tabs.addTab(page, self.t(source))

        # Tab 0: Import & Naming
        form, content = forms[0]
        for key, label, items in [
            ("orientation", "矩阵方向", [("样本 × 蛋白", "samples-by-features"), ("蛋白 × 样本（CSV）", "features-by-samples")]),
            ("header", "CSV 第一行", [("自动识别", "auto"), ("是表头", "yes"), ("是数值数据", "no")]),
            ("index_column", "CSV 第一列", [("自动识别", "auto"), ("是样本/蛋白 ID", "first"), ("是数值数据", "none")]),
            ("layer", "AnnData 数据层", [("X（默认矩阵）", None)]),
            ("missing_values", "缺失值编码", [("NaN / 空值", "nan"), ("NaN 或零值", "zero"), ("NaN 或非正值", "nonpositive")]),
        ]:
            self.field(form, key, label, self.options(items))
        self.field(form, "max_dense_mb", "稠密矩阵上限（MB）", control(("", 2048, 1, 131072)))
        self.field(form, "output_scale", "输出矩阵尺度", self.options([("工作尺度", "working"), ("原始输入尺度", "input")]))
        self.job_name = self.bind(QLineEdit(), "任务名称（留空自动生成）", "setPlaceholderText")
        form.addRow(self.label("任务名称"), self.job_name)
        content.addWidget(self.label("数字型 ID 和数字型表头需要手动指定。", True))
        content.addStretch()

        # Tab 1: Training
        form, content = forms[1]
        devices = [("自动", "auto"), ("CPU", "cpu"), ("NVIDIA CUDA", "cuda")] if cuda_supported() else [("CPU", "cpu")]
        self.field(form, "device", "计算设备", self.options(devices))
        self.field(form, "jingwei_preset", "JINGWEI 参数预设", self.options([("通用设置", "standard"), ("原始 benchmark 参数", "original-benchmark")]))
        self.field(form, "training_engine", "JINGWEI 训练引擎", self.options([("PyTorch（较快）", "torch"), ("Lightning（兼容）", "lightning")]))
        self.field(form, "training_policy", "训练终止方式", self.options([
            ("固定轮数", "fixed"),
            ("验证集收敛", "convergence")]))
        for key, spec in NEURAL.items():
            self.field(form, key, spec[0], control(spec))
        if cuda_supported():
            gpu = QSpinBox()
            gpu.setRange(-1, 128)
            gpu.setValue(-1)
            self.bind(gpu, "默认设备", "setSpecialValueText")
            self.field(form, "gpu_id", "GPU 编号", gpu)
        content.addWidget(self.label("参数优先级：单方法参数 > 原始参数预设 > 通用设置。传统方法使用 CPU。", True))
        content.addStretch()

        # Tab 2: Masking & W2
        form, content = forms[2]
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
            form.addRow("", box)
        self.bind(self.fields["paper_probabilities"], "实际掩蔽率可能与指定比例不同。", "setToolTip")
        w2_row = QHBoxLayout()
        self.w2_note = self.label("W2：所有蛋白", True)
        self.bind(self.w2_note, "W2 比较样本的多维蛋白分布；仅使用所选蛋白上原始完整的参考样本。未定义值留空。", "setToolTip")
        w2_row.addWidget(self.w2_note, 1)
        w2_row.addWidget(self.button("W2 参考蛋白…", self.choose_features))
        content.addLayout(w2_row)
        content.addStretch()

        # Tab 3: Runtime
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
            row.addWidget(self.button("选择", lambda checked=False, e=entry, d=directory: self.choose_backend(e, d), subtle=True))
            self.fields[key] = entry
            form.addRow(self.label(source), row)
        env_row = QHBoxLayout()
        self.environment_text = self.label("正在检查计算环境…", True)
        env_row.addWidget(self.environment_text, 1)
        env_row.addWidget(self.button("重新检查", lambda: self.refresh_environment(force=True)))
        content.addLayout(env_row)
        self.environment_table = QTableWidget(0, 3)
        self.environment_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.environment_table.verticalHeader().setVisible(False)
        self.environment_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.environment_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.environment_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)
        content.addWidget(self.environment_table, 1)

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
        from .gui import value, clean_path_text
        if isinstance(index, bool):
            index = 1
        keys = ("rscript", "r_library", "naguider_source", "dreamai_source")
        before = [clean_path_text(value(self.fields[key])) for key in keys]
        self.advanced_tabs.setCurrentIndex(index)
        self.advanced_dialog.exec()
        if before != [clean_path_text(value(self.fields[key])) for key in keys]:
            self.refresh_environment()
        cleaned = clean_path_text(self.input_path.text())
        if cleaned and self._is_existing_file(cleaned) and not self._inspecting:
            needs_inspect = self.preview_data is None
            if not needs_inspect:
                try:
                    needs_inspect = self.signature() != self.preview_signature
                except OSError:
                    needs_inspect = True
            if needs_inspect:
                self.inspect_data()

    def show_preview(self):
        if self.preview_data:
            self.preview_dialog.exec()

    def current_method_changed(self, item, previous=None):
        name = item.data(0, Qt.UserRole) if item else None
        enabled = bool(PARAMETERS.get(name)) and item is not None and (
            not item.isDisabled() or item.checkState(0) == Qt.Checked or not self.backend_states
        )
        self.parameter_button.setEnabled(enabled)

    def edit_current_method(self):
        item = self.method_tree.currentItem()
        name = item.data(0, Qt.UserRole) if item else None
        if name and PARAMETERS.get(name) and self.parameter_button.isEnabled():
            self.edit_parameters(name)

    def update_method_count(self):
        self.method_count.setText(self.t("已选 {count} 种", count=sum(box.isChecked() for box in self.method_checks.values())))

    def update_mask_controls(self):
        from .gui import value
        self.current_method_changed(self.method_tree.currentItem())
        for name, box in self.method_checks.items():
            box.item.setText(0, self.t(self.method_label(name)))
        evaluation = value(self.fields["mechanism"])
        benchmark = value(self.fields["command"]) == "benchmark"
        self.fields["mechanism"].setEnabled(True)

        # Impute mode allows 0% training mask rate; Benchmark requires > 0%
        mask_spin = self.fields["mask_rate"]
        prev_block = mask_spin.blockSignals(True)
        mask_spin.setMinimum(0.01 if benchmark else 0.0)
        mask_spin.blockSignals(prev_block)

        uncalibrated = value(self.fields["paper_probabilities"])
        controlled = uses_mask_rate(evaluation, uncalibrated)
        mask_spin.setEnabled(controlled)
        self.mask_rate_stack.setCurrentIndex(0 if controlled else 1)
        self.automatic_mask_rate.setToolTip(self.t(
            "使用 get_mask() 的固定概率 0.9 / 0.5 / 0.3，不使用自定义掩蔽比例；实际比例写入运行日志。"
            if evaluation == ORIGINAL_MECHANISM else "使用该机制的未校准概率；实际掩蔽比例写入运行日志。"))
        self.field_labels["mechanism"].setText(self.t("缺失机制" if benchmark else "训练缺失机制"))
        self.field_labels["mask_rate"].setText(self.t("掩蔽比例" if benchmark else "训练掩蔽比例"))
        self.mask_rate_stack.setToolTip(self.t("评测时构造所有方法共用的测试掩蔽；插补时仅为 JINGWEI 构造训练掩蔽，保留原始观测值作为监督目标。"))

        custom = {evaluation} - {"mcar", ORIGINAL_MECHANISM, JINGWEI_MECHANISM}
        self.fields["mask_level"].setEnabled(bool(custom))
        self.fields["paper_probabilities"].setEnabled(bool(custom))
        self.fields["allow_empty_marginals"].setEnabled(evaluation != ORIGINAL_MECHANISM)

        slope_spin = self.fields["logistic_slope"]
        prev_slope_block = slope_spin.blockSignals(True)
        if evaluation == JINGWEI_MECHANISM:
            slope_spin.setMaximum(-0.0001)
            if slope_spin.value() >= 0:
                slope_spin.setValue(-1.0)
        else:
            slope_spin.setMaximum(0.0)
        slope_spin.blockSignals(prev_slope_block)

        for key, mechanisms in {
            "logistic_slope": {"mnar-logistic", JINGWEI_MECHANISM},
            "low_quantile": {"mnar-jingwei", "mnar-threshold"},
            "high_quantile": {"mnar-jingwei"},
            "low_probability": {"mnar-jingwei", "mnar-threshold"},
            "high_probability": {"mnar-jingwei"},
            "threshold_probability": {"mnar-threshold"},
        }.items():
            self.fields[key].setEnabled(evaluation in mechanisms)

    def protocol_changed(self):
        self.update_mask_controls()
        if hasattr(self, "_environment_base_result"):
            self.apply_environment(self._environment_base_result)
        if self.preview_data is not None:
            try:
                if self.signature() == self.preview_signature:
                    return
            except OSError:
                pass
        self.invalidate_preview()
        from .gui import clean_path_text
        raw = clean_path_text(self.input_path.text())
        if raw and self._is_existing_file(raw):
            self.inspect_data()

    def method_label(self, name):
        return LABELS.get(name, name)

    def update_data_labels(self):
        from .gui import clean_path_text
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
            raw = clean_path_text(self.input_path.text())
            if self._inspecting:
                summary = self.t("正在读取并检查数据…")
            elif getattr(self, "_inspect_error", None):
                summary = self.t("数据检查失败：{error}", error=self._inspect_error)
            elif not raw:
                summary = self.t("尚未载入数据")
            elif not self._is_existing_file(raw):
                summary = self.t("文件不存在，请检查路径或重新选择。")
            else:
                summary = self.t("按回车或点击刷新图标重新检查数据。")
            self.data_summary.setText(summary)
        self.preview_button.setEnabled(self.preview_data is not None)
        self.w2_note.setText(self.t("W2：{count} 个蛋白", count=len(self.feature_selection)) if self.feature_selection else self.t("W2：所有蛋白"))

    def update_environment_labels(self):
        if self._environment_checking:
            label = self.t("正在检查计算环境…")
        elif self._environment_error:
            label = self.t("计算环境检查失败")
        else:
            count = sum(state["available"] for state in self.backend_states.values())
            label = self.t("{count}/{total} 种方法可用 · {device}", count=count, total=len(METHODS), device=self._gpu_name or "CPU")
        self.env_badge.setText(label)
        self.environment_text.setText(label)
        self.environment_table.setHorizontalHeaderLabels([self.t(source) for source in ("方法名称", "后端", "状态")])
        from PySide6.QtWidgets import QTableWidgetItem
        self.environment_table.setRowCount(len(self.backend_states))
        prev_tree = self.method_tree.blockSignals(True)
        for i, (name, state) in enumerate(self.backend_states.items()):
            reason = self.t("可用") if state["available"] else str(state["reason"])
            self.method_checks[name].setToolTip(name + " · " + reason)
            for j, display in enumerate((name, state["backend"], reason)):
                self.environment_table.setItem(i, j, QTableWidgetItem(str(display)))
        self.method_tree.blockSignals(prev_tree)

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
            box.item.setText(0, self.t(self.method_label(name)))
        self.method_groups["python"].setText(0, self.t("传统方法"))
        for i, source in enumerate(("结果", "图表", "日志")):
            self.result_tabs.setTabText(i, self.t(source))
        for i, source in enumerate(("导入", "训练", "掩蔽与 W2", "运行环境")):
            self.advanced_tabs.setTabText(i, self.t(source))
        self.update_data_labels()
        self.update_environment_labels()
        self.update_method_count()
        self.update_mask_controls()
        self._last_result_digest = None
        self.refresh_jobs()
        if selected_row >= 0:
            self.result_table.selectRow(selected_row)
            self.show_method_log()
        if not self.selected_job:
            self.plot.empty(self.t("暂无可比较数据"))
            self.current_task_label.setText(self.t("尚无任务"))
            self.current_task_label.setObjectName("muted")
        from .gui_language import remember_language
        remember_language(self.store.root, language)
