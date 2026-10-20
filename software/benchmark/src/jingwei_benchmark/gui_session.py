"""Remember the desktop form independently of task records and language settings."""

from PySide6.QtCore import QByteArray, QTimer
from PySide6.QtWidgets import QCheckBox, QComboBox, QDoubleSpinBox, QLineEdit, QSpinBox

from .gui_jobs import read_json, write_json
import math


class SessionState:
    def restore_session(self, *, load_input):
        from pathlib import Path
        from .gui import set_value
        data = read_json(self.store.root / "session.json")
        if not isinstance(data, dict) or data.get("schema") != 1:
            return False
        fields = data.get("fields", {})
        if not isinstance(fields, dict):
            return False
        fields = dict(fields)
        # Upgrade the saved form to the corrected mechanism. Historical task
        # records are unchanged, and explicitly copying one retains its rule.
        if data.get("mnar_rule_version") != 2:
            from .protocol import ORIGINAL_MECHANISM, JINGWEI_MECHANISM
            if fields.get("mechanism") == ORIGINAL_MECHANISM:
                fields["mechanism"] = JINGWEI_MECHANISM
        parameters = data.get("parameters", {})
        if data.get("fixed_budget_version") != 1:
            # One-time upgrade of the historical shared defaults to the new
            # fixed-budget policy: a saved form without a policy at the old
            # 50/5000 epochs, including old explicit default-policy records.
            # Custom budgets and explicit non-default policies keep their values,
            # and task records on disk are never rewritten.
            from .convergence import DEFAULT_MAX_EPOCHS
            without_policy = "training_policy" not in fields
            if fields.get("max_epochs") in (50, 5000) and fields.get("training_policy") in (None,'fixed','convergence'):
                fields["training_policy"] = "fixed"
                fields["max_epochs"] = DEFAULT_MAX_EPOCHS
            if isinstance(parameters, dict):
                for settings in parameters.values():
                    if isinstance(settings, dict) and settings.get("max_epochs") in (50,5000):
                        settings["max_epochs"] = DEFAULT_MAX_EPOCHS
        # Historical protocol, repeat and training-mask controls are no longer
        # part of the desktop form. Unknown saved fields are ignored below.
        for key, widget in self.fields.items():
            setting = fields.get(key)
            if isinstance(widget, QComboBox):
                if key == "layer" and isinstance(setting, str) and widget.findData(setting) < 0:
                    widget.addItem(setting, setting)
                if widget.findData(setting) >= 0:
                    set_value(widget, setting)
            elif isinstance(widget, QCheckBox) and isinstance(setting, bool):
                set_value(widget, setting)
            elif isinstance(widget, (QSpinBox, QDoubleSpinBox)) and type(setting) in (int, float):
                if type(setting) is float and not math.isfinite(setting):
                    continue
                if isinstance(widget, QDoubleSpinBox):
                    try:
                        setting = float(setting)
                    except OverflowError:
                        continue
                    # FractionControl accepts fractions while its native Qt
                    # bounds are percentages, so let its setter handle bounds.
                    set_value(widget, setting)
                elif type(setting) is int:
                    # Clamp before Qt converts Python integers to its C++ range.
                    set_value(widget, max(widget.minimum(), min(widget.maximum(), setting)))
            elif isinstance(widget, QLineEdit) and isinstance(setting, str):
                set_value(widget, setting)
        self.mode_changed()
        if "output_scale" in fields:
            set_value(self.fields["output_scale"], fields["output_scale"])
        self.device_changed()
        methods = data.get("methods")
        if isinstance(methods, list) and all(isinstance(name, str) for name in methods):
            for name, box in self.method_checks.items():
                box.setChecked(name in methods)
        if isinstance(parameters, dict):
            from .gui_parameters import PARAMETERS, valid_overrides
            self.method_parameters = {name: valid_overrides(name, settings)
                                      for name, settings in parameters.items() if isinstance(settings, dict) and name in PARAMETERS}
        self.job_name.setText(data.get("job_name", "") if isinstance(data.get("job_name", ""), str) else "")
        if isinstance(data.get("output_root"), str):
            self.output_root.setText(data["output_root"])
        for key, restore in (("geometry", self.restoreGeometry), ("splitter", self.main_splitter.restoreState)):
            if isinstance(data.get(key), str):
                restore(QByteArray.fromBase64(data[key].encode("ascii", errors="ignore")))
        metric = self.metric_choice.findData(data.get("metric"))
        if metric >= 0:
            self.metric_choice.setCurrentIndex(metric)
        self.paired.setChecked(data.get("paired", True) is not False)
        source = data.get("input")
        if load_input and isinstance(source, str) and source:
            if Path(source).is_file():
                self.set_input(source, fields.get("layer"))
                features = data.get("w2_features")
                if isinstance(features, list) and all(isinstance(name, str) for name in features):
                    self._pending_features = self.signature(), features
            else:
                self.input_path.setText(source)
                self.data_summary.setText(self.t("上次的数据文件已移动或删除，请重新选择。"))
            return True
        return False

    def remember_session(self, *args):
        if not self._closed:
            self._session_timer.start()

    def connect_session(self):
        self._session_timer = QTimer(self)
        self._session_timer.setSingleShot(True)
        self._session_timer.setInterval(500)
        self._session_timer.timeout.connect(self.save_session)
        for widget in (*self.fields.values(), self.input_path, self.output_root, self.job_name, self.metric_choice, self.paired):
            if isinstance(widget, QComboBox):
                signal = widget.currentIndexChanged
            elif isinstance(widget, QCheckBox):
                signal = widget.toggled
            elif isinstance(widget, (QSpinBox, QDoubleSpinBox)):
                signal = widget.valueChanged
            else:
                signal = widget.textChanged
            signal.connect(self.remember_session)
        self.method_tree.itemChanged.connect(self.remember_session)
        self.result_tabs.currentChanged.connect(self.remember_session)
        self.main_splitter.splitterMoved.connect(self.remember_session)

    def save_session(self):
        from .gui import value
        data = {"schema": 1, "mnar_rule_version": 2, "fixed_budget_version": 1, "dcae_architecture_version": "global-ae-v1",
                "fields": {key: value(widget) for key, widget in self.fields.items()},
                "input": self.input_path.text(), "methods": [name for name, box in self.method_checks.items() if box.isChecked()],
                "parameters": self.method_parameters, "w2_features": self._pending_features[1] if self._pending_features else self.feature_selection,
                "job_name": self.job_name.text(), "output_root": self.output_root.text(),
                "metric": self.metric_choice.currentData(), "paired": self.paired.isChecked(),
                "result_tab": self.result_tabs.currentIndex(),
                "geometry": bytes(self.saveGeometry().toBase64()).decode("ascii"),
                "splitter": bytes(self.main_splitter.saveState().toBase64()).decode("ascii")}
        try:
            write_json(self.store.root / "session.json", data)
        except (OSError, ValueError) as exc:
            self.statusBar().showMessage(str(exc), 5000)
