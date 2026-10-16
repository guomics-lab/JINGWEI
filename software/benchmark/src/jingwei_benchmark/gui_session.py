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
        if not (self.store.root / "session.json").exists():
            history = self.store.history()
            if history:
                last = history[0]
                source = next(Path(last["folder"]).glob("input.*"), None)
                fields = dict(last.get("config", {}))
                if "mice_backend" not in fields and any(name in ("mice", "mice-pmm") for name in fields.get("methods", [])):
                    fields["mice_backend"] = "r"
                if isinstance(fields.get("hidden_dims"), list):
                    fields["hidden_dims"] = ", ".join(map(str, fields["hidden_dims"]))
                if fields.get("gpu_id") is None:
                    fields["gpu_id"] = -1
                data = {"schema": 1, "fields": fields, "methods": fields.get("methods", []),
                        "parameters": last.get("parameters", {}), "job_name": last.get("name", ""),
                        "selected_job": last["id"], "input": str(source) if source else "",
                        "w2_features": fields.get("w2_features")}
        if not isinstance(data, dict) or data.get("schema") != 1:
            return False
        fields = data.get("fields", {})
        if not isinstance(fields, dict):
            return False
        for key, widget in self.fields.items():
            setting = fields.get(key)
            if isinstance(widget, QComboBox):
                if key == "layer" and isinstance(setting, str) and widget.findData(setting) < 0:
                    widget.addItem(setting, setting)
                if widget.findData(setting) >= 0:
                    set_value(widget, setting)
            elif isinstance(widget, QCheckBox) and isinstance(setting, bool):
                set_value(widget, setting)
            elif isinstance(widget, (QSpinBox, QDoubleSpinBox)) and type(setting) in (int, float) and math.isfinite(setting):
                if isinstance(widget, QDoubleSpinBox) or isinstance(setting, int):
                    set_value(widget, setting)
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
        parameters = data.get("parameters", {})
        if isinstance(parameters, dict):
            from .gui_parameters import PARAMETERS
            self.method_parameters = {name: {k: v for k, v in settings.items() if k in PARAMETERS.get(name, {})}
                                      for name, settings in parameters.items() if isinstance(settings, dict) and name in PARAMETERS}
        self.job_name.setText(data.get("job_name", "") if isinstance(data.get("job_name", ""), str) else "")
        selected = data.get("selected_job")
        if isinstance(selected, str) and len(selected) == 32 and all(c in "0123456789abcdef" for c in selected):
            if (self.store.jobs / selected / "request.json").is_file():
                self.selected_job = selected
        for key, restore in (("geometry", self.restoreGeometry), ("splitter", self.main_splitter.restoreState)):
            if isinstance(data.get(key), str):
                restore(QByteArray.fromBase64(data[key].encode("ascii", errors="ignore")))
        metric = self.metric_choice.findData(data.get("metric"))
        if metric >= 0:
            self.metric_choice.setCurrentIndex(metric)
        self.paired.setChecked(data.get("paired", True) is not False)
        tab = data.get("result_tab")
        if isinstance(tab, int) and 0 <= tab < self.result_tabs.count():
            self.result_tabs.setCurrentIndex(tab)
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
        for widget in (*self.fields.values(), self.input_path, self.job_name, self.metric_choice, self.paired):
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
        self.job_combo.currentIndexChanged.connect(self.remember_session)
        self.result_tabs.currentChanged.connect(self.remember_session)
        self.main_splitter.splitterMoved.connect(self.remember_session)

    def save_session(self):
        from .gui import value
        data = {"schema": 1, "fields": {key: value(widget) for key, widget in self.fields.items()},
                "input": self.input_path.text(), "methods": [name for name, box in self.method_checks.items() if box.isChecked()],
                "parameters": self.method_parameters, "w2_features": self._pending_features[1] if self._pending_features else self.feature_selection,
                "job_name": self.job_name.text(), "selected_job": self.selected_job,
                "metric": self.metric_choice.currentData(), "paired": self.paired.isChecked(),
                "result_tab": self.result_tabs.currentIndex(),
                "geometry": bytes(self.saveGeometry().toBase64()).decode("ascii"),
                "splitter": bytes(self.main_splitter.saveState().toBase64()).decode("ascii")}
        try:
            write_json(self.store.root / "session.json", data)
        except (OSError, ValueError) as exc:
            self.statusBar().showMessage(str(exc), 5000)
