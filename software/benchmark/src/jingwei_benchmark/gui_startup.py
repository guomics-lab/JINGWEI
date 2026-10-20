"""Complete backend initialization before revealing the desktop workbench."""

from PySide6.QtCore import QObject, QTimer, Qt, QEvent
from PySide6.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QProgressBar, QPushButton


class StartupGate(QObject):
    def __init__(self, window, *, hidden=False):
        super().__init__(window)
        self.window = window
        self.revealed = False
        self.ready_before_show = False
        self.hidden = hidden
        self._closing = False
        self.panel = QWidget(None, Qt.SplashScreen)
        self.panel.setWindowTitle("jingwei")
        self.panel.setWindowIcon(window.windowIcon())
        self.panel.installEventFilter(self)
        if hidden:
            self.panel.setAttribute(Qt.WA_DontShowOnScreen, True)
            window.setAttribute(Qt.WA_DontShowOnScreen, True)
        layout = QVBoxLayout(self.panel)
        layout.setContentsMargins(24, 20, 24, 20)
        heading = QHBoxLayout()
        icon = QLabel()
        icon.setPixmap(window.windowIcon().pixmap(40, 40))
        heading.addWidget(icon)
        title = QLabel("jingwei")
        title.setStyleSheet("font-size:20px;font-weight:600")
        heading.addWidget(title, 1)
        layout.addLayout(heading)
        self.message = QLabel(window.t("正在检查计算环境…"))
        self.message.setWordWrap(True)
        layout.addWidget(self.message)
        self.progress = QProgressBar()
        self.progress.setRange(0, 0)
        layout.addWidget(self.progress)
        buttons = QHBoxLayout()
        buttons.addStretch()
        self.retry_button = QPushButton(window.t("重新检查"))
        self.retry_button.clicked.connect(self.retry)
        self.retry_button.hide()
        buttons.addWidget(self.retry_button)
        cancel = QPushButton(window.t("取消"))
        cancel.clicked.connect(self.cancel)
        buttons.addWidget(cancel)
        layout.addLayout(buttons)
        self.panel.resize(380, 165)
        self.timer = QTimer(self)
        self.timer.setInterval(40)
        self.timer.timeout.connect(self.poll)

    def start(self):
        if not self.window._environment_checking and self.window.backend_states:
            self.poll()
            return
        self.panel.show()
        self.timer.start()

    def poll(self):
        if self.window._closed:
            self.timer.stop()
            self.panel.close()
            return
        if self.window._environment_checking:
            return
        if self.window._environment_error:
            self.message.setText(self.window.t("计算环境检查失败") + "\n" + self.window._environment_error[-500:])
            self.progress.hide()
            self.retry_button.show()
            return
        if not self.window.backend_states:
            return
        self.timer.stop()
        self.ready_before_show = not self.window._environment_checking
        self.window._startup_pending = False
        self.revealed = True
        # Keep a visible window throughout the handover so Qt does not quit
        # when the splash closes. No workbench is shown during initialization.
        self.window.show()
        self.panel.close()

    def retry(self):
        self.retry_button.hide()
        self.message.setText(self.window.t("正在检查计算环境…"))
        self.progress.show()
        self.window.refresh_environment(force=True)

    def cancel(self):
        if self._closing:
            return
        self._closing = True
        self.timer.stop()
        self.window.close()
        self.panel.close()
        QApplication.instance().quit()

    def eventFilter(self, watched, event):
        if watched is self.panel and event.type() == QEvent.Close and not self.revealed and not self._closing:
            self.cancel()
            return True
        return super().eventFilter(watched, event)
