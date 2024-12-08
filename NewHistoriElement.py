from PyQt6.QtWidgets import (
    QLabel, QHBoxLayout
)
from PyQt6.QtGui import QDrag, QPainter, QLinearGradient, QFont, QBrush, QTextOption, QColor, QFontMetrics, QPainterPath, QPen
from PyQt6.QtCore import Qt, QMimeData
from PyQt6.QtGui import QDrag
from UI import StyleButton
class LabelHistori(QLabel):
    callback: str
    def __init__(self, label: str, css_name: str, window, *, custom_callback: str = None):
        super().__init__(label)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setSizePolicy(self.sizePolicy().Policy.Expanding, self.sizePolicy().Policy.Expanding)
        self.setContentsMargins(0, 0, 0, 0)
        self.setObjectName(css_name)
        font = self.font()
        font.setPointSize(20)
        self.setFont(font)
        if custom_callback:
            self.callback = custom_callback
        else:
            self.callback = label
        self.window = window

    def mousePressEvent(self, event):
        drag = QDrag(self)
        mime_data = QMimeData()
        mime_data.setText(self.callback)
        drag.setMimeData(mime_data)
        drag.exec(Qt.DropAction.MoveAction)
    def paintEvent(self, event):
        StyleButton(self, self.window)



class BoxHistoriElement(QHBoxLayout):
    def __init__(self, expression: str, window):
        super().__init__()
        result: str = window.activateResult()
        self.setSpacing(0)
        self.setContentsMargins(0, 0, 0, 0)
        self.addWidget(LabelHistori(expression, "keybord", window))
        self.addWidget(LabelHistori("=", "keybord", window, custom_callback = expression + "=" + result))
        self.addWidget(LabelHistori(result, "keybord", window))
