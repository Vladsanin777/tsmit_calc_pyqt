from PyQt6.QtWidgets import (
    QLabel, QHBoxLayout, QVBoxLayout
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



class BaseBoxHistoriElement(QHBoxLayout):
    def __init__(
            self, expression: str, window,
            *, result: str
    ):
        super().__init__()
        self.setSpacing(0)
        self.setContentsMargins(0, 0, 0, 0)
        self.addWidget(LabelHistori(expression, "keybord", window))
        self.addWidget(LabelHistori("=", "keybord", window, custom_callback = expression + "=" + result))
        self.addWidget(LabelHistori(result, "keybord", window))
class BasicBoxHistoriElement(QVBoxLayout):
    def __init__(
            self, expression: str, window,
            *, result: str, name_operation: str
    ):
        super().__init__()
        self.setSpacing(0)
        self.setContentsMargins(0, 0, 0, 0)
        self.addWidget(LabelHistori(name_operation, "keybord", window))
        self.addLayout(
            BaseBoxHistoriElement(
                expression, window,
                result = result,
            )
        )

class SubCustomBoxHistoriElement(QHBoxLayout):
    def __init__(
        self, window, *,
        label_1: str, text_1: str,
        label_2: str, text_2: str
    ):
        super().__init__()
        self.setSpacing(0)
        self.setContentsMargins(0, 0, 0, 0)
        print(
            label_1, text_1,
            label_2, text_2
        )
        self.addWidget(LabelHistori(label_1, "keybord", window))
        self.addWidget(LabelHistori("=", "keybord", window, custom_callback = label_1 + "=" + text_1))
        self.addWidget(LabelHistori(text_1, "keybord", window))
        
        self.addWidget(LabelHistori(label_2, "keybord", window))
        self.addWidget(LabelHistori("=", "keybord", window, custom_callback = label_2 + "=" + text_2))
        self.addWidget(LabelHistori(text_2, "keybord", window))


class CustomBoxHistoriElement(QVBoxLayout):
    def __init__(
            self, expression: str, window, *,
            number_tab: str, name_operation,
            label_1: str, text_1: str, 
            label_2: str, text_2: str
    ):
        super().__init__()
        self.setSpacing(0)
        self.setContentsMargins(0, 0, 0, 0)
        self.addWidget(LabelHistori(name_operation, "keybord", window))
        self.addLayout(
            SubCustomBoxHistoriElement(
                window, 
                label_1 = label_1, text_1 = text_1,
                label_2 = label_2, text_2 = text_2
            )
        )
        self.addLayout(
            BaseBoxHistoriElement(
                window.getLineEdit(number_tab, 2).text(),
                window,
                result = window.getResult(number_tab, 2)
            )
        )
