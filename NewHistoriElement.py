from PyQt6.QtWidgets import (
    QLabel, QHBoxLayout
)
from PyQt6.QtCore import Qt
class LabelHistori(QLabel):
    callback: str
    def __init__(self, label: str, css_name: str, *, custom_callback: str = None):
        super().__init__(label)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setSizePolicy(self.sizePolicy().Policy.Expanding, self.sizePolicy().Policy.Expanding)
        self.setContentsMargins(0, 0, 0, 0)
        self.setObjectName(css_name)
        if custom_callback:
            self.callback = custom_callback
        else:
            self.callback = label

    def mousePressEvent(self, event):
        drag = QDrag(self)
        mime_data = QMimeData()
        mime_data.setText(self.callback)
        drag.setMimeData(mime_data)
        drag.exec(Qt.DropAction.MoveAction)

class BoxHistoriElement(QHBoxLayout):
    def __init__(self, expression: str, result: str):
        super().__init__()
        self.setSpacing(0)
        self.setContentsMargins(0, 0, 0, 0)
        self.addWidget(LabelHistori(expression, "keybord"))
        self.addWidget(LabelHistori("=", "keybord", custom_callback = expression + "=" + result))
        self.addWidget(LabelHistori(result, "keybord"))
