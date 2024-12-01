from PyQt6.QtWidgets import (
    QPushButton, QApplication, QSizePolicy
)
from functools import partial
from PyQt6.QtCore import Qt, QMimeData, QRectF
from PyQt6.QtGui import QDrag
from UI import StyleButton
class ButtonBase(QPushButton):
    def __init__(self, label, *, callback=None, menu=None, css_name = "title-menu-button", window = None, font_size = 20, min_width = 63):
        super().__init__(label)
        if callback:
            self.clicked.connect(partial(callback, self, window) if window else callback)
        if menu:
            self.setMenu(menu)
        self.setObjectName(css_name)
        font = self.font()
        self.setMinimumSize(min_width, 35)
        font.setPointSize(font_size)
        self.setFont(font)
        self.setContentsMargins(0, 0, 0, 0)
    def paintEvent(self, event):
        StyleButton(self)

class ButtonDrag(ButtonBase):
    def __init__(self, label: str, *, callback = None, menu = None, css_name = "keybord", window = None):
        super().__init__(label, callback = callback, menu = menu, css_name = css_name, window = window)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._start_pos = event.pos()  # Сохраняем начальную позицию
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if (event.buttons() & Qt.MouseButton.LeftButton) and \
                (event.pos() - self._start_pos).manhattanLength() > QApplication.startDragDistance():
            drag = QDrag(self)
            mime_data = QMimeData()
            mime_data.setText(self.text())
            drag.setMimeData(mime_data)
            drag.exec(Qt.DropAction.MoveAction)
        else:
            super().mouseMoveEvent(event)


class ButtonDragAndDrop(ButtonDrag):
    def __init__(self, label, *, callback = None, menu = None, css_name = "keybord", window = None):
        super().__init__(label, callback = callback, menu = menu, css_name = css_name, window = window)
        self.setAcceptDrops(True)

    def dragEnterEvent(self, event):
        if event.mimeData().hasText():
            event.acceptProposedAction()

    def dropEvent(self, event):
        self.setText(event.mimeData().text())
        event.acceptProposedAction()
