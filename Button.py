from PyQt6.QtWidgets import (
    QPushButton, QApplication, QSizePolicy
)
from functools import partial
from PyQt6.QtCore import Qt, QMimeData, QRectF
from PyQt6.QtGui import QDrag
from UI import StyleButton
class ButtonBase(QPushButton):
    def __init__(self, label, *, callback=None, menu=None, css_name = "title-menu-button", window, font_size = 20, min_width = 64, width = 0, is_addition_callback = True):
        self.window = window
        super().__init__(label)
        self.setContentsMargins(0, 0, 0, 0)
        if callback:
            self.clicked.connect(partial(callback, self, window) if is_addition_callback else callback)
        if menu:
            self.setMenu(menu)
        self.setObjectName(css_name)
        self.setFixedHeight(35)
        if width:
            self.setFixedWidth(width)
        else:
            self.setMinimumWidth(min_width)
        font = self.font()
        font.setPointSize(font_size)
        self.setFont(font)
    def paintEvent(self, event):
        StyleButton(self, self.window)

class ButtonDrag(ButtonBase):
    def __init__(self, label: str, *, callback = None, menu = None, css_name = "keybord", window, font_size = 20, min_width = 64, width = 0):
        super().__init__(label, callback = callback, menu = menu, css_name = css_name, window = window, font_size = font_size, min_width = min_width, width = width)

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
    def __init__(self, label, *, callback = None, menu = None, css_name = "keybord", window, font_size = 20, min_width = 64, width = 0):
        super().__init__(label, callback = callback, menu = menu, css_name = css_name, window = window, font_size = font_size, min_width = min_width, width = width)
        self.setAcceptDrops(True)

    def dragEnterEvent(self, event):
        if event.mimeData().hasText():
            event.acceptProposedAction()

    def dropEvent(self, event):
        self.setText(event.mimeData().text())
        event.acceptProposedAction()
