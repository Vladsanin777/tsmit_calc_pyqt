from PyQt6.QtWidgets import (
    QPushButton, QApplication
)

from PyQt6.QtCore import Qt, QMimeData
from PyQt6.QtGui import QDrag
class ButtonBase(QPushButton):
    def __init__(self, label, *, callback=None, menu=None, css_name = "title-menu-button"):
        super().__init__(label)
        self.setSizePolicy(self.sizePolicy().Policy.Expanding, self.sizePolicy().Policy.Expanding)
        if callback:
            self.clicked.connect(callback)
        if menu:
            self.setMenu(menu)
        self.setObjectName(css_name)
class ButtonDrag(ButtonBase):
    def __init__(self, label: str, *, callback = None, menu = None, css_name = "keybord"):
        super().__init__(label, callback = callback, menu = menu, css_name = css_name)

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
    def __init__(self, label, *, callback = None, menu = None, css_name = "keybord"):
        super().__init__(label, callback = callback, menu = menu, css_name = css_name)
        self.setAcceptDrops(True)

    def dragEnterEvent(self, event):
        if event.mimeData().hasText():
            event.acceptProposedAction()

    def dropEvent(self, event):
        self.setText(event.mimeData().text())
        event.acceptProposedAction()
