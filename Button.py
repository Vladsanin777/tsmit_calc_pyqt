from PyQt6.QtWidgets import (
    QPushButton, QApplication
)
from functools import partial
from PyQt6.QtCore import Qt, QMimeData, QRectF
from PyQt6.QtGui import QDrag, QPainter, QLinearGradient, QFont, QBrush, QTextOption, QColor, QFontMetrics, QPainterPath, QPen
class ButtonBase(QPushButton):
    def __init__(self, label, *, callback=None, menu=None, css_name = "title-menu-button", window = None):
        super().__init__(label)
        self.setSizePolicy(self.sizePolicy().Policy.Expanding, self.sizePolicy().Policy.Expanding)
        if callback:
            self.clicked.connect(partial(callback, self, window) if window else callback)
        if menu:
            self.setMenu(menu)
        self.setObjectName(css_name)
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Задаём фон (например, белый)
        painter.fillRect(self.rect(), QColor("transparent"))

        # Установка шрифта
        painter.setFont(self.font())

        # Подготовка текста
        text = self.text if isinstance(self.text, str) else self.text()
        metrics = QFontMetrics(self.font())
        text_width = metrics.horizontalAdvance(text)
        text_height = metrics.height()

        # Центрирование текста
        x = (self.width() - text_width) / 2
        y = (self.height() + text_height) / 2

        # Создаём путь текста
        path = QPainterPath()
        path.addText(x, y - metrics.descent(), self.font(), text)

        # Рисуем белую обводку
        pen = QPen(QColor("white"))
        pen.setWidth(2)  # Толщина обводки
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)  # Убираем заливку для обводки
        painter.drawPath(path)

        # Установка градиента
        gradient = QLinearGradient(0, 0, self.width(), 0)
        gradient.setColorAt(0, Qt.GlobalColor.red)
        gradient.setColorAt(1, Qt.GlobalColor.blue)

        # Рисуем текст с градиентом
        painter.setBrush(QBrush(gradient))
        painter.setPen(Qt.PenStyle.NoPen)  # Убираем обводку для текста
        painter.drawPath(path)

        painter.end()

    """
    def paintEvent(self, event):
        metrics = QFontMetrics(self.font())
        text_width = metrics.horizontalAdvance(self.text())
        text_height = metrics.height()

        # Центрирование текста
        x = (self.width() - text_width) / 2
        y = (self.height() + text_height) / 2

        # Создаём путь текста
        path = QPainterPath()
        path.addText(x, y - metrics.descent(), self.font(), self.text())

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setFont(self.font())

        # Установка градиента
        gradient = QLinearGradient(0, 0, self.width(), 0)
        gradient.setColorAt(0, Qt.GlobalColor.red)
        gradient.setColorAt(1, Qt.GlobalColor.blue)

        painter.setBrush(QBrush(gradient))
        
        painter.setPen(Qt.PenStyle.NoPen)

        # Рисуем текст с градиентом
        painter.drawPath(path)

        painter.end()
    """

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
