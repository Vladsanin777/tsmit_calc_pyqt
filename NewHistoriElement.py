from PyQt6.QtWidgets import (
    QLabel, QHBoxLayout
)
from PyQt6.QtGui import QDrag, QPainter, QLinearGradient, QFont, QBrush, QTextOption, QColor, QFontMetrics, QPainterPath, QPen
from PyQt6.QtCore import Qt, QMimeData
from PyQt6.QtGui import QDrag
class LabelHistori(QLabel):
    callback: str
    def __init__(self, label: str, css_name: str, *, custom_callback: str = None):
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

    def mousePressEvent(self, event):
        drag = QDrag(self)
        mime_data = QMimeData()
        mime_data.setText(self.callback)
        drag.setMimeData(mime_data)
        drag.exec(Qt.DropAction.MoveAction)
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



class BoxHistoriElement(QHBoxLayout):
    def __init__(self, expression: str, result: str):
        super().__init__()
        self.setSpacing(0)
        self.setContentsMargins(0, 0, 0, 0)
        self.addWidget(LabelHistori(expression, "keybord"))
        self.addWidget(LabelHistori("=", "keybord", custom_callback = expression + "=" + result))
        self.addWidget(LabelHistori(result, "keybord"))
