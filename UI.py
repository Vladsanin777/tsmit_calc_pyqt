from PyQt6.QtGui import QDrag, QPainter, QLinearGradient, QFont, QBrush, QTextOption, QColor, QFontMetrics, QPainterPath, QPen
from PyQt6.QtCore import Qt

class CreateGradient(QLinearGradient):
    def __init__(self, width, list_gradient: list[list[int, Qt.GlobalColor]]):
        super().__init__(0, 0, width, 0)
        for colorAt in list_gradient:
            self.setColorAt(colorAt[0], colorAt[1])



class ButtonPen(QPen):
    def __init__(self):
        super().__init__()
        self.setColor(QColor("white"))
        self.setWidth(2)  # Толщина обводки
class StyleButton(QPainter):
    def __init__(self, parent):
        super().__init__(parent)
        self.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Задаём фон (например, белый)
        self.fillRect(parent.rect(), QColor("transparent"))

        # Установка шрифта
        self.setFont(parent.font())

        metrics = QFontMetrics(parent.font())

        # Создаём путь текста
        path = QPainterPath()
        path.addText(
            (parent.width() - metrics.horizontalAdvance(parent.text())) / 2, 
            (parent.height() + metrics.height()) / 2 - metrics.descent(), 
            parent.font(), 
            parent.text()
        )

        # Рисуем белую обводку
        self.setPen(ButtonPen())
        self.setBrush(Qt.BrushStyle.NoBrush)  # Убираем заливку для обводки
        self.drawPath(path)

        # Установка градиента
        gradient = CreateGradient(parent.width(), [[0, Qt.GlobalColor.red], [1, Qt.GlobalColor.blue]])
        # Рисуем текст с градиентом
        self.setBrush(QBrush(gradient))
        self.setPen(Qt.PenStyle.NoPen)  # Убираем обводку для текста
        self.drawPath(path)

        self.end()

