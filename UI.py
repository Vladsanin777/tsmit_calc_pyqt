from PyQt6.QtGui import QDrag, QPainter, QLinearGradient, QFont, QBrush, QTextOption, QColor, QFontMetrics, QPainterPath, QPen
from PyQt6.QtCore import Qt, QPoint

class CreateGradient(QLinearGradient):
    def __init__(self, widget, window, *, list_gradient: list[list[int, Qt.GlobalColor]] = [[0, Qt.GlobalColor.red], [1, Qt.GlobalColor.blue]], is_tab = False):

        position_widget = widget.mapToGlobal(QPoint(0, 0))
        position_window = window.mapToGlobal(QPoint(0, 0))
        y = position_widget.y() - position_window.y()
        x = position_widget.x() - position_window.x()
        width = window.width()
        height = window.height()
        if is_tab:
            print(x, y, x + width, y - height)
            super().__init__(x, y, x + width, y - height)
        else:
            super().__init__(x, y, x - width, y - height)
        for colorAt in list_gradient:
            self.setColorAt(colorAt[0], colorAt[1])



class ButtonPen(QPen):
    def __init__(self):
        super().__init__()
        self.setColor(QColor("white"))
        self.setWidth(2)  # Толщина обводки
class StyleButton(QPainter):
    def __init__(self, parent, window):
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
            (parent.width() - metrics.horizontalAdvance(parent.text())) / 3, 
            (parent.height() + metrics.height()) / 2 - metrics.descent(), 
            parent.font(), 
            parent.text()
        )

        # Рисуем белую обводку
        self.setPen(ButtonPen())
        self.setBrush(Qt.BrushStyle.NoBrush)  # Убираем заливку для обводки
        self.drawPath(path)



        # Установка градиента
        gradient = CreateGradient(
            parent,
            window
        )
        # Рисуем текст с градиентом
        self.setBrush(QBrush(gradient))
        self.setPen(Qt.PenStyle.NoPen)  # Убираем обводку для текста
        self.drawPath(path)

        self.end()

