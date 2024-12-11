from PyQt6.QtGui import QDrag, QPainter, QLinearGradient, QFont, QBrush, QTextOption, QColor, QFontMetrics, QPainterPath, QPen
from PyQt6.QtCore import Qt, QPoint

class CreateGradient(QLinearGradient):
    def __init__(self, widget, window, *, list_gradient: list[list[int, Qt.GlobalColor]] = [[0, Qt.GlobalColor.red], [1, Qt.GlobalColor.blue]]):
        width = window.width()
        height = window.height()
        if widget:
            position_widget = widget.mapToGlobal(QPoint(0, 0))
            position_window = window.mapToGlobal(QPoint(0, 0))
            y = position_widget.y() - position_window.y()
            x = position_widget.x() - position_window.x()
            super().__init__(-x, -y, width // 2, height // 2)
        else:
            super().__init__(0, 0, width, height)
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

        text_x: float

        if metrics.horizontalAdvance(parent.text()) > parent.width():
            text_x = 0.0
        else:
            text_x = (parent.width() - metrics.horizontalAdvance(parent.text())) / 2 
        # Создаём путь текста
        path = QPainterPath()
        path.addText(
            text_x,
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

class StyleLineEdit(QPainter):
    def __init__(self, parent, window):
        # Создаём QPainter
        super().__init__(parent)
        self.setRenderHint(QPainter.RenderHint.Antialiasing)


        # Настройка шрифта и метрик
        metrics = QFontMetrics(parent.font())
        text = parent.text()

         # Получаем смещение текста (начало видимой области)
        scroll_offset = parent.cursorRect().x() - metrics.horizontalAdvance(text[:parent.cursorPosition()])

        # Вычисляем начальную позицию текста с учётом прокрутки
        text_x = scroll_offset+5
        text_y = (parent.height() + metrics.ascent() - metrics.descent()) / 2

        # Создаём путь для текста
        path = QPainterPath()
        path.addText(text_x, text_y, self.font(), text)

        # Рисуем обводку текста
        pen = QPen(QColor("white"))
        pen.setWidth(2)
        self.setPen(pen)
        self.setBrush(Qt.BrushStyle.NoBrush)
        self.drawPath(path)

        # Рисуем текст с градиентом
        self.setBrush(QBrush(CreateGradient(parent, window)))
        self.setPen(Qt.PenStyle.NoPen)
        self.drawPath(path)

        # Завершаем рисование
        self.end()
