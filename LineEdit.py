from PyQt6.QtWidgets import QLineEdit
from PyQt6.QtGui import QDrag, QPainter, QLinearGradient, QFont, QBrush, QTextOption, QColor, QFontMetrics, QPainterPath, QPen
from PyQt6.QtCore import pyqtSignal, Qt
from PyQt6.QtGui import QFocusEvent
from LogicButton import LogicCalculate
class LineEdit(QLineEdit):
    cursorEntered = pyqtSignal()
    def __init__(self, window):
        super().__init__()
        self.window = window
        self.setSizePolicy(self.sizePolicy().Policy.Expanding, self.sizePolicy().Policy.Expanding)
        self.setObjectName("keybord")
        self.textChanged.connect(self.on_entry_changed)
    def focusInEvent(self, event: QFocusEvent):
            if event.reason():
                self.cursorEntered.emit()  # Эмитируем сигнал, когда фокус установлен
            super().focusInEvent(event)  # Вызываем стандартное поведение

    def on_entry_changed(self, text_line_edit):
        logic_calc = LogicCalculate(text_line_edit, self.window)
        if text_line_edit: # если убрать это условие то будет срабатывать button other() с самого начала и будет ошибка
            if "_ALL" in text_line_edit:
                logic_calc.button__ALL()
            elif "_DO" in text_line_edit:
                logic_calc.button__DO()
            elif "_POST" in text_line_edit:
                logic_calc.button__POST()
            elif "_O" in text_line_edit:
                logic_calc.button__O()
            elif "=" in text_line_edit:
                logic_calc.button_result()
            elif "_RES" in text_line_edit:
                logic_calc.button_result()
            else:
                logic_calc.button_other()
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
