from PyQt6.QtWidgets import (
    QWidget, QTabWidget, QGridLayout, QTabBar
)
from PyQt6.QtGui import QDrag, QPainter, QLinearGradient, QFont, QBrush, QTextOption, QColor, QFontMetrics, QPainterPath, QPen
from PyQt6.QtCore import Qt
from Grid import (
    GridCalculateKeybord, GridBasicCalc, GridIntegralCalc
)

class CustomLinearGradient(QLinearGradient):
    def __init__(self):
        super().__init__(0, 0, 400, 0)
        self.setColorAt(0, Qt.GlobalColor.red)
        self.setColorAt(1, Qt.GlobalColor.blue)


class CustomTabBar(QTabBar):
    def __init__(self):
        super().__init__()
        # Задаем уникальные градиенты для каждой вкладки
        self.gradients = [
            CustomLinearGradient(),  # Градиент для первой вкладки
            CustomLinearGradient(),  # Градиент для второй вкладки
            CustomLinearGradient(),  # Градиент для третьей вкладки
            CustomLinearGradient()  # Градиент для четвертой вкладки
        ]
        
        font = self.font()
        font.setPointSize(20)
        self.setFont(font)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        for index in range(self.count()):
            rect = self.tabRect(index)
            is_selected = self.currentIndex() == index

            # Установка шрифта
            font = self.font()
            if is_selected:
                font.setBold(True)  # Жирный шрифт для выделенной вкладки
            painter.setFont(font)

            # Получение текста вкладки
            text = self.tabText(index)
            metrics = QFontMetrics(font)
            text_width = metrics.horizontalAdvance(text)
            text_height = metrics.height()

            # Центрирование текста внутри вкладки
            x = rect.x() + (rect.width() - text_width) / 2
            y = rect.y() + (rect.height() + text_height) / 2 - metrics.descent()

            # Создаём путь текста
            path = QPainterPath()
            path.addText(x, y, font, text)

            # Рисуем белую обводку текста
            pen = QPen(QColor("white"))
            pen.setWidth(2)
            painter.setPen(pen)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawPath(path)

            # Установка градиента
            gradient = self.gradients[index % len(self.gradients)]

            # Рисуем текст с индивидуальным градиентом
            painter.setBrush(QBrush(gradient))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawPath(path)

        painter.end()
class TabQWidget(QWidget):
    def __init__(self, tab):
        super().__init__()
        self.setLayout(tab)

class TabWidgetKeybord(QTabWidget):
    def __init__(self, window):
        super().__init__()
        self.tabBar().setExpanding(True)
        self.setTabBar(CustomTabBar())  # Устанавливаем кастомный TabBar
        self.addTab(TabQWidget(GridCalculateKeybord([["1", "2", "3", "4", "5"], ["6", "7", "8", "9", "0"]], window)), "digits 10")
        self.addTab(TabQWidget(GridCalculateKeybord([["A", "B", "C"], ["D", "E", "F"]], window)), "digits 16")
        self.addTab(TabQWidget(GridCalculateKeybord([["+", "-", ":", "*", "^"], ["!", "sqrt", "ln", "log", "lg"]], window)), "operators")
        self.addTab(TabQWidget(GridCalculateKeybord([["_E", "_PI"]], window)), "consts")
        self.addTab(TabQWidget(GridCalculateKeybord([["round", "mod", "0x"], ["0b", "0t", ","]], window)), "other")




#Main TabWidget

class MainTabWidget(QTabWidget):
    def __init__(self, window):
        super().__init__()
        self.tabBar().setExpanding(True)
        self.setTabBar(CustomTabBar())  # Устанавливаем кастомный TabBar
        self.addTab(TabQWidget(GridBasicCalc(window)), "Basic")
        self.addTab(TabQWidget(GridIntegralCalc(window)), "Integral")
        self.addTab(TabQWidget(QGridLayout()), "Tab 3")
        self.addTab(TabQWidget(QGridLayout()), "Tab 4")
    
    
