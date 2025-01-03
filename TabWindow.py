from PyQt6.QtWidgets import (
    QWidget, QTabWidget, QGridLayout, 
    QTabBar, QStackedWidget, QVBoxLayout,
    QSizePolicy
)
from PyQt6.QtGui import (
    QDrag, QPainter, QLinearGradient,
    QFont, QBrush, QTextOption, 
    QColor, QFontMetrics, QPainterPath, 
    QPen
)
from PyQt6.QtCore import (
    Qt, QPropertyAnimation, QRect
)
from Grid import (
    GridCalculateKeybord, GridBasicCalc, GridReplacementCalc,
    GridIntegralCalc, GridDerivativeOrIntegrateCalc
)
from UI import CreateGradient

from typing import Self



class CustomTabBar(QTabBar):
    gradient: CreateGradient
    
    def __init__(self, tab_widget, window):
        super().__init__()
        self.tab_widget = tab_widget
        self.window = window
        font = self.font()
        font.setPointSize(20)
        self.setFont(font)

    def set_style(self):
        # Задаем градиенты
        self.gradient = CreateGradient(self.tab_widget, self.window)
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        for index in range(self.count()):
            rect = self.tabRect(index)
            is_selected = self.currentIndex() == index

            # Установка шрифта
            font = self.font()
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

            # Рисуем текст разным цветом для активной вкладки
            pen = QPen()
            pen.setColor(QColor("white"))  # Цвет текста неактивной вкладки
            if is_selected:
                pen.setWidth(4)
            else:
                pen.setWidth(2)
            painter.setPen(pen)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawPath(path)

            # Установка градиента для вкладки
            painter.setBrush(QBrush(self.gradient))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawPath(path)

        painter.end()



class TabQWidget(QWidget):
    def __init__(self: Self, tab) -> Self:
        super().__init__()
        self.setLayout(tab)
class TabWidgetKeyboard(QTabWidget):
    def __init__(self: Self, window):
        self.window = window
        super().__init__()
        self.setFixedHeight(110)
        self.tab_bar = CustomTabBar(self, window)
        self.setTabBar(self.tab_bar)  # Устанавливаем кастомный TabBar
        self.addTab(TabQWidget(GridCalculateKeybord([["1", "2", "3", "4", "5"], ["6", "7", "8", "9", "0"]], window)), "digits 10")
        self.addTab(TabQWidget(GridCalculateKeybord([["A", "B", "C"], ["D", "E", "F"]], window)), "digits 16")
        self.addTab(TabQWidget(GridCalculateKeybord([["+", "-", ":", "*", "^"], ["!", "sqrt", "ln", "log", "lg"]], window)), "operators")
        self.addTab(TabQWidget(GridCalculateKeybord([["_E", "_PI"]], window)), "consts")
        self.addTab(TabQWidget(GridCalculateKeybord([["sin(", "cos(", "tan("], ["sec(", "csc(", "cot("]], window)), "trigonometric functions")
        self.addTab(TabQWidget(GridCalculateKeybord([["sgn(", "abs(", "mod"]], window)), "other functions")
        self.addTab(TabQWidget(GridCalculateKeybord([["0x", "0b", "0t"]], window)), "number system")
        self.addTab(TabQWidget(GridCalculateKeybord([["%", "mod", ".", "|"]], window)), "other")
    def paintEvent(self, event) -> None:
        self.tab_bar.set_style()



#Main TabWidget

class MainTabWidget(QTabWidget):
    def __init__(self: Self, window) -> Self:
        super().__init__()
        #self.tabBar().setExpanding(True)
        self.tab_bar = CustomTabBar(self, window)
        self.setTabBar(self.tab_bar)  # Устанавливаем кастомный TabBar
        self.addTab(TabQWidget(GridBasicCalc(window)), "Basic")
        self.addTab(TabQWidget(GridIntegralCalc(window)), "Integral")
        self.addTab(TabQWidget(GridDerivativeOrIntegrateCalc(window, 2)), "Derivative")
        self.addTab(TabQWidget(GridDerivativeOrIntegrateCalc(window, 3)), "Integrate")
        self.addTab(TabQWidget(GridReplacementCalc(window)), "Replacement")
        self.window = window

    def paintEvent(self: Self, event):
        self.tab_bar.set_style()
