from PyQt6.QtWidgets import (
    QWidget, QTabWidget, QGridLayout, QTabBar
)
from PyQt6.QtGui import QDrag, QPainter, QLinearGradient, QFont, QBrush, QTextOption, QColor, QFontMetrics, QPainterPath, QPen
from PyQt6.QtCore import Qt
from Grid import (
    GridCalculateKeybord, GridBasicCalc, GridIntegralCalc
)
from UI import CreateGradient



class CustomTabBar(QTabBar):
    def __init__(self, tab_widget, window):
        super().__init__()
        self.tab_widget = tab_widget
        self.window = window
        self.gradients = []
        font = self.font()
        font.setPointSize(20)
        self.setFont(font)

    def set_style(self):
        # Задаем уникальные градиенты для каждой вкладки
        self.gradients = [CreateGradient(self.tab_widget, self.window, is_tab=True) for _ in range(4)]
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
            if self.gradients:
                gradient = self.gradients[index % len(self.gradients)]
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
        self.window = window
        super().__init__()
        #self.tabBar().setExpanding(True)
        self.tab_bar = CustomTabBar(self, window)
        self.setTabBar(self.tab_bar)  # Устанавливаем кастомный TabBar
        self.addTab(TabQWidget(GridCalculateKeybord([["1", "2", "3", "4", "5"], ["6", "7", "8", "9", "0"]], window)), "digits 10")
        self.addTab(TabQWidget(GridCalculateKeybord([["A", "B", "C"], ["D", "E", "F"]], window)), "digits 16")
        self.addTab(TabQWidget(GridCalculateKeybord([["+", "-", ":", "*", "^"], ["!", "sqrt", "ln", "log", "lg"]], window)), "operators")
        self.addTab(TabQWidget(GridCalculateKeybord([["_E", "_PI"]], window)), "consts")
        self.addTab(TabQWidget(GridCalculateKeybord([["0x", "0b", "0t"]], window)), "number system")
        self.addTab(TabQWidget(GridCalculateKeybord([["%", "mod", ".", "|"]], window)), "other")
    def paintEvent(self, event):
        self.tab_bar.set_style()



#Main TabWidget

class MainTabWidget(QTabWidget):
    def __init__(self, window):
        super().__init__()
        #self.tabBar().setExpanding(True)
        self.tab_bar = CustomTabBar(self, window)
        self.setTabBar(self.tab_bar)  # Устанавливаем кастомный TabBar
        self.addTab(TabQWidget(GridBasicCalc(window)), "Basic")
        self.addTab(TabQWidget(GridIntegralCalc(window)), "Integral")
        self.addTab(TabQWidget(QGridLayout()), "Tab 3")
        self.addTab(TabQWidget(QGridLayout()), "Tab 4")
        self.window = window

    def paintEvent(self, event):
        self.tab_bar.set_style()
    """
    def set_style(self):
        print(3)
        self.setTabBar(CustomTabBar(self, self.window))  # Устанавливаем кастомный TabBar
        self.tabBar().show()
   """ 
