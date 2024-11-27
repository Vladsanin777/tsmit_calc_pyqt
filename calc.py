from PyQt6.QtWidgets import (
    QApplication, QWidget
)
from PyQt6.QtCore import (
    Qt, QMimeData
)
from PyQt6.QtGui import (
    QColor, QPalette, 
    QLinearGradient, QBrush
)
import random

from Data import Data

from Window import Window

colors_background = ["#99FF18", "#FFF818", "#FFA918", "#FF6618", "#FF2018", "#FF1493", "#FF18C9", "#CB18FF"]


class GradientWindow(QLinearGradient):
    def __init__(self):
        super().__init__(0, 0, 1, 1)
        random_color_1 = random.choice(colors_background)
        random_color_2 = random.choice(colors_background)
        while random_color_1 == random_color_2:
            random_color_2 = random.choice(colors_background)
        self.setCoordinateMode(QLinearGradient.CoordinateMode.ObjectBoundingMode)
        self.setColorAt(0, QColor(random_color_1))
        self.setColorAt(1, QColor(random_color_2))

class PaletteWindow(QPalette):
    def __init__(self):
        super().__init__()
        self.setBrush(QPalette.ColorRole.Window, QBrush(GradientWindow()))

class Application(QApplication):
    def __init__(self):
        super().__init__([])
        self.setStyleSheet("""
            QPushButton#title-button,
            QPushButton#title-menu-button,
            QPushButton#calculate {
                background-color: rgba(0, 0, 0, 0.3);
                color: white;
                border: none;
            }
            QPushButton#calculate {
                font-size: 20px;
            }
            QPushButton#title-menu-button {
                padding: 5px 10px;
            }
            QPushButton#title-button:hover, 
            QPushButton#title-menu-button:hover {
                background-color: rgba(0, 0, 0, 0.6);
            }
            QScrollArea{
                border: none;
                background: transparent;
            }
            QTabBar::tab {
                background: rgba(0, 0, 0, 0.3);
                border: none;
                padding: 5px auto;
                color: rgb(255, 255, 255);
            }
            QTabBar::tab:selected, QTabBar::tab:hover {
                background: transparent;
                color: rgb(0, 0, 0);
            }
            QTabWidget::pane {
                border: none;
                background: transparent;
            }
            QTabBar QToolButton {
                border: none;
                background: rgba(0, 0, 0, 0.3);
                color: rgb(0, 0, 0); 
            }
            #keybord {
                margin: 0px;
                border: none;
                font-size: 30px;
                background: rgba(0, 0, 0, 0.3);
                color: rgb(255, 255, 255);
            }
            #keybord:hover {
                background: transparent;
                color: rgb(0, 0, 0);
            }
            QMenu {
                background-color: rgba(0, 255, 0, 0);  /* Полупрозрачный фон */
                color: white;                            /* Цвет текста */
                border: 1px solid white;                 /* Рамка */
            }
            QWidget#histori {
                background-color: transparent;
            }
            QScrollBar:vertical {
                background: rgba(0, 0, 0, 0.3);
                width: 14px;
                margin: 0px 0px 0px 0px;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                background: none;
                height: 0px;
                border: none;
            }
            QScrollBar::handle:vertical {
                background-color: rgba(0, 0, 0, 0.6);
                border-radius: 7px; /* Делаем ползунок скруглённым */
            }
            QScrollBar::handle:vertical:hover {
                background-color: rgba(255, 255, 255, 0.3);
            }
        """)

        self.change_fon()
    def change_fon(self):
        self.setPalette(PaletteWindow())
    def add_window(self):
        setattr(self, "window_" + str(Data.count_window), Window())

Data.app = Application()
Data.app.add_window()
Data.app.exec()

