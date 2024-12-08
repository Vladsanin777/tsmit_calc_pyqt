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


class Application(QApplication):
    def __init__(self):
        super().__init__([])
        self.setStyleSheet("""
            QMenu {
                background: transparent;
                border: 1px solid white;
            }
            #histori {
                background-color: transparent;
            }
            QTabWidget::pane {
                border: none;
                background: transparent;
            }
            
            QTabBar QToolButton {
                border: none;
                background: rgba(0, 0, 0, 0.3);
                color: white; 
            }
            QTabWidget {
                background: transparent;
                border: none;
                margin: 0px;
            }
            QTabBar::tab {
                background: transparent;
                border: none;
                padding: 0px auto;
                margin: 0px;
                color: trnsparent;
            }
            QTabBar {
                background: transparent;
                border: none;
                margin: 0px;
            }
            QLineEdit {
                background: transparent;
                border: 1px solid white;
                text-align: center;
                color: transparent;
            }
        """)
        
    def change_fon(self):
        """self.setPalette(PaletteWindow())"""
    def add_window(self):
        print(Data.count_window, 1003)
        setattr(self, "window_" + str(Data.count_window), Window())
        Data.count_window += 1

Data.app = Application()
Data.app.add_window()
Data.app.exec()

