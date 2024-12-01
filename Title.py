from PyQt6.QtWidgets import (
    QPushButton, QWidgetAction, QMenu,
    QHBoxLayout, QWidget
)
from PyQt6.QtCore import Qt
from Button import ButtonBase
from Data import Data
# Title Bar

class TitleWidgetAction(QWidgetAction):
    def __init__(self, parent, button):
        super().__init__(parent)
        self.setDefaultWidget(button)

class TitleMenu(QMenu):
    def __init__(self, buttons):
        super().__init__()
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        for button in buttons:
            self.addAction(TitleWidgetAction(self, button))

class TitleLayout(QHBoxLayout):
    def __init__(self, window):
        super().__init__()
        self.window = window
        self.setContentsMargins(0, 0, 0, 0)
        self.setSpacing(0)
        self.addWidget(ButtonBase("+ Add", callback = self.add_window, font_size = 15))
        self.addWidget(ButtonBase("EN", callback=self.language_callback, font_size = 15))
        self.addWidget(ButtonBase("Fon", callback=Data.app.change_fon, font_size = 15))
        self.addWidget(ButtonBase(
            "View",
            menu=TitleMenu([
                ButtonBase("Global History", callback=self.global_histori_callback, font_size = 15),
                ButtonBase("Local History", menu=TitleMenu([
                    ButtonBase("Basic", callback=self.local_histori_basic_callback, font_size = 15),
                    ButtonBase("Integral", callback=self.local_histori_integral_callback, font_size = 15),
                    ButtonBase("Tab 3", font_size = 15),
                    ButtonBase("Tab 4", font_size = 15)
                ]), font_size = 15)
            ]), font_size = 15
        ))

    def add_window(self):
        Data.count_window += 1
        Data.app.add_window()
    def language_callback(self):
        print("Language button clicked")

    def global_histori_callback(self):
        self.window.global_histori.setVisible(not self.window.global_histori.isVisible())

    def local_histori_basic_callback(self):
        self.window.local_histori[0].setVisible(not self.window.local_histori[0].isVisible())
    def local_histori_integral_callback(self):
        self.window.local_histori[1].setVisible(not self.window.local_histori[1].isVisible())

class TitleBar(QWidget):
    def __init__(self, Window):
        super().__init__()
        self.setFixedHeight(30)
        self.setLayout(TitleLayout(Window))

