from PyQt6.QtWidgets import (
    QPushButton, QWidgetAction, QMenu,
    QHBoxLayout, QWidget
)
from PyQt6.QtCore import Qt
from Button import BaseButton
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
    def __init__(self):
        super().__init__()
        self.setContentsMargins(0, 0, 0, 0)
        self.setSpacing(0)
        self.addWidget(BaseButton("EN", callback=self.language_callback))
        self.addWidget(BaseButton("Fon", callback=Data.app.change_fon))
        self.addWidget(BaseButton(
            "View",
            menu=TitleMenu([
                BaseButton("Global History", callback=self.global_histori_callback),
                BaseButton("Local History", menu=TitleMenu([
                    BaseButton("Basic", callback=self.local_histori_basic_callback),
                    BaseButton("Tab 2"),
                    BaseButton("Tab 3"),
                    BaseButton("Tab 4")
                ]))
            ])
        ))

    def language_callback(self):
        print("Language button clicked")

    def global_histori_callback(self):
        global global_hislori
        global_histori.setVisible(not global_histori.isVisible())

    def local_histori_basic_callback(self):
        global local_histori_basic
        local_histori_basic.setVisible(not local_histori_basic.isVisible())

class TitleBar(QWidget):
    def __init__(self):
        super().__init__()
        self.setFixedHeight(30)
        self.setLayout(TitleLayout())

