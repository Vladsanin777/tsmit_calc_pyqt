from PyQt6.QtWidgets import (
    QWidget, QTabWidget, QGridLayout
)
from Grid import (
    GridCalculateKeybord, GridBasicCalc, GridIntegralCalc
)
class TabQWidget(QWidget):
    def __init__(self, tab):
        super().__init__()
        self.setLayout(tab)

class TabWidgetKeybord(QTabWidget):
    def __init__(self, window):
        super().__init__()
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
        self.addTab(TabQWidget(GridBasicCalc(window)), "Basic")
        self.addTab(TabQWidget(GridIntegralCalc(window)), "Integral")
        self.addTab(TabQWidget(QGridLayout()), "Tab 3")
        self.addTab(TabQWidget(QGridLayout()), "Tab 4")
