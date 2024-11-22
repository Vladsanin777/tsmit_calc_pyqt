from PyQt6.QtWidgets import (
    QWidget, QTabWidget, QGridLayout
)
from Grid import GridCalculateKeybord
class TabQWidget(QWidget):
    def __init__(self, tab):
        super().__init__()
        self.setLayout(tab)

class TabWidgetKeybord(QTabWidget):
    def __init__(self):
        super().__init__()
        self.addTab(TabQWidget(GridCalcBasicKeybord([["1", "2", "3", "4", "5"], ["6", "7", "8", "9", "0"]])), "digits 10")
        self.addTab(TabQWidget(GridCalcBasicKeybord([["A", "B", "C"], ["D", "E", "F"]])), "digits 16")
        self.addTab(TabQWidget(GridCalcBasicKeybord([["+", "-", ":", "*", "^"], ["!", "sqrt", "ln", "log", "lg"]])), "operators")
        self.addTab(TabQWidget(GridCalcBasicKeybord([["_E", "_PI"]])), "consts")
        self.addTab(TabQWidget(GridCalcBasicKeybord([["round", "mod", "0x"], ["0b", "0t", ","]])), "other")



#Main TabWidget

class MainTabWidget(QTabWidget):
    def __init__(self):
        super().__init__()
        self.tabBar().setExpanding(True)
        self.addTab(TabQWidget(QGridLayout()), "Basic")
        self.addTab(TabQWidget(QGridLayout()), "Tab 2")
        self.addTab(TabQWidget(QGridLayout()), "Tab 3")
        self.addTab(TabQWidget(QGridLayout()), "Tab 4")


