from PyQt6.QtWidget import (
    QTabWidget, QGridLayout
)
from QWidget
class MainTabWidget(QTabWidget):
    def __init__(self):
        super().__init__()
        self.tabBar().setExpanding(True)
        self.addTab(TabQWidget(QGridLayout()), "Basic")
        self.addTab(TabQWidget(QGridLayout()), "Tab 2")
        self.addTab(TabQWidget(QGridLayout()), "Tab 3")
        self.addTab(TabQWidget(QGridLayout()), "Tab 4")
