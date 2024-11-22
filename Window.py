from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout
)
from Title import TitleBar
from CreateHistori import HistoriScroll
from Data import Data
from TabWindow import (
    MainTabWidget, TabWidgetKeybord
)
from Grid import GridCalculateKeybord
# Main Content
class MainLayout(QVBoxLayout):
    def __init__(self):
        super().__init__()
        self.setContentsMargins(0, 0, 0, 0)
        self.setSpacing(0)
        self.addWidget(TitleBar())
        Data.global_histori = HistoriScroll()
        self.addWidget(Data.global_histori)
        Data.app.add_global_histori = Data.global_histori.getAddHistori()
        self.addWidget(MainTabWidget())
        self.addLayout(GridCalculateKeybord())
        self.addWidget(TabWidgetKeybord())
        
# Window
class Window(QWidget):
    def __init__(self):
        super().__init__()
        self.setLayout(MainLayout())
        self.setWindowTitle("Calculate")
        self.resize(400, 800)
        self.setObjectName("window")
        app.change_fon()
        self.show()

