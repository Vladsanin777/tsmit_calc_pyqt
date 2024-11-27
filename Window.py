from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QLineEdit,
    QPushButton
)
from Title import TitleBar
from CreateHistori import (
    HistoriScroll, HistoriWidget, HistoriVBox
)
from TabWindow import (
    MainTabWidget, TabWidgetKeybord
)
from Grid import (
    GridCalculateKeybord, GridCalculateCommon
)
# Main Content
class MainLayout(QVBoxLayout):
    def __init__(self, window):
        super().__init__()
        self.setContentsMargins(0, 0, 0, 0)
        self.setSpacing(0)
        self.addWidget(TitleBar(window))
        window.global_histori = HistoriScroll()
        window.add_global_histori = window.global_histori.getAddHistori()
        window.resize_global_histori = window.global_histori.getResizeHistori()
        self.addWidget(window.global_histori)
        self.addWidget(MainTabWidget(window))
        self.addLayout(GridCalculateCommon(window))
        self.addWidget(TabWidgetKeybord())
        
# Window
class Window(QWidget):

    add_global_hitori: HistoriVBox
    resize_global_histori: HistoriWidget
    global_histori: HistoriScroll
    set_for_result: QPushButton
    add_local_histori: list[HistoriWidget] = list()
    resize_local_histori: list[HistoriWidget] = list()
    local_histori: list[HistoriScroll] = list()
    line_edit: list[list[QLineEdit]] = list()
    inputtin: list[int] = [0, 0]
    result: list[str] = ["0", "0"]
    def __init__(self):


        super().__init__()
        self.setLayout(MainLayout(self))
        self.setWindowTitle("Calculate")
        self.resize(400, 800)
        self.setObjectName("window")
        self.show()
    def activateLocalHistori(self):
        return self.local_histori[self.inputtin[0]]
    def activateAddLocalHistori(self):
        return self.add_local_histori[self.inputtin[0]]
    def activateResizeLocalHistori(self):
        return self.resize_local_histori[self.inputtin[0]]
    def globalHistori(self):
        return self.global_histori
    def addGlobalHistori(self):
        return self.add_global_histori
    def resizeGlobalHistori(self):
        return self.resize_global_histori

