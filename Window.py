from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QLineEdit,
    QPushButton
)
from Title import TitleBar
from CreateHistori import (
    HistoriScroll, HistoriVBox
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
        self.addWidget(window.global_histori)
        self.addWidget(MainTabWidget(window))
        self.addLayout(GridCalculateCommon(window))
        self.addWidget(TabWidgetKeybord())
        
# Window
class Window(QWidget):

    add_global_hitori: HistoriVBox
    global_histori: HistoriScroll
    add_local_histori_basic: HistoriVBox
    local_histori_basic: HistoriScroll
    line_edit_calc_basic: QLineEdit
    set_for_result_basic_calc: QPushButton
    result_basic_calc: str = "0"
    def __init__(self):


        super().__init__()
        self.setLayout(MainLayout(self))
        self.setWindowTitle("Calculate")
        self.resize(400, 800)
        self.setObjectName("window")
        self.show()

