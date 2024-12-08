from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QLineEdit,
    QPushButton
)
from PyQt6.QtGui import QPainter, QLinearGradient, QColor, QBrush
from PyQt6.QtCore import Qt
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
        window.main_tab_widget = MainTabWidget(window)
        self.addWidget(window.main_tab_widget)
        print(4)
        self.addLayout(GridCalculateCommon(window))
        window.tab_widget_keybord = TabWidgetKeybord(window)
        self.addWidget(window.tab_widget_keybord)
        
# Window
class Window(QWidget):

    add_global_hitori: HistoriVBox
    resize_global_histori: HistoriWidget
    global_histori: HistoriScroll
    set_for_result: QPushButton
    add_local_histori: list[HistoriWidget]
    resize_local_histori: list[HistoriWidget]
    local_histori: list[HistoriScroll]
    line_edit: list[list[QLineEdit]]
    inputtin: list[int]
    result: list[list[str]]
    def __init__(self):
        
        self.add_local_histori = list()
        self.resize_local_histori = list()
        self.local_histori = list()
        self.line_edit = list()
        self.inputtin = [0, 0]
        self.result = [["0"], ["0", "0", "0", "0"]]
        super().__init__()
        self.setLayout(MainLayout(self))
        self.setWindowTitle("Calculate")
        self.resize(400, 800)
        self.setObjectName("window")
        #self.gradient = GradientWindow()
        self.show()
    def paintEvent(self, event):
        # Создаём градиент, который охватывает всю область окна
        
        gradient = QLinearGradient(0, 0, self.width(), self.height())
        gradient.setColorAt(0.0, QColor(100, 0, 0))  # Красный
        gradient.setColorAt(0.5, QColor(0, 0, 0))    # Чёрный
        gradient.setColorAt(1.0, QColor(0, 0, 100))  # Синий

        # Создаём QPainter для рисования
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Применяем градиент как кисть и заполняем окно
        brush = QBrush(gradient)
        painter.fillRect(self.rect(), brush)
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
    def activateLineEdit(self):
        return self.line_edit[self.inputtin[0]][self.inputtin[1]]
    def activateResult(self):
        return self.result[self.inputtin[0]][self.inputtin[1]]
    def activateSetResult(self, new_result):
        print(self.result)
        self.result[self.inputtin[0]][self.inputtin[1]] = new_result
