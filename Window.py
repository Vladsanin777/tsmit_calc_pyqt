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

"""
class GradientWindow(QLinearGradient):
    def __init__(self):
        super().__init__(0, 0, 1, 1)
        self.setCoordinateMode(QLinearGradient.CoordinateMode.ObjectBoundingMode)
        self.setColorAt(0.0, QColor("rgb(140, 0, 0)"))
        self.setColorAt(0.5, QColor("black"))
        self.setColorAt(1.0, QColor("rgb(0, 0, 140)"))
"""



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
        self.addWidget(TabWidgetKeybord(window))
        
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
    result: list[str] = [["0"], ["0", "0", "0", "0"]]
    def __init__(self):


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
        return self.window.result[self.window.inputtin[0]][self.window.inputtin[1]]
