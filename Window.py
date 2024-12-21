from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout,
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
from UI import CreateGradient

from LineEdit import LineEdit

# Main Content
class MainLayout(QVBoxLayout):
    def __init__(self, window):
        super().__init__()
        self.setContentsMargins(0, 0, 0, 0)
        self.setSpacing(0)
        self.addWidget(TitleBar(window))
        window.global_histori = HistoriScroll()
        window.add_global_histori = window.global_histori.add_histori
        window.resize_global_histori = window.global_histori.resize_histori
        self.addWidget(window.global_histori)
        self.addWidget(MainTabWidget(window))
        self.addLayout(GridCalculateCommon(window))
        self.addWidget(TabWidgetKeybord(window))
        
# Window
class Window(QWidget):

    __add_global_hitori: HistoriVBox
    __resize_global_histori: HistoriWidget
    __global_histori: HistoriScroll
    __set_for_result: QPushButton
    __add_local_histori: list[HistoriWidget]
    __resize_local_histori: list[HistoriWidget]
    __local_histori: list[HistoriScroll]
    __line_edit: list[list[LineEdit]]
    __inputtin: tuple[int, int]
    __result: list[list[str]]
    def __init__(self):
        
        self.__add_local_histori = list()
        self.__resize_local_histori = list()
        self.__local_histori = list()
        self.__line_edit = [[], []]
        self.__inputtin = 0, 0
        self.__result = [["0"], ["0", "0", "0", "0"]]
        super().__init__()
        self.setLayout(MainLayout(self))
        self.setWindowTitle("Calculate")
        self.resize(400, 800)
        self.setObjectName("window")
        self.show()
    def paintEvent(self, event) -> None:
        # Создаём QPainter для рисования
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Применяем градиент как кисть и заполняем окно
        painter.fillRect(self.rect(), 
            QBrush(CreateGradient(
                None, self,
                list_gradient = [
                    [0.0, QColor(100, 0, 0)],
                    [0.5, QColor(0, 0, 0)],
                    [1.0, QColor(0, 0, 100)]
                ]
            ))
        )
        return
    def local_histori_method(self, index):
        print(self.__local_histori[index])
        return self.__local_histori[index]
    @property
    def local_histori(self) -> HistoriScroll:
        return self.__local_histori[self.__inputtin[0]]
    @local_histori.setter
    def local_histori(self, new_local_histori: HistoriScroll) -> None:
        if isinstance(new_local_histori, HistoriScroll):
            self.__local_histori.append(new_local_histori)
            self.add_local_histori = new_local_histori.add_histori
            self.resize_local_histori = new_local_histori.resize_histori
        return
    @property
    def add_local_histori(self) -> HistoriVBox:
        return self.__add_local_histori[self.__inputtin[0]]
    
    @add_local_histori.setter
    def add_local_histori(self, new_add_local_histori: HistoriVBox) -> None:
        if isinstance(new_add_local_histori, HistoriVBox):
            self.__add_local_histori.append(new_add_local_histori)
        return
    
    @property
    def resize_local_histori(self) -> HistoriWidget:
        return self.__resize_local_histori[self.__inputtin[0]]
    @resize_local_histori.setter
    def resize_local_histori(self, new_resize_local_histori: HistoriWidget) -> None:
        if isinstance(new_resize_local_histori, HistoriWidget):
            self.__resize_local_histori.append(new_resize_local_histori)
        return
    @property
    def global_histori(self) -> HistoriScroll:
        return self.__global_histori
    @global_histori.setter
    def global_histori(self, new_global_histori: HistoriScroll) -> None:
        if isinstance(new_global_histori, HistoriScroll):
            self.__global_histori = new_global_histori
            self.__add_global_histori = new_global_histori.add_histori
            self.__resize_global_histori = new_global_histori.resize_histori
        return
    @property
    def add_global_histori(self) -> HistoriVBox:
        return self.__add_global_histori
    @add_global_histori.setter
    def add_global_histori(self, new_add_global_histori: HistoriVBox) -> None:
        if isinstance(new_add_global_histori, HistoriVBox):
            self.__add_global_histori = new_add_global_histori
        return
    @property
    def resize_global_histori(self) -> HistoriWidget:
        return self.__resize_global_histori
    @resize_global_histori.setter
    def resize_global_histori(self, new_resize_global_histori: HistoriWidget) -> None:
        if isinstance(new_resize_global_histori, HistoriWidget):
            self.__resize_global_histori = new_resize_global_histori
        return
    @property
    def line_edit(self) -> LineEdit:
        print(self.__line_edit)
        return self.__line_edit[self.__inputtin[0]][self.__inputtin[1]]
    @line_edit.setter
    def line_edit(self, value: tuple[int, LineEdit]) -> None:
        if isinstance(value, tuple) and len(value) == 2 and \
                isinstance(value[0], int) and isinstance(value[1], LineEdit):
            self.__line_edit[value[0]].append(value[1])
        return
    @property
    def result(self) -> str:
        return self.__result[self.__inputtin[0]][self.__inputtin[1]]
    @result.setter
    def result(self, new_result) -> None:
        if isinstance(new_result, str):
            self.__result[self.__inputtin[0]][self.__inputtin[1]] = new_result
        return
    def getResult(self, tab: int, line_edit: int) -> str:
        if isinstance(tab, int) and isinstance(line_edit, int):
            return self.__result[tab][line_edit]
    @property
    def inputtin(self) -> tuple[int, int]:
        return self.__inputtin
    @inputtin.setter
    def inputtin(self, value: tuple[int, int]) -> None:
        print(value, 156)
        if isinstance(value, tuple) and isinstance(value[0], int) and isinstance(value[1], int):
            self.__inputtin = value
        return

