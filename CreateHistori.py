from PyQt6.QtWidgets import (
    QScrollArea, QWidget, QVBoxLayout
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPalette, QColor, QPainter

class HistoriVBox(QVBoxLayout):
    def __init__(self):
        super().__init__()
        self.setSpacing(0)
        self.setContentsMargins(0, 0, 0, 0)
        self.setObjectName("histori")

class HistoriWidget(QWidget):
    __add_histori: HistoriVBox
    def __init__(self):
        super().__init__()
        self.setObjectName("histori")
        self.__add_histori = HistoriVBox()
        self.setLayout(self.__add_histori)
        self.setContentsMargins(0, 0, 0, 0)
    @property
    def add_histori(self):
        return self.__add_histori

class HistoriScroll(QScrollArea):
    __add_histori: HistoriVBox
    __resize_histori: HistoriWidget
    def __init__(self):
        super().__init__()
        # self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOn)  # Всегда отображать полосу прокрутки
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)  # Отключить горизонтальную прокрутку
        self.setObjectName("histori")
        self.setWidgetResizable(True)  # Виджет содержимого будет автоматически подгоняться под ширину
        self.setMinimumHeight(100)
        self.__resize_histori = HistoriWidget()
        self.setWidget(self.__resize_histori)
        self.__add_histori = self.__resize_histori.add_histori

    @property
    def resize_histori(self):
        return self.__resize_histori
    @property
    def add_histori(self):
        return self.__add_histori
    
