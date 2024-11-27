from PyQt6.QtWidgets import (
    QScrollArea, QWidget, QVBoxLayout
)
from PyQt6.QtCore import Qt

class HistoriVBox(QVBoxLayout):
    def __init__(self):
        super().__init__()
        self.setSpacing(0)
        self.setContentsMargins(0, 0, 0, 0)

class HistoriWidget(QWidget):
    _add_histori: HistoriVBox
    def __init__(self):
        super().__init__()
        self.setObjectName("histori")
        self._add_histori = HistoriVBox()
        self.setLayout(self._add_histori)
        self.setContentsMargins(0, 0, 0, 0)
    def getAddHistori(self):
        return self._add_histori

class HistoriScroll(QScrollArea):
    _add_histori: HistoriVBox
    _resize_histori: HistoriWidget
    def __init__(self):
        super().__init__()
        # self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOn)  # Всегда отображать полосу прокрутки
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)  # Отключить горизонтальную прокрутку
        self.setWidgetResizable(True)  # Виджет содержимого будет автоматически подгоняться под ширину
        self.setMinimumHeight(150)
        self._resize_histori = HistoriWidget()
        self.setWidget(self._resize_histori)
        self._add_histori = self._resize_histori.getAddHistori()
    def getResizeHistori(self):
        return self._resize_histori
    def getAddHistori(self):
        return self._add_histori
