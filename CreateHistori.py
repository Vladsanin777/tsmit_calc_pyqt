from PyQt6.QtWidgets import (
    QScrollArea, QVBoxLayout
)
class HistoriVBoxLayout(QVBoxLayout):
    def __init__(self):
        super().__init__()
        self.setSpacing(0)
        self.setContentsMargins(0, 0, 0, 0)

class HistoriScroll(QScrollArea):
    _add_histori: QVBoxLayout
    def __init__(self):
        super().__init__()
        self._add_histori = HistoriVBoxLayout()
        self.setLayout(self._add_histori)
    def getAddHistori(self):
        return self._add_histori
