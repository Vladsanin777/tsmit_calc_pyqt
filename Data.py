from dataclasses import dataclass
from PyQt6.QtWidgets import QApplication
@dataclass
class Data:
    app: QApplication
    count_window: int = 0
