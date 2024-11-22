from dataclasses import dataclass
from PyQt6.QtWidgets import (
    QVBoxLayout, QLineEdit, 
    QPushButton, QApplication
)
from CreateHistori import HistoriScroll
@dataclass
class Data:
    add_global_hitori: QVBoxLayout
    global_histori: HistoriScroll
    add_local_histori_basic: QVBoxLayout
    local_histori_basic: HistoriScroll
    line_edit_calc_basic: QLineEdit
    set_for_result_basic_calc: QPushButton
    app: QApplication
    result_basic_calc: str = "0"
