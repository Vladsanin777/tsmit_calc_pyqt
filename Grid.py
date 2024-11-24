from PyQt6.QtWidgets import QGridLayout
from Button import ButtonDrag, ButtonDragAndDrop
from CreateHistori import HistoriScroll
from LineEdit import LineEdit

class BuildingGridKeybord():
    def __init__(self, list_button: list[list[str]], grid: QGridLayout, row: int = 0):
        for row_labels_for_button in list_button:
            column: int = 0
            for one_button in row_labels_for_button:
                grid.addWidget(ButtonDragAndDrop(one_button), row, column, 1, 1)
                column += 1
            row += 1

class GridCalculateKeybord(QGridLayout):
    def __init__(self, list_button: list[list[str]]):
        super().__init__()

        self.setContentsMargins(0, 0, 0, 0)
        self.setSpacing(0)
        BuildingGridKeybord(list_button, self)


class GridCalculateCommon(QGridLayout):
    def __init__(self, window):
        super().__init__()
        self.setSpacing(0)
        self.setContentsMargins(0, 0, 0, 0)
        self.button("_ALL", 0, 0)

        self.button("_DO", 0, 1)
        self.button("_RES", 0, 2)
        self.button("_POST", 0, 3)
        self.button("_O", 0, 4)
        BuildingGridKeybord([
            ["()", "(", ")", "mod", "_PI"], 
            ["7", "8", "9", ":", "sqrt"], 
            ["4", "5", "6", "*", "^"], 
            ["1", "2", "3", "-", "!"], 
            ["0", ".", "%", "+", "_E"]
        ], self, 1)
        window.set_for_result_basic_calc = ButtonDrag(window.result_basic_calc)
        self.addWidget(window.set_for_result_basic_calc, 6, 0, 1, 2) 
        self.button("", 6, 2)
        self.button("", 6, 3)
        self.button("=", 6, 4)

    def button(self, label: str, row: int, column: int) -> None:
        self.addWidget(ButtonDrag(label, css_name = "keybord"), row, column, 1, 1)


class GridBasicCalc(QGridLayout):
    def __init__(self, window):
        super().__init__()
        self.setSpacing(0)
        self.setContentsMargins(0, 0, 0, 0)
        window.local_histori_basic = HistoriScroll()
        window.add_local_histori_basic = window.local_histori_basic.getAddHistori()
        self.addWidget(window.local_histori_basic)
        window.line_edit_calc_basic = LineEdit(window)
        self.addWidget(window.line_edit_calc_basic)
