from PyQt6.QtWidgets import QGridLayout, QHBoxLayout
from Button import ButtonDrag, ButtonDragAndDrop, ButtonBase
from CreateHistori import HistoriScroll
from LineEdit import LineEdit
from LogicButton import LogicCalculate
from UI import CreateGradient
from typing import Self

class BuildingGridKeybord():
    def __init__(self, list_button: list[list[str]], grid: QGridLayout, window, row: int = 0, *, button = ButtonDrag):
        for row_labels_for_button in list_button:
            column: int = 0
            for one_button in row_labels_for_button:
                grid.addWidget(button(one_button, callback = LogicCalculate.inputing_line_edit, window = window), row, column, 1, 1)
                column += 1
            row += 1

class GridCalculateKeybord(QGridLayout):
    def __init__(self, list_button: list[list[str]], window):
        super().__init__()

        self.setContentsMargins(0, 0, 0, 0)
        self.setSpacing(0)
        BuildingGridKeybord(list_button, self, window)


class GridCalculateCommon(QGridLayout):
    def __init__(self, window):
        super().__init__()
        self.setSpacing(0)
        self.setContentsMargins(0, 0, 0, 0)

        self.window = window

        self.button("_ALL", 0, 0)

        self.button("_DO", 0, 1)
        self.button("_RES", 0, 2)
        self.button("_POS", 0, 3)
        self.button("_O", 0, 4)
        BuildingGridKeybord([
            ["()", "(", ")", "mod", "_PI"], 
            ["7", "8", "9", ":", "sqrt"], 
            ["4", "5", "6", "*", "^"], 
            ["1", "2", "3", "-", "!"], 
            ["0", ".", "%", "+", "_E"],
            ["", "", "", "", ""]
        ], self, window, 1, button = ButtonDragAndDrop)
        window.set_for_result = ButtonDrag(window.result[0][0], window = window)
        self.addWidget(window.set_for_result, 7, 0, 1, 5) 

    def button(self, label: str, row: int, column: int, *, button = ButtonDrag) -> None:
        self.addWidget(button(label, css_name = "keybord", callback = LogicCalculate.inputing_line_edit, window = self.window), row, column, 1, 1)

class GridBaseCalc(QGridLayout):
    def __init__(self, window):
        super().__init__()
        self.setSpacing(0)
        self.setContentsMargins(0, 0, 0, 0)
        local_histori = HistoriScroll()
        window.local_histori = local_histori
        self.addWidget(local_histori, 0, 0, 1, 6)

class GridBasicCalc(GridBaseCalc):
    def __init__(self, window):
        super().__init__(window)
        line_edit = LineEdit(window, (0, 0))
        window.line_edit = 0, line_edit
        self.addWidget(line_edit)


class GridIntegralCalc(GridBaseCalc):
    def __init__(self, window):
        super().__init__(window)

        self.addWidget(ButtonBase("a = ", css_name = "calculate", width = 64, window = window), 1, 0, 1, 1)
        a_line_edit = LineEdit(window, (1, 0), text = "1")
        window.line_edit = 1, a_line_edit
        self.addWidget(a_line_edit, 1, 1, 1, 2)
        self.addWidget(ButtonBase("b = ", css_name = "calculate", width = 64, window = window), 1, 3, 1, 1)
        b_line_edit = LineEdit(window, (1, 1), text = "2")
        window.line_edit = 1, b_line_edit
        self.addWidget(b_line_edit, 1, 4, 1, 2)
        main_line_edit = LineEdit(window, (1, 2))
        window.line_edit = 1, main_line_edit
        self.addWidget(main_line_edit, 2, 0, 1, 6)

class GridDerivativeOrIntegrateCalc(GridBaseCalc):
    def __init__(self: Self, window, number_tab: int):
        super().__init__(window)
        line_edit = LineEdit(window, (number_tab, 0))
        window.line_edit = number_tab, line_edit
        self.addWidget(line_edit)
class GridReplacementCalc(GridBaseCalc):
    def __init__(self: Self, window):
        super().__init__(window)
        self.addWidget(ButtonBase("with =", css_name = "calculate", width = 100, window = window), 1, 0, 1, 1)
        with_line_edit = LineEdit(window, (4, 0), text = "x")
        window.line_edit = 4, with_line_edit
        self.addWidget(with_line_edit, 1, 1, 1, 2)
        self.addWidget(ButtonBase("on =", css_name = "calculate", width = 100, window = window), 1, 3, 1, 1)
        on_line_edit = LineEdit(window, (4, 1), text = "0")
        window.line_edit = 4, on_line_edit
        self.addWidget(on_line_edit, 1, 4, 1, 2)
        main_line_edit = LineEdit(window, (4, 2))
        window.line_edit = 4, main_line_edit
        self.addWidget(main_line_edit, 2, 0, 1, 6)       

