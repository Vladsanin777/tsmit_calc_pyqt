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
            ["0", ".", "%", "+", "_E"]
        ], self, window, 1, button = ButtonDragAndDrop)
        window.set_for_result = ButtonDrag(window.result[0][0], window = window)
        self.addWidget(window.set_for_result, 6, 0, 1, 2) 
        self.button("", 6, 2, button = ButtonDragAndDrop)
        self.button("", 6, 3, button = ButtonDragAndDrop)
        self.button("=", 6, 4)

    def button(self, label: str, row: int, column: int, *, button = ButtonDrag) -> None:
        self.addWidget(button(label, css_name = "keybord", callback = LogicCalculate.inputing_line_edit, window = self.window), row, column, 1, 1)


class GridBasicCalc(QGridLayout):
    def __init__(self, window):
        super().__init__()
        self.setSpacing(0)
        self.setContentsMargins(0, 0, 0, 0)
        local_histori = HistoriScroll()
        window.local_histori = local_histori
        self.addWidget(local_histori)
        line_edit = LineEdit(window, (0, 0))
        window.line_edit = 0, line_edit
        self.addWidget(line_edit)


class GridIntegralCalc(QGridLayout):
    def __init__(self, window):
        print(window)
        super().__init__()
        self.setSpacing(0)
        self.setContentsMargins(0, 0, 0, 0)
        local_histori = HistoriScroll()
        window.local_histori = local_histori
        self.addWidget(local_histori, 0, 0, 1, 6)

        self.addWidget(ButtonBase("a = ", css_name = "calculate", max_width = 45, window = window), 1, 0, 1, 1)
        a_line_edit = LineEdit(window, (1, 1))
        window.line_edit = 1, a_line_edit
        self.addWidget(a_line_edit, 1, 1, 1, 2)
        self.addWidget(ButtonBase("b = ", css_name = "calculate", max_width = 45, window = window), 1, 3, 1, 1)
        b_line_edit = LineEdit(window, (1, 2))
        window.line_edit = 1, b_line_edit
        self.addWidget(b_line_edit, 1, 4, 1, 2)
        main_line_edit = LineEdit(window, (1, 3))
        window.line_edit = 1, main_line_edit
        self.addWidget(main_line_edit, 2, 0, 1, 6)

class GridDerivateCalc(QGridLayout):
    def __init__(self: Self, window):
        super().__init__()
        self.setSpacing(0)
        self.setContentsMargins(0, 0, 0, 0)
        local_histori = HistoriScroll()
        window.local_histori = local_histori
        self.addWidget(local_histori, 0, 0, 1, 6)
        ordinar_derivate_line_edit = LineEdit(window, (2, 1))
        window.line_edit = 2, ordinar_derivate_line_edit
        self.addWidget(ordinar_derivate_line_edit, 1, 0, 1, 6)
        ordinar_derivate_line_edit = LineEdit(window, (2, 1))
        window.line_edit = 2, ordinar_derivate_line_edit
        self.addWidget(ordinar_derivate_line_edit, 2, 0, 1, 6)
