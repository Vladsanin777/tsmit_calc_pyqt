from PyQt6.QtWidgets import QGridLayout
from Button import ButtonDrag, ButtonDragAndDrop, ButtonBase
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
        window.line_edit.append(list())
        window.line_edit[0].append(LineEdit(window))
        window.line_edit[0][0].cursorEntered.connect(lambda: setattr(window, 'inputtin', [0, 0]))
        self.addWidget(window.line_edit[0][0])

class GridIntegralCalc(QGridLayout):
    def __init__(self, window):
        super().__init__()
        self.setSpacing(0)
        self.setContentsMargins(0, 0, 0, 0)
        window.local_histori_integral = HistoriScroll()
        window.add_local_histori_integral = window.local_histori_integral.getAddHistori()
        self.addWidget(window.local_histori_integral, 0, 0, 1, 5)
        window.line_edit.append(list())
        self.addWidget(ButtonBase("a = "), 1, 0, 1, 1)
        window.line_edit[1].append(LineEdit("a"))
        window.line_edit[1][0].cursorEntered.connect(lambda: setattr(window, 'inputtin', [1, 0]))
        self.addWidget(window.line_edit[1][0], 1, 1, 1, 1)
        self.addWidget(ButtonBase("b = "), 1, 2, 1, 1)
        window.line_edit[1].append(LineEdit(window))
        window.line_edit[1][1].cursorEntered.connect(lambda: setattr(window, 'inputtin', [1, 1]))
        self.addWidget(window.line_edit[1][1], 1, 3, 1, 2)
        window.line_edit[1].append(LineEdit(window))
        window.line_edit[1][2].cursorEntered.connect(lambda: setattr(window, 'inputtin', [1, 2]))
        self.addWidget(window.line_edit[1][2], 2, 0, 1, 5)
        

