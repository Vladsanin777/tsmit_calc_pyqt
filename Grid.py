from PyQt6.QtWidgets import QGridLayout
from BuildingKeybord import BuildingGridKeybord
from Button import ButtonDrag

#Basic calculate
class GridCalculateKeybord(QGridLayout):
    def __init__(self):
        super().__init__()
        self.setSpacing(0)
        self.setContentsMargins(0, 0, 0, 0)
        self.button_for_calc_basic("_ALL", 4, 0)

        self.button_for_calc_basic("_DO", 10, 1)
        self.button_for_calc_basic("_POST", 10, 3)
        self.button_for_calc_basic("_O", 4, 4)
        BuildingGridKeybord([
            ["()", "(", ")", "mod", "_PI"], 
            ["7", "8", "9", ":", "sqrt"], 
            ["4", "5", "6", "*", "^"], 
            ["1", "2", "3", "-", "!"], 
            ["0", ".", "%", "+", "_E"]
        ], self, 5)
        global set_for_result_basic_calc, result_basic_calc
        self.addWidget(set_for_result_basic_calc := ButtonDrag(result_basic_calc), 10, 0, 1, 2) 
        self.button_for_calc_basic("=", 10, 4)

    def button_for_calc_basic(self, label: str, row: int, column: int) -> None:
        self.addWidget(ButtonDrag(label, css_name = "keybord"), row, column, 1, 1)


