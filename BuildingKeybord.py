from PyQt6.QtWidgets import QGridLayout

class BuildingGridKeybord():
    def __init__(self, list_button: list[list[str]], grid: QGridLayout, row: int = 0):
        for row_labels_for_button in list_button:
            column: int = 0
            for one_button in row_labels_for_button:
                grid.addWidget(ButtonDragAndDrop(one_button), row, column, 1, 1)
                column += 1
            row += 1

class GridCalcBasicKeybord(QGridLayout):
    def __init__(self, list_button: list[list[str]]):
        super().__init__()

        self.setContentsMargins(0, 0, 0, 0)
        self.setSpacing(0)
        BuildingButtonInGridLayout(list_button, self)


