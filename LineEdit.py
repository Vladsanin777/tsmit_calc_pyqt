from PyQt6.QtWidgets import QLineEdit
from PyQt6.QtGui import QDrag, QPainter, QLinearGradient, QFont, QBrush, QTextOption, QColor, QFontMetrics, QPainterPath, QPen
from PyQt6.QtCore import pyqtSignal, Qt
from PyQt6.QtGui import QFocusEvent
from LogicButton import LogicCalculate
from UI import StyleButton, StyleLineEdit
class LineEdit(QLineEdit):
    __inputtin: tuple[int, int]
    def __init__(self, window, inputtin: tuple[int, int], *, text = ""):
        super().__init__()
        self.setText(text)
        self.window = window
        self.setSizePolicy(self.sizePolicy().Policy.Expanding, self.sizePolicy().Policy.Expanding)
        self.setObjectName("keybord")
        self.textChanged.connect(self.on_entry_changed)
        font = self.font()
        font.setPointSize(25)
        self.setFont(font)
        self.setMaximumHeight(40)
        self.setMinimumWidth(40)
        self.setContentsMargins(0, 0, 0, 0)
        self.__inputtin = inputtin
    def focusInEvent(self, event: QFocusEvent):
            if event.reason():
                self.window.inputtin = self.__inputtin
            super().focusInEvent(event)  # Вызываем стандартное поведение

    def on_entry_changed(self, text_line_edit):
        logic_calc = LogicCalculate(text_line_edit, self.window)
        if "_ALL" in text_line_edit:
            logic_calc.button__ALL()
        elif "_DO" in text_line_edit:
            logic_calc.button__DO()
        elif "_POS" in text_line_edit:
            logic_calc.button__POS()
        elif "_O" in text_line_edit:
            logic_calc.button__O()
        elif "=" in text_line_edit:
            pass
        elif "_RES" in text_line_edit:
            logic_calc.button__RES()
        else:
            logic_calc.button_other()
    def paintEvent(self, event):
        super().paintEvent(event)
        StyleLineEdit(self, self.window)
