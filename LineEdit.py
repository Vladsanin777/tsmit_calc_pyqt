from PyQt6.QtWidgets import QLineEdit
from PyQt6.QtCore import pyqtSignal
from PyQt6.QtGui import QFocusEvent
from LogicButton import LogicCalculate
class LineEdit(QLineEdit):
    cursorEntered = pyqtSignal()
    def __init__(self, window):
        super().__init__()
        self.window = window
        self.setSizePolicy(self.sizePolicy().Policy.Expanding, self.sizePolicy().Policy.Expanding)
        self.setObjectName("keybord")
        self.textChanged.connect(self.on_entry_changed)
    def focusInEvent(self, event: QFocusEvent):
            if event.reason():
                self.cursorEntered.emit()  # Эмитируем сигнал, когда фокус установлен
            super().focusInEvent(event)  # Вызываем стандартное поведение

    def on_entry_changed(self, text_line_edit):
        logic_calc = LogicCalculate(text_line_edit, self.window)
        if text_line_edit: # если убрать это условие то будет срабатывать button other() с самого начала и будет ошибка
            if "_ALL" in text_line_edit:
                logic_calc.button__ALL()
            elif "_DO" in text_line_edit:
                logic_calc.button__DO()
            elif "_POST" in text_line_edit:
                logic_calc.button__POST()
            elif "_O" in text_line_edit:
                logic_calc.button__O()
            elif "=" in text_line_edit:
                logic_calc.button_result()
            elif "_RES" in text_line_edit:
                logic_calc.button_result()
            else:
                logic_calc.button_other()

