from Calculate import Calculate 
from NewHistoriElement import BoxHistoriElement
class LogicCalculate():
    entry_text: str = ""

    def __init__(self, line_edit_text: str, window):
        self.line_edit_text = line_edit_text
        self.window = window

    def button__ALL(self):
        if (line_edit_text := "".join(self.line_edit_text.split("_ALL"))) != "":
            self.window.add_global_histori.addLayout(BoxHistoriElement(line_edit_text, str(self.window.result_basic_calc)))
            self.window.add_local_histori_basic.addLayout(BoxHistoriElement(line_edit_text, str(self.window.result_basic_calc)))
            self.window.result_basic_calc = "0"
            self.window.set_for_result_basic_calc.setText(self.window.result_basic_calc)
        self.window.line_edit_calc_basic.setText("")

        
    def button__DO(self):
        if (line_edit_text := "".join(line_edit_text_list := self.line_edit_text.split("_DO"))) != "":
            self.window.add_global_histori.addLayout(BoxHistoriElement(line_edit_text, str(self.window.result_basic_calc)))
            self.window.add_local_histori_basic.addLayout(BoxHistoriElement(line_edit_text, str(self.window.result_basic_calc)))
        self.window.line_edit_calc_basic.setText(line_edit_text_list[1])

    def button__POST(self):
        if (line_edit_text := "".join(line_edit_text_list := self.window.line_edit_calc_basic.text().split("_POST"))) != "":
            self.window.add_global_histori.addLayout(BoxHiskoriElement(line_edit_text, str(self.window.result_basic_calc)))
            self.window.add_local_histori_basic.addLayout(BoxHistoriElement(line_edit_text, str(self.window.result_basic_calc)))
        self.window.line_edit_calc_basic.setText(line_edit_text_list[0])

    def button_result(self):
        if (line_edit_text := "".join(line_edit_text_list := self.line_edit_text.split("="))) != "":
            self.window.add_global_histori.addLayout(BoxHistoriElement(line_edit_text, str(self.window.result_basic_calc)))
            self.window.add_local_histori_basic.addLayout(BoxHistoriElement(line_edit_text, str(self.window.result_basic_calc)))
        self.window.line_edit_calc_basic.setText(self.window.result_basic_calc)
        self.window.line_edit_calc_basic.setCursorPosition(len(self.window.result_basic_calc)-1)


    def button__O(self) -> None:
        if (line_edit_text := "".join(line_edit_text_list := self.line_edit_text.split("_O"))) != "":
            element_position = len(line_edit_text_list[0])-1
            print(element_position, "22")
            self.window.line_edit_calc_basic.setText(self.line_edit_text[:element_position] + self.line_edit_text[element_position+3:])
            
    def button_other(self) -> None:
        self.window.result_basic_calc = Calculate(self.line_edit_text).calc()
        print(self.window.result_basic_calc)
        self.window.set_for_result_basic_calc.setText(self.window.result_basic_calc)
    
    @staticmethod
    def inputing_line_edit(button, window) -> None:
        global line_edit_calc_basic
        label: str = button.text()
        text: str = window.line_edit_calc_basic.text()
        position_cursor: int = window.line_edit_calc_basic.cursorPosition()
        window.line_edit_calc_basic.setText(text[:position_cursor] + label + text[position_cursor:])
        window.line_edit_calc_basic.setCursorPosition(position_cursor + len(label))


