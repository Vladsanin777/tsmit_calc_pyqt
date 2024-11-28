from Calculate import Calculate 
from NewHistoriElement import BoxHistoriElement
class LogicCalculate():
    entry_text: str = ""

    def __init__(self, line_edit_text: str, window):
        self.line_edit_text = line_edit_text
        self.window = window
        print(line_edit_text, window)

    def getLineEditAndResult(self) -> list:
        return (self.window.line_edit[self.window.inputtin[0]][self.window.inputtin[1]], 
            self.window.result[self.window.inputtin[0]]
        )

    def button__ALL(self):
        if (line_edit_text := "".join(self.line_edit_text.split("_ALL"))) != "":
            self.window.add_global_histori.addLayout(BoxHistoriElement(line_edit_text, str(self.window.result[self.window.inputtin[0]])))
            self.window.activateLocalHistori().setLayout(BoxHistoriElement(line_edit_text, str(self.window.result[self.window.inputtin[0]])))
            self.window.result_basic_calc = "0"
            self.window.set_for_result.setText(self.window.result[self.window.inputtin[0]])
        self.window.line_edit[self.window.inputtin[0]][self.window.inputtin[1]].setText("")

        
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

        line_edit, result = self.getLineEditAndResult()
        if (line_edit_text := "".join(line_edit_text_list := self.line_edit_text.split("="))) != "":
            self.window.add_global_histori.addLayout(BoxHistoriElement(line_edit_text, str(result)))
            self.window.resize_global_histori.adjustSize()
            self.window.global_histori.verticalScrollBar().setValue(self.window.global_histori.verticalScrollBar().maximum())
            self.window.activateAddLocalHistori().addLayout(BoxHistoriElement(line_edit_text, str(result)))
            self.window.activateResizeLocalHistori().adjustSize()
            scroll_histori = self.window.activateLocalHistori()
            scroll_histori.verticalScrollBar().setValue(scroll_histori.verticalScrollBar().maximum())
        line_edit.setText(result)
        line_edit.setCursorPosition(len(result)-1)


    def button__O(self) -> None:
        if (line_edit_text := "".join(line_edit_text_list := self.line_edit_text.split("_O"))) != "":
            element_position = len(line_edit_text_list[0])-1
            print(element_position, "22")
            self.window.line_edit_calc_basic.setText(self.line_edit_text[:element_position] + self.line_edit_text[element_position+3:])
            
    def button_other(self) -> None:
        print(3)
        result = self.window.result[self.window.inputtin[0]] = Calculate(self.line_edit_text).calc()
        self.window.set_for_result.setText(result)
    
    @staticmethod
    def inputing_line_edit(button, window) -> None:
        label: str = button.text()
        line_edit = window.activateLineEdit()
        text: str = line_edit.text()
        position_cursor: int = line_edit.cursorPosition()
        line_edit.setText(text[:position_cursor] + label + text[position_cursor:])
        line_edit.setCursorPosition(position_cursor + len(label))


