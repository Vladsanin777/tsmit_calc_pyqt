from Calculate import Calculate, Integral 
from NewHistoriElement import BoxHistoriElement
class LogicCalculate():
    entry_text: str = ""

    def __init__(self, line_edit_text: str, window):
        self.line_edit_text = line_edit_text
        self.window = window
        print(line_edit_text, window)


    def button__ALL(self):
        if (line_edit_text := "".join(self.line_edit_text.split("_ALL"))) != "":
            self.window.add_global_histori.addLayout(BoxHistoriElement(line_edit_text, self.window))
            self.window.resize_global_histori.adjustSize()
            self.window.global_histori.verticalScrollBar().setValue(self.window.global_histori.verticalScrollBar().maximum())

            self.window.add_local_histori.addLayout(BoxHistoriElement(line_edit_text, self.window))

            self.window.resize_local_histori.adjustSize()
            scroll_histori = self.window.local_histori
            scroll_histori.verticalScrollBar().setValue(scroll_histori.verticalScrollBar().maximum())
            self.window.result = "0"

        self.window.line_edit.setText("")

        
    def button__DO(self):
        if (line_edit_text := "".join(line_edit_text_list := self.line_edit_text.split("_DO"))) != "":
            self.window.add_global_histori.addLayout(BoxHistoriElement(line_edit_text, self.window))
            self.window.add_local_histori.addLayout(BoxHistoriElement(line_edit_text, self.window))
        self.window.line_edit.setText(line_edit_text_list[1])

    def button__POS(self):
        if (line_edit_text := "".join(line_edit_text_list := self.line_edit_text.split("_POS"))) != "":
            self.window.add_global_histori.addLayout(BoxHistoriElement(line_edit_text, self.window))
            self.window.add_local_histori.addLayout(BoxHistoriElement(line_edit_text, self.window))
        self.window.line_edit.setText(line_edit_text_list[0])

    def button__RES(self):
        result = self.window.result
        if (line_edit_text := "".join(line_edit_text_list := self.line_edit_text.split("_RES"))) != "":
            self.window.add_global_histori.addLayout(BoxHistoriElement(line_edit_text, self.window))
            self.window.resize_global_histori.adjustSize()
            self.window.global_histori.verticalScrollBar().setValue(self.window.global_histori.verticalScrollBar().maximum())
            self.window.add_local_histori.addLayout(BoxHistoriElement(line_edit_text, self.window))
            self.window.resize_local_histori.adjustSize()
            scroll_histori = self.window.local_histori
            scroll_histori.verticalScrollBar().setValue(scroll_histori.verticalScrollBar().maximum())
        line_edit = self.window.line_edit
        line_edit.setText(result)
        line_edit.setCursorPosition(len(result)-1)


    def button__O(self) -> None:
        if (line_edit_text := "".join(line_edit_text_list := self.line_edit_text.split("_O"))) != "":
            element_position = len(line_edit_text_list[0])-1
            print(element_position, "22")
            self.window.line_edit.setText(self.line_edit_text[:element_position] + self.line_edit_text[element_position+3:])
            
    def button_other(self) -> None:
        print(3)
        window = self.window
        print(
            Calculate(self.line_edit_text) 
            if window.inputtin != (1, 3) else 
            Integral(
                a = window.getResult(1, 1), 
                b = window.getResult(1, 2), 
                EPS = window.getResult(1, 0), 
                equation = self.line_edit_text
            )
        )
        window.result = (
            result := str(
                Calculate(self.line_edit_text) 
                if window.inputtin != (1, 3) else 
                Integral(
                    a = window.getResult(1, 1), 
                    b = window.getResult(1, 2), 
                    EPS = window.getResult(1, 0), 
                    equation = self.line_edit_text
                )
            )
        )
        window.set_for_result.setText(result)
    
    @staticmethod
    def inputing_line_edit(button, window) -> None:
        label: str = button.text()
        line_edit = window.line_edit
        text: str = line_edit.text()
        position_cursor: int = line_edit.cursorPosition()
        line_edit.setText(text[:position_cursor] + label + text[position_cursor:])
        line_edit.setCursorPosition(position_cursor + len(label))


