from Calculate import Calculate, Integral, Derivative
from NewHistoriElement import (
    CustomBoxHistoriElement,
    BasicBoxHistoriElement
)
class LogicCalculate():
    entry_text: str = ""

    def __init__(self, line_edit_text: str, window):
        self.line_edit_text = line_edit_text
        self.window = window
        print(line_edit_text, window)


    def button__ALL(self):
        if (line_edit_text := "".join(self.line_edit_text.split("_ALL"))) != "":
            
            self.line_edit_text = line_edit_text
            self.button_other()
            self.add_histori(self.window, line_edit_text)

            self.window.result = "0"

        self.window.line_edit.setText("")

        
    def button__DO(self):
        if (line_edit_text := "".join(line_edit_text_list := self.line_edit_text.split("_DO"))) != "":
            self.line_edit_text = line_edit_text
            self.button_other()
            self.add_histori(self.window, line_edit_text)
        self.window.line_edit.setText(line_edit_text_list[1])

    def button__POS(self):
        if (line_edit_text := "".join(line_edit_text_list := self.line_edit_text.split("_POS"))) != "":
            self.line_edit_text = line_edit_text
            self.button_other()
            self.add_histori(self.window, line_edit_text)
        self.window.line_edit.setText(line_edit_text_list[0])

    def button__RES(self):
        window = self.window
        result = window.result
        if (line_edit_text := "".join(line_edit_text_list := self.line_edit_text.split("_RES"))) != "":
            self.line_edit_text = line_edit_text
            self.button_other()
            self.add_histori(window, line_edit_text)
        line_edit = self.window.line_edit
        line_edit.setText(result)
        line_edit.setCursorPosition(len(result)-1)
    def add_histori(self, window, line_edit_text):
        tab = window.inputtin[0]
        match tab:
            case 1:

                cls = lambda: CustomBoxHistoriElement(
                    line_edit_text, self.window, 
                    number_tab = 1, name_operation = "Integral",
                    label_1 = "a", text_1 = window.getResult(1, 0),
                    label_2 = "b", text_2 = window.getResult(1, 1)
                )
            case 4:
                cls = lambda: CustomBoxHistoriElement(
                    line_edit_text, self.window, 
                    number_tab = 4, name_operation = "Replacement",
                    label_1 = "with", text_1 = window.getResult(4, 0),
                    label_2 = "on", text_2 = window.getResult(4, 1)
                )
            case _:
                lst_tabs: dict = {0: "Basic", 2: "Derivate", 3: "Integrate"}
                cls = lambda: BasicBoxHistoriElement(
                    line_edit_text, self.window,
                    result = window.result, name_operation = lst_tabs[tab]
                )
        self.window.add_global_histori.addLayout(cls())
        self.window.resize_global_histori.adjustSize()
        self.window.global_histori.verticalScrollBar().setValue(self.window.global_histori.verticalScrollBar().maximum())
        self.window.add_local_histori.addLayout(cls())
        self.window.resize_local_histori.adjustSize()
        scroll_histori = self.window.local_histori
        scroll_histori.verticalScrollBar().setValue(scroll_histori.verticalScrollBar().maximum())


    def button__O(self) -> None:
        if (line_edit_text := "".join(line_edit_text_list := self.line_edit_text.split("_O"))) != "":
            element_position = len(line_edit_text_list[0])-1
            print(element_position, "22")
            self.window.line_edit.setText(self.line_edit_text[:element_position] + self.line_edit_text[element_position+3:])
            
    def button_other(self) -> None:
        print(self.line_edit_text, 'line_edit_text')
        print(3)
        window = self.window
        result: Union[Calculate, Derivative, Integral, str] = window.result
        print(str(window.inputtin), "op")
        def integral():
            print(window.getLineEdit(1, 2))
            window.setResult(
                (1, 2),
                str(
                    Integral(
                        a = window.getResult(1, 0), 
                        b = window.getResult(1, 1), 
                        equation = window.getLineEdit(1, 2).text()
                    )
                )
            )
        def other_tab():
                window.result = str(Calculate(self.line_edit_text))
        match window.inputtin:
            case (1, 0) | (1, 1):
                other_tab()
                integral()
            case (1, 2): 
                integral()
            case (2, 0):
                window.result = str(Derivative(self.line_edit_text))
            case (3, 0):
                window.result = str(Derivative(self.line_edit_text, True))
            case (4, 0) | (4, 1):
                window.result = self.line_edit_text
            case (4, 2):
                window.result = str(
                    Calculate(
                        self.line_edit_text.replace(
                            window.getResult(4, 0), 
                            window.getResult(4, 1)
                        )
                    )
                )
            case _:
                other_tab()
        window.set_for_result.setText(window.result)
    
    @staticmethod
    def inputing_line_edit(button, window) -> None:
        label: str = button.text()
        line_edit = window.line_edit
        text: str = line_edit.text()
        position_cursor: int = line_edit.cursorPosition()
        line_edit.setText(text[:position_cursor] + label + text[position_cursor:])
        line_edit.setCursorPosition(position_cursor + len(label))


