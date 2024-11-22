from Calculate import Calculate 
class LogicCalculateBasic():
    entry_text: str = ""

    def __init__(self, line_edit_text: str):
        self.line_edit_text = line_edit_text

    def button__ALL(self):
        global add_global_histori, add_local_histori_basic, result_basic_calc, line_edit_calc_basic
        if (line_edit_text := "".join(self.line_edit_text.split("_ALL"))) != "":
            add_global_histori.addLayout(BoxHistoriElement(line_edit_text, str(result_basic_calc)))
            add_local_histori_basic.addLayout(BoxHistoriElement(line_edit_text, str(result_basic_calc)))
            result_basic_calc = "0"
            set_for_result_basic_calc.setText(result_basic_calc)
        line_edit_calc_basic.setText("")

        
    def button__DO(self):
        global add_global_histori, add_local_histori_basic, result_basic_calc, line_edit_calc_basic
        if (line_edit_text := "".join(line_edit_text_list := self.line_edit_text.split("_DO"))) != "":
            add_global_histori.addLayout(BoxHistoriElement(line_edit_text, str(result_basic_calc)))
            add_local_histori_basic.addLayout(BoxHistoriElement(line_edit_text, str(result_basic_calc)))
        line_edit_calc_basic.setText(line_edit_text_list[1])

    def button__POST(self):
        global add_global_histori, add_local_histori_basic, result_basic_calc, line_edit_calc_basic
        if (line_edit_text := "".join(line_edit_text_list := line_edit_calc_basic.text().split("_POST"))) != "":
            add_global_histori.addLayout(BoxHiskoriElement(line_edit_text, str(result_basic_calc)))
            add_local_histori_basic.addLayout(BoxHistoriElement(line_edit_text, str(result_basic_calc)))
        line_edit_calc_basic.setText(line_edit_text_list[0])

    def button_result(self):
        global add_global_histori, add_local_histori_basic, result_basic_calc, line_edit_calc_basic
        if (line_edit_text := "".join(line_edit_text_list := self.line_edit_text.split("="))) != "":
            add_global_histori.addLayout(BoxHistoriElement(line_edit_text, str(result_basic_calc)))
            add_local_histori_basic.addLayout(BoxHistoriElement(line_edit_text, str(result_basic_calc)))
        line_edit_calc_basic.setText(result_basic_calc)
        line_edit_calc_basic.setCursorPosition(len(result_basic_calc)-1)


    def button__O(self) -> None:
        global line_edit_calc_basic
        if (line_edit_text := "".join(line_edit_text_list := self.line_edit_text.split("_O"))) != "":
            element_position = len(line_edit_text_list[0])-1
            print(element_position, "22")
            line_edit_calc_basic.setText(self.line_edit_text[:element_position] + self.line_edit_text[element_position+3:])
            
    def button_other(self) -> None:
        global set_for_result_basic_calc, result_basic_calc
        result_basic_calc = asyncio.run(CalculateMain(self.line_edit_text).calc())
        print(result_basic_calc)
        set_for_result_basic_calc.setText(result_basic_calc)
    
    @staticmethod
    def inputing_line_edit(button) -> None:
        global line_edit_calc_basic
        label: str = button.text()
        text: str = line_edit_calc_basic.text()
        position_cursor: int = line_edit_calc_basic.cursorPosition()
        line_edit_calc_basic.setText(text[:position_cursor] + label + text[position_cursor:])
        line_edit_calc_basic.setCursorPosition(position_cursor + len(label))


