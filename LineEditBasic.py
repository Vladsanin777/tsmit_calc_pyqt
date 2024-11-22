from PyQt6.QtWidget import QLineEdit
class LineEditCalculateBasic(QLineEdit):
    def __init__(self):
        super().__init__()
        self.setSizePolicy(self.sizePolicy().Policy.Expanding, self.sizePolicy().Policy.Expanding)
        self.setObjectName("keybord")
        self.textChanged.connect(self.on_entry_changed)
    def on_entry_changed(self, text_entry):
        logic_calc_basic = LogicCalculateBasic(text_entry)
        if "_ALL" in text_entry:
            logic_calc_basic.button__ALL()
        elif "_DO" in text_entry:
            logic_calc_basic.button__DO()
        elif "_POST" in text_entry:
            logic_calc_basic.button__POST()
        elif "_O" in text_entry:
            logic_calc_basic.button__O()
        elif "=" in text_entry:
            logic_calc_basic.button_result()
        elif "_RES" in text_entry:
            logic_calc_basic.button_result()
        else:
            logic_calc_basic.button_other()
