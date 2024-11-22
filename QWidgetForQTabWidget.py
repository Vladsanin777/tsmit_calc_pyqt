
class TabQWidget(QWidget):
    def __init__(self, tab):
        super().__init__()
        self.setLayout(tab)
