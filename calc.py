from PyQt6.QtWidgets import QApplication, QMainWindow, QWidget, QHBoxLayout, QPushButton, QVBoxLayout, QToolBar
from PyQt6.QtGui import QPalette, QColor
from PyQt6.QtCore import QFile, QTextStream

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Calculate")
        self.setGeometry(100, 100, 400, 800)

        # Загрузка CSS
        self.load_stylesheet("styles.css")

        # Устанавливаем настраиваемую панель заголовка
        self.addToolBar(CustomTitleBar(self))

        # Примените цветовую схему
        self.set_palette()

    def load_stylesheet(self, path):
        file = QFile(path)
        if file.open(QFile.OpenModeFlag.ReadOnly | QFile.OpenModeFlag.Text):
            stream = QTextStream(file)
            stylesheet = stream.readAll()
            self.setStyleSheet(stylesheet)
            file.close()

    def set_palette(self):
        palette = QPalette()
        palette.setColor(QPalette.ColorRole.Window, QColor("#f0f0f0"))
        palette.setColor(QPalette.ColorRole.WindowText, QColor("#000000"))
        self.setPalette(palette)


class CustomTitleBar(QToolBar):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMovable(False)

        # Кнопка для смены языка
        lang_button = QPushButton("EN")
        lang_button.clicked.connect(self.on_language_clicked)
        self.addWidget(lang_button)

        # Кнопка для изменения цвета фона
        fon_button = QPushButton("Fon")
        fon_button.clicked.connect(self.on_fon_clicked)
        self.addWidget(fon_button)

        """# Меню
        view_menu = QAction("View", self)
        self.addAction(view_menu)
        """
    def on_language_clicked(self):
        # Логика смены языка
        pass

    def on_fon_clicked(self):
        # Логика изменения цвета фона
        pass


class CalcApplication(QApplication):
    def __init__(self):
        super().__init__([])
        self.main_window = MainWindow()
        self.main_window.show()

# Запуск приложения
app = CalcApplication()
app.exec()
