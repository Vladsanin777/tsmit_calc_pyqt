from PyQt6.QtWidgets import QApplication, QMainWindow, QWidget, QHBoxLayout, QPushButton, QVBoxLayout, QToolBar, QLabel, QMenu, QWidgetAction, QGridLayout
from PyQt6.QtGui import QPalette, QColor, QAction, QIcon
from PyQt6.QtCore import QFile, QTextStream, Qt
"""
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint)  # Убираем стандартную рамку окна
        self.setWindowTitle("Calculate")
        self.setGeometry(100, 100, 400, 800)

        # Загрузка CSS
        self.load_stylesheet("styles.css")
                self.setCentralWidget(CentralWidget(self))

        # Переменные для перетаскивания окна
        self.old_pos = None
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
    
    # Обработка событий для перетаскивания окна
    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.old_pos = event.globalPosition().toPoint()

    def mouseMoveEvent(self, event):
        if self.old_pos is not None:
            delta = event.globalPosition().toPoint() - self.old_pos
            self.move(self.x() + delta.x(), self.y() + delta.y())
            self.old_pos = event.globalPosition().toPoint()

    def mouseReleaseEvent(self, event):
        self.old_pos = None
"""

class LabelMenuButtonTitlebar(QLabel):
    def __init__(self, text):
        super().__init__(text)
        self.setAlignment(Qt.AlignCenter)


class ButtonTitleBar(QPushButton):
    def __init__(self, label, callback, css_class=None):
        super().__init__(label)
        self.clicked.connect(callback)
        if css_class:
            self.setObjectName(css_class)


class GridLocalHistoriTitleBar(QWidget):
    def __init__(self):
        super().__init__()
        layout = QGridLayout(self)

        layout.addWidget(ButtonTitleBar("Basic", self.button_settings_view_local_histori_basic), 0, 0)
        layout.addWidget(ButtonTitleBar("Tab 2", self.button_settings_view_local_histori_tab_2), 1, 0)
        layout.addWidget(ButtonTitleBar("Tab 3", self.button_settings_view_local_histori_tab_3), 2, 0)

    def button_settings_view_local_histori_basic(self):
        # Переключаем видимость box_local_histori_basic
        global box_local_histori_basic
        box_local_histori_basic.setVisible(not box_local_histori_basic.isVisible())

    def button_settings_view_local_histori_tab_2(self):
        pass

    def button_settings_view_local_histori_tab_3(self):
        pass


class PopoverLocalHistoriTitleBar(QMenu):
    def __init__(self, parent=None):
        super().__init__(parent)
        widget_action = QWidgetAction(self)
        widget = GridLocalHistoriTitleBar()
        widget_action.setDefaultWidget(widget)
        self.addAction(widget_action)


class MenuButtonLocalHistoriTitleBar(QPushButton):
    def __init__(self, label):
        super().__init__()
        self.setText(label)
        self.setMenu(PopoverLocalHistoriTitleBar())
        self.setObjectName("in_popover")


class GridMainTitleBar(QWidget):
    def __init__(self):
        super().__init__()
        layout = QGridLayout(self)

        layout.addWidget(ButtonTitleBar("general\nhistori", self.button_settings_view_general_histori), 0, 0)
        layout.addWidget(MenuButtonLocalHistoriTitleBar("local\nhistori"), 0, 1)

    def button_settings_view_general_histori(self):
        global scrolled_window_general_histori
        scrolled_window_general_histori.setVisible(not scrolled_window_general_histori.isVisible())


class PopoverMainTitleBar(QMenu):
    def __init__(self, parent=None):
        super().__init__(parent)
        widget_action = QWidgetAction(self)
        widget = GridMainTitleBar()
        widget_action.setDefaultWidget(widget)
        self.addAction(widget_action)


class MenuButtonMainTitleBar(QPushButton):
    def __init__(self, label):
        super().__init__()
        self.setText(label)
        self.setMenu(PopoverMainTitleBar())
        self.setObjectName("header_element")


class CustomTitleBar(QToolBar):
    def __init__(self, parent=None):
        super().__init__(parent)

        # Кнопка для смены языка
        self.addWidget(ButtonTitleBar("EN", self.on_language_clicked))

        # Кнопка для изменения цвета фона
        self.addWidget(ButtonTitleBar("Fon", self.change_background_color))

        # Меню "View"
        self.addWidget(MenuButtonMainTitleBar("View"))

    def on_language_clicked(self):
        pass

    def change_background_color(self):
        # Здесь ваш метод для изменения цвета
        pass

#######################
#Cretion Window


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Title Bar Example")

        # Создаем титул-бар и добавляем его
        self.title_bar = CustomTitleBar(self)
        self.addToolBar(Qt.ToolBarArea.TopToolBarArea, self.title_bar)

        # Основное содержимое окна
        main_widget = QWidget()
        main_layout = QVBoxLayout()
        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)
   def load_stylesheet(self, path):
        file = QFile(path)
        if file.open(QFile.OpenModeFlag.ReadOnly | QFile.OpenModeFlag.Text):
            stream = QTextStream(file)
            stylesheet = stream.readAll()
            self.setStyleSheet(stylesheet)
            file.close()
########################
#Cretion Application

class CalcApplication(QApplication):
    def __init__(self):
        super().__init__([])
        self.main_window = MainWindow()
        self.main_window.show()

# Запуск приложения
app = CalcApplication()
app.exec()
