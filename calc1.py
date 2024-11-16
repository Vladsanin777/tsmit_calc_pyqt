from PyQt6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QLabel, QPushButton, QHBoxLayout
from PyQt6.QtCore import Qt, QPoint
import os, subprocess

class FramelessWindow(QWidget):
    def __init__(self):
        super().__init__()
        

        # Убираем рамку, чтобы окно можно было перемещать по всей области
        #self.setWindowFlags(Qt.WindowType.FramelessWindowHint)

        # Храним начальную позицию мыши при захвате
        self._drag_pos = None

        # Убираем рамки окна
        #self.setWindowFlags(Qt.WindowType.FramelessWindowHint)
        
        self.move(100, 600)
        self.show()
        # Настраиваем размеры окна
        self.resize(800, 600)
        self.setStyleSheet("background-color: lightblue;")
        """
        # Центральный виджет
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        """
        # Основной макет
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        
        # Титульная панель
        self.title_bar = self.create_title_bar()
        self.layout.addWidget(self.title_bar)

        # Контентная область
        content = QLabel("Контент окна", self)
        content.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.layout.addWidget(content)

        # Переменные для перемещения окна
        self.is_moving = True
        self.start_position = QPoint()

    def create_title_bar(self):
        """Создаёт кастомный заголовок окна."""
        title_bar = QWidget()
        title_bar.setStyleSheet("background-color: darkblue; height: 30px;")
        title_bar.setFixedHeight(30)

        layout = QHBoxLayout(title_bar)
        layout.setContentsMargins(5, 0, 5, 0)

        # Заголовок
        title = QLabel("Кастомное окно")
        title.setStyleSheet("color: white;")
        layout.addWidget(title)

        # Кнопки управления
        minimize_button = QPushButton("-")
        minimize_button.setFixedSize(20, 20)
        minimize_button.clicked.connect(self.showMinimized)

        close_button = QPushButton("x")
        close_button.setFixedSize(20, 20)
        close_button.clicked.connect(self.close)

        layout.addWidget(minimize_button)
        layout.addWidget(close_button)

        # Пространство для выравнивания заголовка слева
        layout.insertStretch(1)

        # Устанавливаем фильтр событий для заголовка
        title_bar.installEventFilter(self)

        return title_bar


if __name__ == "__main__":
    import sys

    app = QApplication(sys.argv)
    window = FramelessWindow()
    window.show()
    sys.exit(app.exec())

