from PyQt6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QLabel, QPushButton, QHBoxLayout
from PyQt6.QtCore import Qt, QPoint

class FramelessWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        

        self.setWindowFlags(Qt.WindowType.Window | Qt.WindowType.FramelessWindowHint)


        # Убираем рамки окна
        #self.setWindowFlags(Qt.WindowType.FramelessWindowHint)
        
        self.move(100, 600)
        self.show()
        # Настраиваем размеры окна
        self.resize(800, 600)
        self.setStyleSheet("background-color: lightblue;")

        # Центральный виджет
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        # Основной макет
        self.layout = QVBoxLayout(self.central_widget)
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

    def eventFilter(self, obj, event):
        """Обрабатывает события мыши для перетаскивания окна."""
        if obj == self.title_bar:
            #print(f"Event Type: {event.type()}")
            if event.type() == event.Type.MouseButtonPress:
                self.is_moving = True
                self.start_position = event.globalPosition().toPoint() - self.frameGeometry().topLeft()
                print(self.start_position)
                event.accept()
                return True

            elif event.type() == event.Type.MouseMove and self.is_moving:
                print("Перед move():", window.pos())
                self.move(event.globalPosition().toPoint() - self.start_position)
                self.show()
                print("После move():", window.pos())
                print(event.globalPosition().toPoint() - self.start_position)
                event.accept()
                return True

            elif event.type() == event.Type.MouseButtonRelease and event.button() == Qt.MouseButton.LeftButton:
                self.is_moving = False
                event.accept()
                return True

        return super().eventFilter(obj, event)

if __name__ == "__main__":
    import sys

    app = QApplication(sys.argv)
    window = FramelessWindow()
    window.show()
    sys.exit(app.exec())

