from PyQt6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QLabel, QPushButton, QHBoxLayout
from PyQt6.QtCore import Qt, QPoint
import os, subprocess

class TitleButton(QPushButton):
    def __init__(self, label, callback):
        super().__init__(label)
        self.clicked.connect(callback)
        self.setStyleSheet("""
        QPushButton {
            background-color: rgba(0, 0, 0, 0); /* Прозрачный фон */
            border: none; /* Убираем рамку */
        }
        QPushButton:hover {
            background-color: rgba(0, 0, 0, 0.3); /* Лёгкий прозрачный эффект при наведении */
        }
        """)

class TitleLayout(QHBoxLayout):
    def __init__(self, parent = None):
        super().__init__(parent)
        self.setContentsMargins(0, 0, 0, 0)
        self.addWidget(TitleButton("EN", self.language))
        self.addWidget(TitleButton("Fon", self.change_fon))
    def language(self):
        pass
    def change_fon(self):
        pass

class TitleBar(QWidget):
    def __init__(self):
        super().__init__()
        self.setFixedHeight(30)
        self.setLayout(TitleLayout())
        self.setObjectName("title-bar")
        self.setStyleSheet("""
        QWidget#title-bar { 
            background-color: rgba(0, 0, 0, 0.9);
        }
        """)
        self.setAutoFillBackground(True)
        print(self.styleSheet())

class MainLayout(QVBoxLayout):
    def __init__(self):
        super().__init__()
        self.setContentsMargins(0, 0, 0, 0)
        self.setSpacing(0)  # Убираем зазоры между виджетами
        self.addWidget(TitleBar())
        content = QLabel("Основной контент")
        content.setStyleSheet("background: lightgray;")
        self.addWidget(content)
class Window(QWidget):
    def __init__(self):
        super().__init__()
        self.setLayout(MainLayout())

        # Настраиваем размеры окна
        self.resize(800, 600)
        self.setStyleSheet("background-color: blue;")
        

if __name__ == "__main__":
    import sys

    app = QApplication(sys.argv)
    window = Window()
    window.show()
    sys.exit(app.exec())

