from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, 
    QWidget, QLabel, QPushButton, 
    QHBoxLayout, QMenu, QWidgetAction, 
    QScrollArea, QGridLayout, QTabWidget, 
    QLineEdit
)
from PyQt6.QtCore import Qt, QMimeData
from PyQt6.QtGui import (
    QColor, QPalette, QBrush, 
    QLinearGradient, QDrag
)
import random, asyncio

from decimal import Decimal

class BaseCalc:
    # Удаление нулей в конце числа после запятой (наследие с C++)
    @staticmethod
    async def removing_zeros(expression: str) -> str:
        if '.' in expression and not 'E' in expression:
            while expression[-1] == '0': expression = expression[:-1]
            if expression and expression[-1] == '.': expression = expression[:-1]
        return expression if expression else '0'


class CalculateMain:
    expression: str
    def __init__(self, expression):
        self.expression = expression.replace(" ", "")

    async def _find_nth_occurrence(self, string: str, substring: str, n: int) -> int:
        start = 0
        for _ in range(n):
            start = string.find(substring, start)
            if start == -1:
                return -1  # Если подстрока не найдена
            start += len(substring)
        return start - len(substring)
    # Метод для поиска приоритетных скобок
    async def _searching_for_priority_brackets(self, expression: str, number_of_bracket: int) -> list[int]:
        return [(number_last_open_brackets := await self._find_nth_occurrence(expression, '(', number_of_bracket)), expression.find(')', number_last_open_brackets)]
    async def _float(self, number: str) -> Decimal:
        match number[:2] :
            case "0x":
                return Decimal(float.fromhex(number))
            case "0b":
                number = number[2:]
                # Разделяем целую и дробную части
                integer_part, fractional_part = (number.split('.') if '.' in number else (number, ''))

                # Преобразуем целую часть
                integer_value = int(integer_part, base = 2) if integer_part else 0

                # Преобразуем дробную часть
                fractional_value = sum(int(bit) * 2**(-i) for i, bit in enumerate(fractional_part, start=1))

                # Итоговое значение
                return Decimal(integer_value) + Decimal(fractional_value)
            case "0t":    
                number = number[2:]
                integer_part, fractional_part = (number.split('.') if '.' in number else (number, ''))

                # Преобразуем целую часть
                integer_value = int(integer_part, 8) if integer_part else 0

                # Преобразуем дробную часть
                fractional_value = sum(int(digit) * 8**(-i) for i, digit in enumerate(fractional_part, start=1))

                # Итоговое значение
                return Decimal(integer_value) + Decimal(fractional_value)
            case _:
                return Decimal(number)
    
    # Main method for calculate
    async def _calculate_expression_base(self, tokens: list[str]) -> str:
        print(tokens, 73)
        priority_operator_index = 0
        while len(tokens) != 1:
            if '^' in tokens:
                priority_operator_index = tokens.index('^')
                b = await self._float(tokens.pop(priority_operator_index+1))
                tokens.pop(priority_operator_index)
                a = await self._float(tokens[priority_operator_index-1])
                tokens[priority_operator_index-1] = str(a**b)
            if '*' in tokens:
                priority_operator_index = tokens.index('*')
                b = await self._float(tokens.pop(priority_operator_index+1))
                tokens.pop(priority_operator_index)
                a = await self._float(tokens[priority_operator_index-1])
                tokens[priority_operator_index-1] = str(a*b)
            elif ':' in tokens:
                priority_operator_index = tokens.index(':')
                b = await self._float(tokens.pop(priority_operator_index+1))
                tokens.pop(priority_operator_index)
                a = await self._float(tokens[priority_operator_index-1])
                tokens[priority_operator_index-1] = str(a/b)
            elif '/' in tokens:
                priority_operator_index = tokens.index('/')
                b = await self._float(tokens.pop(priority_operator_index+1))
                tokens.pop(priority_operator_index)
                a = await self._float(tokens[priority_operator_index-1])
                tokens[priority_operator_index-1] = str(a/b)
            elif '-' in tokens:
                priority_operator_index = tokens.index('-')
                b = await self._float(tokens.pop(priority_operator_index+1))
                tokens.pop(priority_operator_index)
                a = await self._float(tokens[priority_operator_index-1])
                tokens[priority_operator_index-1] = str(a-b)
            elif '+' in tokens:
                priority_operator_index = tokens.index('+')
                b = await self._float(tokens.pop(priority_operator_index+1))
                tokens.pop(priority_operator_index)
                a = await self._float(tokens[priority_operator_index-1])
                tokens[priority_operator_index-1] = str(a+b)
            elif '|' in tokens:
                print(tokens)
                tokens[0] = "".join([tokens[0], tokens[1], tokens[2] if len(tokens) == 3 else str(e)])
                while len(tokens) > 1:
                    tokens.pop()
        else:
            tokens[0] = str(await self._float(tokens[0]))
        print(tokens[0], 35) 
        return tokens[0]
    # Calculate persent and factorial
    async def _calculate_expression_list(self, tokens: list[str]) -> str:
        print(tokens)
        t: int
        while '%' in tokens:
            t = tokens.index('%')
            print(tokens)
            print(t)
            tokens.pop(t)
            print(tokens[t-1])
            tokens[t-1] = str(await self._float(tokens[t-1])/ await self._float(str(100)))
        while '!' in tokens:
            t = tokens.index('!')
            tokens.pop(t)
            tokens[t-1] = str(factorial(int(await self._float(tokens[t-1]))))
        while 'n' in tokens:
            t = tokens.index('n')
            tokens.pop(t)
            tokens[t] = str(log(int(await self._float(tokens[t]))))
            print(tokens, 67)
        while 'g' in tokens:
            t = tokens.index('g')
            tokens.pop(t)
            print(tokens[t], 45)
            tokens[t] = str(log10(int(await self._float(tokens[t]))))
        while 's' in tokens:
            t = tokens.index('s')
            tokens.pop(t)
            tokens[t] = str(sqrt(int(await self._float(tokens[t]))))
        while 'l' in tokens:
            t = tokens.index('l')
            tokens.pop(t)
            if len(tokens) >= t+2:
                print(True)
                if tokens[t+1] == '|':
                    tokens.pop(t+1)
                    tokens[t] = str(log(float(await self._float(tokens[t])), float(await self._float(tokens.pop(t+1) if len(tokens) >= t+2 else str(e)))))
                else:
                    tokens[t] = str(log(float(await self._float(tokens[t]))))

            else:
                print(False)
                tokens[t] = str(log(float(await self._float(tokens[t]))))
            print(890)

        return await self._calculate_expression_base(tokens)
    async def debuger(self):
        self.expression = self.expression.replace("sqrt", "s")
        self.expression = self.expression.replace("ln", "n")
        self.expression = self.expression.replace("log", "l")
        self.expression = self.expression.replace("lg", "g")
        self.expression = self.expression.replace("**", "^")
        replay: bool = True
        while replay:
            replay = False
            match self.expression[-1]:
                case "*" | "/" | ":" | "+" | "-" | "^" | "l" | "m" | "n" | "g" | "s":
                    self.expression = self.expression[:-1]
                    replay = True

        while self.expression.count("(") > self.expression.count(")"):
            self.expression += ")"
    # Разбиение строки на отдельные элементы (числа и операторы)
    async def _tokenize(self, expression: str) -> str:
        # sheach first digit
        positions = [element_i for element_i in range(len(expression)) if "%!-+*/:^sngol|".find(expression[element_i]) != -1]
        print(positions, 78)
        result_list = list()
        while positions != []:
            print(positions, 77)
            # wrating first digit in result list
            if (element := expression[(number_operators := positions.pop())+1:]) != "": result_list.append(element)
            # wrating operator in result list
            result_list.append(expression[number_operators])
            print(result_list, 76)
            # trimming string expression
            expression = expression[:number_operators]
            print(expression, 75)
        if result_list == []:
            print(89)
            if expression: result_list = [expression]
        else:
            if expression: result_list.append(expression)
            result_list = result_list[::-1]
        print(result_list, 34)
        return await self._calculate_expression_list(result_list)



    # Основная функция подсчёта
    async def calc(self) -> str:
        if not self.expression: return "0"
        try:
            await self.debuger()
            
            expression_1 = self.expression
            while (count_brackets := expression_1.count("(")) != 0:
                print("1")
                priority_brackets = await self._searching_for_priority_brackets(
                    expression_1, 
                    count_brackets
                )
                inner_expression = expression_1[priority_brackets[0] + 1:priority_brackets[1]]
                expression_1 = (
                    expression_1[:priority_brackets[0]] +
                    await self._tokenize(inner_expression) +
                    expression_1[priority_brackets[1] + 1:]
                )
            return await BaseCalc.removing_zeros(str(await self._tokenize(expression_1)))
        except:
            return "Error"


colors_background = ["#99FF18", "#FFF818", "#FFA918", "#FF6618", "#FF2018", "#FF1493", "#FF18C9", "#CB18FF"]

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
        global add_general_histori, add_local_histori_basic, result_basic_calc, line_edit_calc_basic
        if (line_edit_text := "".join(line_edit_text_list := self.line_edit_text.split("_DO"))) != "":
            add_general_histori.addLayout(BoxHistoriElement(line_edit_text, str(result_basic_calc)))
            add_local_histori_basic.addLayout(BoxHistoriElement(line_edit_text, str(result_basic_calc)))
        line_edit_calc_basic.setText(line_edit_text_list[1])

    def button__POST(self):
        global add_general_histori, add_local_histori_basic, result_basic_calc, line_edit_calc_basic
        if (line_edit_text := "".join(line_edit_text_list := line_edit_calc_basic.text().split("_POST"))) != "":
            add_general_histori.addLayout(BoxHiskoriElement(line_edit_text, str(result_basic_calc)))
            add_local_histori_basic.addLayout(BoxHistoriElement(line_edit_text, str(result_basic_calc)))
        line_edit_calc_basic.setText(line_edit_text_list[0])

    @staticmethod
    def inputing_line_edit(button: QPushButton) -> None:
        print(button.text(), 56)
        global line_edit_calc_basic
        label: str = button.text()
        text: str = line_edit_calc_basic.text()
        position_cursor: int = line_edit_calc_basic.cursorPosition()
        line_edit_calc_basic.setText(text[:position_cursor] + label + text[position_cursor:])
        line_edit_calc_basic.setCursorPosition(position_cursor + len(label))

    def button_result(self):
        global add_general_histori, add_local_histori_basic, result_basic_calc, line_edit_calc_basic
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
        result_basic_calc = asyncio.run(CalculateMain(self.entry_text).calc())
        set_for_result_basic_calc.setText(result_basic_calc)

#Global and Local Histori
class HistoriVBoxLayout(QVBoxLayout):
    def __init__(self):
        super().__init__()
        self.setSpacing(0)
        self.setContentsMargins(0, 0, 0, 0)

class HistoriScroll(QScrollArea):
    _add_histori: QVBoxLayout
    def __init__(self):
        super().__init__()
        self._add_histori = HistoriVBoxLayout()
        self.setLayout(self._add_histori)
    def getAddHistori(self):
        return self._add_histori


add_global_hitori: QVBoxLayout
global_histori: HistoriScroll
add_local_histori_basic: QVBoxLayout
local_histori_basic: HistoriScroll
line_edit_calc_basic: QLineEdit
set_for_result_basic_calc: QPushButton
result_basic_calc: str = "0"

class LabelHistori(QLabel):
    callback: str
    def __init__(self, label: str, css_name: str, *, custom_callback: str = None):
        super().__init__(label)
        self.setSizePolicy(self.sizePolicy().Policy.Expanding, self.sizePolicy().Policy.Expanding)
        self.setContentsMargins(0, 0, 0, 0)
        self.setObjectName(css_name)
        if custom_callback:
            self.callback = custom_callback
        else:
            self.callback = label

    def mousePressEvent(self, event):
        drag = QDrag(self)
        mime_data = QMimeData()
        mime_data.setText(self.callback)
        drag.setMimeData(mime_data)
        drag.exec(Qt.DropAction.MoveAction)

class BoxHistoriElement(QHBoxLayout):
    def __init__(self, expression: str, result: str):
        super().__init__()
        self.setSpacing(0)
        self.setContentsMargins(0, 0, 0, 0)
        self.addWidget(LabelHistori(expression, "keybord"))
        self.addWidget(LabelHistori("=", "keybord", custom_callback = expression + "=" + result))
        self.addWidget(LabelHistori(result, "keybord"))


class ButtonDrag(QPushButton):
    def __init__(self, label: str, *, css_name = "keybord", callback = None):
        super().__init__(label)
        self.setSizePolicy(self.sizePolicy().Policy.Expanding, self.sizePolicy().Policy.Expanding)
        self.setObjectName(css_name)
        if not callback: callback = LogicCalculateBasic.inputing_line_edit
        self.clicked.connect(callback)

    def mousePressEvent(self, event):
        drag = QDrag(self)
        mime_data = QMimeData()
        mime_data.setText(self.text())
        drag.setMimeData(mime_data)
        drag.exec(Qt.DropAction.MoveAction)

class ButtonDragAndDrop(ButtonDrag):
    def __init__(self, label, *, css_name = "keybord", callback = None):
        super().__init__(label, css_name = css_name, callback = callback)

    def dragEnterEvent(self, event):
        if event.mimeData().hasText():
            event.acceptProposedAction()

    def dropEvent(self, event):
        self.setText(event.mimeData().text())
        event.acceptProposedAction()


class LineEditCalculateBasic(QLineEdit):
    def __init__(self):
        super().__init__()
        self.setSizePolicy(self.sizePolicy().Policy.Expanding, self.sizePolicy().Policy.Expanding)
        self.setObjectName("keybord")
        self.textChanged.connect(self.on_entry_changed)
    def on_entry_changed(self, text_entry):
        if text_entry:
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
            else:
                logic_calc_basic.button_other()

class BuildingButtonInGridLayout():
    def __init__(self, list_button: list[list[str]], grid: QGridLayout, row: int = 0):
        for row_labels_for_button in list_button:
            column: int = 0
            for one_button in row_labels_for_button:
                grid.addWidget(ButtonDrag(one_button), row, column, 1, 1)
                column += 1
            row += 1
#Basic calculate
class BasicCalculateGridLayout(QGridLayout):
    def __init__(self):
        super().__init__()
        self.setSpacing(0)
        self.setContentsMargins(0, 0, 0, 0)
        global local_histori_basic, add_local_histori_basic
        self.addWidget(local_histori_basic := HistoriScroll(), 0, 0, 3, 5)
        add_local_histori_basic = local_histori_basic.getAddHistori()
        self.button_for_calc_basic("_ALL", 4, 0)
        global line_edit_calc_basic
        self.addWidget(line_edit_calc_basic := LineEditCalculateBasic(), 4, 1, 1, 3)
        self.button_for_calc_basic("_O", 4, 4)
        BuildingButtonInGridLayout([
            ["()", "(", ")", "mod", "_PI"], 
            ["7", "8", "9", ":", "sqrt"], 
            ["4", "5", "6", "*", "^"], 
            ["1", "2", "3", "-", "!"], 
            ["0", ".", "%", "+", "_E"]
        ], self, 5)
        global set_for_result_basic_calc, result_basic_calc
        self.addWidget(set_for_result_basic_calc := ButtonDrag(result_basic_calc), 10, 0, 1, 2) 
        self.button_for_calc_basic("_DO", 10, 2)
        self.button_for_calc_basic("_POST", 10, 3)
        self.button_for_calc_basic("=", 10, 4)


    def button_for_calc_basic(self, label: str, row: int, column: int) -> None:
        self.addWidget(ButtonDrag(label, css_name = "keybord"), row, column, 1, 1)

#Main TabWidget
class TabQWidget(QWidget):
    def __init__(self, tab):
        super().__init__()
        self.setLayout(tab)

class MainTabWidget(QTabWidget):
    def __init__(self):
        super().__init__()
        self.tabBar().setExpanding(True)
        self.addTab(TabQWidget(BasicCalculateGridLayout()), "Basic")
        self.addTab(TabQWidget(QGridLayout()), "Tab 2")
        self.addTab(TabQWidget(QGridLayout()), "Tab 3")
        self.addTab(TabQWidget(QGridLayout()), "Tab 4")



# Title Bar
class TitleButton(QPushButton):
    def __init__(self, label, *, callback=None, menu=None):
        super().__init__(label)
        self.setSizePolicy(self.sizePolicy().Policy.Expanding, self.sizePolicy().Policy.Expanding)
        if callback:
            self.clicked.connect(callback)
        if menu:
            self.setMenu(menu)
        self.setObjectName("title-button")

class TitleWidgetAction(QWidgetAction):
    def __init__(self, parent, button):
        super().__init__(parent)
        self.setDefaultWidget(button)

class TitleMenu(QMenu):
    def __init__(self, buttons):
        super().__init__()
        for button in buttons:
            self.addAction(TitleWidgetAction(self, button))

class TitleLayout(QHBoxLayout):
    def __init__(self):
        super().__init__()
        self.setContentsMargins(0, 0, 0, 0)
        self.setSpacing(0)
        self.addWidget(TitleButton("EN", callback=self.language_callback))
        self.addWidget(TitleButton("Fon", callback=app.change_fon))
        self.addWidget(TitleButton(
            "View",
            menu=TitleMenu([
                TitleButton("Global History", callback=self.global_histori_callback),
                TitleButton("Local History", menu=TitleMenu([
                    TitleButton("Basic", callback=self.local_histori_basic_callback),
                    TitleButton("Tab 2"),
                    TitleButton("Tab 3"),
                    TitleButton("Tab 4")
                ]))
            ])
        ))

    def language_callback(self):
        print("Language button clicked")

    def global_histori_callback(self):
        global global_hislori
        global_histori.setVisible(not global_histori.isVisible())

    def local_histori_basic_callback(self):
        print("Local history basic button clicked")

class TitleBar(QWidget):
    def __init__(self):
        super().__init__()
        self.setFixedHeight(30)
        self.setLayout(TitleLayout())
# Main Content
class MainLayout(QVBoxLayout):
    def __init__(self):
        super().__init__()
        self.setContentsMargins(0, 0, 0, 0)
        self.setSpacing(0)
        self.addWidget(TitleBar())
        global global_histori, add_global_histori
        self.addWidget(global_histori := HistoriScroll())
        add_global_histori = global_histori.getAddHistori()
        self.addWidget(MainTabWidget())
        
# Window
class Window(QWidget):
    def __init__(self):
        super().__init__()
        self.setLayout(MainLayout())
        self.setWindowTitle("Calculate")
        self.resize(400, 800)
        self.setObjectName("window")
        app.change_fon()
        self.show()

class GradientWindow(QLinearGradient):
    def __init__(self):
        super().__init__(0, 0, 1, 1)
        random_color_1 = random.choice(colors_background)
        random_color_2 = random.choice(colors_background)
        while random_color_1 == random_color_2:
            random_color_2 = random.choice(colors_background)
        self.setCoordinateMode(QLinearGradient.CoordinateMode.ObjectBoundingMode)
        self.setColorAt(0, QColor(random_color_1))
        self.setColorAt(1, QColor(random_color_2))

class PaletteWindow(QPalette):
    def __init__(self):
        super().__init__()
        self.setBrush(QPalette.ColorRole.Window, QBrush(GradientWindow()))

class Application(QApplication):
    def __init__(self):
        super().__init__([])
        self.setStyleSheet("""
            QPushButton#title-button {
                background-color: rgba(0, 0, 0, 0.3);
                color: white;
                border: none;
            }
            QPushButton#title-button:hover {
                background-color: rgba(0, 0, 0, 0.6);
            }
            QScrollArea{
                border: none;
                background: transparent;
            }
            QTabBar::tab {
                background: rgba(0, 0, 0, 0.3);
                border: none;
                padding: 5px auto;
                color: rgb(255, 255, 255);
            }
            QTabBar::tab:selected, QTabBar::tab:hover {
                background: transparent;
                color: rgb(0, 0, 0);
            }
            QTabWidget::pane {
                border: none;
                background: transparent;
            }
            QTabBar QToolButton {
                border: none;
                background: rgba(0, 0, 0, 0.3);
                color: rgb(0, 0, 0); 
            }
            #keybord {
                margin: 0px;
                border: none;
                font-size: 30px;
                background: rgba(0, 0, 0, 0.3);
                color: rgb(255, 255, 255);
            }
            #keybord:hover {
                background: transparent;
                color: rgb(0, 0, 0);
            }
        """)

    def change_fon(self):
        self.setPalette(PaletteWindow())

app = Application()
window = Window()
app.exec()

