from math import *
from decimal import Decimal
from re import sub, findall
import re
import traceback
import threading
from functools import wraps
from typing import Self, Any, List, Union


def threaded_class(cls):
    """Декоратор для выполнения инициализации класса в отдельном потоке."""
    class ThreadedClass:
        def __init__(self: Self, *args, **kwargs):
            self._init_done = threading.Event()
            self._thread = threading.Thread(target=self._initialize, args=(cls, args, kwargs), daemon=True)
            self._thread.start()
            self._init_done.wait()

        def _initialize(self: Self, cls, args, kwargs):
            self._instance = cls(*args, **kwargs)
            self._init_done.set()

        def __getattr__(self, item):
            return getattr(self._instance, item)

        def join(self):
            """Ожидание завершения потока."""
            self._thread.join()
        def __str__(self):
            return str(self._instance)
        def __iter__(self):
            return iter(self._instance)
    return ThreadedClass

class Debuger:
        """
        Класс для обработки математических выражений: очистка, замена операторов и проверка скобок.
        """
        def __init__(self, expression: str):

            expression = expression.replace("_E", str(e)).replace("_PI", str(pi))
            expression = expression.replace("**", "^").replace(":", "/").replace(",", ".").replace("//", "/").replace("--", "+")
            while "++" in expression:
                expression = expression.replace("++", "+")
            while "//" in expression:
                expression = expression.replace("//", "/")
            while "^^" in expression:
                expression = expression.replace("^^", "^")
            expression = expression.replace("^*", "^")
            # Удаление символов в конце строки
            expression = sub(r'[*/:+\-\^()]+$', '', expression)
            self.expression = expression
            # Добавление недостающих закрывающих скобок
            self.expression += ")" * self.verification_brackets()

        def verification_brackets(self) -> int:
            """
            Проверяет баланс скобок в выражении. Возвращает количество недостающих закрывающих скобок.
            Выбрасывает BracketsError, если баланс нарушен (лишние закрывающие скобки).
            """
            bracket = 0
            for symbol in self.expression:
                if symbol == "(":
                    bracket += 1
                elif symbol == ")":
                    bracket -= 1
                if bracket < 0:
                    raise self.BracketsError("verification_brackets")
            return bracket

        def __str__(self) -> str:
            """
            Возвращает обработанное выражение.
            """
            return self.expression

        class BracketsError(Exception): ...

class SimpleExpression:
    def __init__(self, expression: str):
        """
        Разбирает строку математического выражения на список токенов (чисел и операторов).
        :param expression: Строка математического выражения
        """
        # Разбиваем выражение на токены
        tokens = self.tokenize_expression(expression)
        print("Tokens:", tokens)  # Для отладки

        # Создаём дерево выражения
        self.result = self._build_expression_tree(tokens)
        if isinstance(self.result, str):
            self.result = [self.result]
        print(self.result, "result")
    def __iter__(self):
        return iter(self.result)



    def tokenize_expression(self: Self, expression: str) -> List[str]:
        """
        Разбивает строку выражения на токены, включая числа, операторы и функции.
        Если после операторов /, *, ^, +, -, mod, | или перед ним ( идёт знак '-',
        он объединяется с числом. Также учитывается, если выражение начинается с '-'.
        """
        # Регулярное выражение для чисел, операторов и функций
        token_pattern = re.compile(r'(\d+\.\d+|\d+|[a-zA-Z]+|[-+*/^|()=]|x|mod|-\d+(?:\.\d+)?)')

        # Используем регулярное выражение для разбиения строки
        tokens = token_pattern.findall(expression)

        # Преобразуем токены в строковый список
        return tokens

    def _build_expression_tree(self, tokens: List[str]) -> Union[str, List]:
        """
        Рекурсивно строит дерево выражения из токенов.
        """
        def parse_expression(tokens: List[str]) -> List:
            stack = []  # Стек для хранения выражений
            operators = []  # Стек операторов

            def apply_operator():
                # Применяет оператор к верхним элементам стека
                if len(stack) < 2:
                    if operators[-1] == '-':
                        operators.pop()
                        def add_minus(x):
                            if isinstance(x, list):
                                x[0] = add_minus(x[0])
                                return x
                            else:
                                return '-' + x
                        stack.append(add_minus(stack.pop()))
                        
                    else:
                        raise ValueError(f"Недостаточно операндов для оператора '{operators[-1]}'. Текущий стек: {stack}")
                else:
                    operator = operators.pop()
                    right = stack.pop()
                    left = stack.pop()
                    stack.append([left, operator, right])

            precedence = {'|': 0, '+': 1, '-': 1, '*': 2, '/': 2, '^': 3}
            i = 0
            while i < len(tokens):
                token = tokens[i]
                if re.match(r'^\d+(\.\d+)?$', token) or token == 'x':
                    # Число или переменная
                    stack.append(token)
                elif token in ['sin', 'cos', 'tan', 'cot', 'log', 'ln', 'lg', 'arcsin', 'arccos', 'arctan', 'arccot', 'arcsec', 'arccsc', 'ln', 'log', 'lg', 'sgn', 'abs', 'mod']:
                    # Функция
                    if i + 1 < len(tokens) and tokens[i + 1] == '(':
                        # Найти соответствующую закрывающую скобку
                        depth = 1
                        j = i + 2
                        while j < len(tokens):
                            if tokens[j] == '(':
                                depth += 1
                            elif tokens[j] == ')':
                                depth -= 1
                            if depth == 0:
                                break
                            j += 1
                        if depth != 0:
                            raise ValueError("Несоответствие скобок в выражении.")
                        # Рекурсивно разобрать подвыражение внутри скобок
                        function_argument = parse_expression(tokens[i + 2:j])
                        stack.append([token, function_argument])
                        i = j  # Перейти к следующему токену после закрывающей скобки
                    else:
                        raise ValueError(f"У функции '{token}' отсутствует аргумент.")
                elif token == '(':
                    # Найти соответствующую закрывающую скобку
                    depth = 1
                    j = i + 1
                    while j < len(tokens):
                        if tokens[j] == '(':
                            depth += 1
                        elif tokens[j] == ')':
                            depth -= 1
                        if depth == 0:
                            break
                        j += 1
                    if depth != 0:
                        raise ValueError("Несоответствие скобок в выражении.")
                    # Рекурсивно разобрать подвыражение внутри скобок
                    stack.append(parse_expression(tokens[i + 1:j]))
                    i = j  # Перейти к следующему токену после закрывающей скобки
                elif token in precedence:
                    # Оператор
                    while (operators and precedence.get(operators[-1], 0) >= precedence[token]):
                        apply_operator()
                    operators.append(token)
                elif token == ')':
                    # Закрывающая скобка завершает подвыражение
                    break
                else:
                    raise ValueError(f"Неизвестный токен: {token}")
                i += 1

            # Применить оставшиеся операторы
            while operators:
                apply_operator()

            return stack[0]

        return parse_expression(tokens)



class FloatDecimal:
    def __init__(self, number: str):
        print(number)
        minus = False
        if number.startswith('-'):
            number = number[1:]
            minus = True
        value: Decimal
        match number[:2]:
            case "0x":
                print(number, 17)
                value = Decimal(float.fromhex(number))
            case "0b":
                number = number[2:]
                integer_part, fractional_part = (number.split('.') if '.' in number else (number, ''))
                integer_value = int(integer_part, 2) if integer_part else 0
                fractional_value = sum(
                    int(bit) * 2**(-i) for i, bit in enumerate(fractional_part, start=1)
                )
                value = Decimal(integer_value + fractional_value)
            case "0t":
                number = number[2:]
                integer_part, fractional_part = (number.split('.') if '.' in number else (number, ''))
                integer_value = int(integer_part, 8) if integer_part else 0
                fractional_value = sum(
                    int(digit) * 8**(-i) for i, digit in enumerate(fractional_part, start=1)
                )
                value = Decimal(integer_value + fractional_value)
            case _:
                print(number[:2], "gh")
                value = Decimal(number)
        print(15)
        self.value = -value if minus else value
    def float(self):
        return self.value
    def __str__(self):
        print(str(self.value), 657)
        return str(self.value)

class Calculate:
    result: str
    expression: list[Any]
    def __init__(self, expression: Union[str, None], expression_list: Union[List, None] = None):
        self.result = "0"
        try:
            if expression:
                expression = str(Debuger(expression))
                self.expression = list(SimpleExpression(expression))
                print(self.expression)
                self.result = self.calc(self.expression)
            elif expression_list:
                self.expression = expression_list
                self.result = self.calc(self.expression)
        except Exception as e:
            print(e)
            traceback.print_exc()
            self.result = "Error"
    def calc(self: Self, expression: list[Any]):
        result: str = "Error"
        print(expression)
        match len(expression):
            case 1:
                result = expression[0]
            case 2:
                expression_1 = expression[1]
                expression_0 = expression[0]
                minus: bool = False
                if isinstance(expression[1], list):
                    expression_1 = self.calc(expression_1)
                if expression_0[0] == '-':
                    minus = True
                func: function
                match expression[0][-6:]:
                    case "arcsin":
                        func = asin
                    case "arccos":
                        func = acos
                    case "arctan":
                        func = atan
                    case "arccot":
                        func = lambda x: 1 / atan(x)
                    case "arcsec":
                        func = lambda x: 1 / acos(x)
                    case "arccsc":
                        func = lambda x: 1 / asin(x)
                    case _:
                        match expression[0][-3:]:
                            case "sin":
                                func = sin
                            case "cos":
                                func = cos
                            case "tan":
                                func = tan
                            case "cot":
                                func = lambda x: 1 / tan(x)
                            case "sec":
                                func = lambda x: 1 / cos(x)
                            case "csc":
                                func = lambda x: 1 / sin(x)
                            
                            case "log":
                                if isinstance(expression_1, list):
                                    func = lambda x: log(x, self.cash)
                                    self.cash = float(expression_1[2])
                                    expression_1 = expression_1[0]
                                    print(expression_1, "expression_1")
                                else:
                                    func = log
                            case "abs":
                                func = abs
                            case "sgn":
                                print('sgn')
                                func = lambda x: 1 if x > 0 else -1 if x < 0 else 0 
                            case _:
                                match expression[0][-2:]:
                                    case "ln":
                                        func = log
                                    case "lg":
                                        func = log10
                preliminary_result = str(func(float(expression_1)))
                preliminary_result_0 = preliminary_result[0]
                if minus and preliminary_result_0 == '-':
                    result = preliminary_result[1:]
                elif minus and not preliminary_result_0 == '-':
                    result = "-" + preliminary_result
                else:
                    result = preliminary_result
            case 3:
                expression_0 = expression[0]
                expression_1 = expression[1]
                expression_2 = expression[2]
                if isinstance(expression_0, list):
                    expression_0 = self.calc(expression_0)
                if isinstance(expression_2, list):
                    expression_2 = self.calc(expression_2)
                expression_0_Decimal = FloatDecimal(expression_0).float()
                expression_2_Decimal = FloatDecimal(expression_2).float()
                match expression_1:
                    case '*':
                        result = expression_0_Decimal * expression_2_Decimal
                    case '/':
                        result = expression_0_Decimal / expression_2_Decimal
                    case '+':
                        result = expression_0_Decimal + expression_2_Decimal
                    case '-':
                        result = expression_0_Decimal - expression_2_Decimal
                    case '^':
                        result = expression_0_Decimal ** expression_2_Decimal
                    case '|':
                        result = expression
                    case 'mod':
                        result = expression_0_Decimal % expression_2_Decimal
        if isinstance(result, list):
            return result
        else:
            return str(result)
    def __str__(self: Self) -> str:
        return self.result


@threaded_class
class Derivative(Calculate):
    result: str
    expression: list[Any] 
    diff: Union[List, str]
    def __init__(self: Self, expression: str, reverse_derivate: bool = False) -> Self:
        print("reverse_derivate", reverse_derivate)
        expression = expression.replace(" ", "")
        if expression == "": self.result = "0"
        else:
            try:
                expression = str(Debuger(expression))
                self.expression = list(SimpleExpression(expression))
                print(self.expression, 61)
                derivate = self.reverse_derivate if reverse_derivate else self.ordinar_derivate
                print(self.expression, 61)
                self.diff = derivate(self.expression)
    
            except Exception as e:
                print(e)
                traceback.print_exc()
                self.result = "Error"
                self.diff = "Error"
    def __iter__(self: Self):
        return iter(self.diff)
    def __str__(self: Self):
        return str(self.diff)
    def ordinar_derivate(self: Self, expression: list[Any]):
        match len(expression):
            case 1:
                print(expression)
                if len(expression[0]) == 1:
                    expression = '0' if expression[0][0] in "0123456789" else '1'
                else: 
                    expression = '0' if expression[0][1] in "0123456789" else '1'
            case 2:
                if expression[1][0] in "0123456789":
                    expression = '0'
                else:
                    expression_0: str
                    expression_1: Union[str, list] = expression[1]
                    complex_expression: bool = isinstance(expression_1, list)
                    is_minus: bool = expression[0][0] == '-'
                    match expression_0[-6:]:
                        case 'arcsin':
                            expression = [
                                    '1', '/', ['sqrt', ['1', '-',
                                    [expression_1, '^', '2']]]]
                        case 'arccos':
                            expression = [
                                    '-1', '/', ['sqrt', ['1', '-', 
                                    [expression_1, '^', '2']]]]
                        case 'arctan':
                            expression = [
                                    '1', '/', ['sqrt', ['1', '+',
                                    [expression_1, '^', '2']]]]
                        case 'arccot':
                            expression = [
                                    '-1', '/', ['sqrt', ['1', '+', 
                                    [expression_1, '^', '2']]]]
                        case 'arcsec':
                            expression = [
                                    '1', '/', [['abs', expression_1],
                                    '*', ['sqrt', ['1', '-',
                                    [expression_1, '^', '2']]]]]
                        case 'arccsc':
                            expression = [
                                    '-1', '/', [['abs', expression_1],
                                    '*', ['sqrt', ['1', '-',
                                    [expression_1, '^', '2']]]]]
                        case _:
                            match expression_0[-3:]:
                                case 'sin':
                                    expression = [
                                        '-cos' if is_minus else 'cos', 
                                        expression_1
                                    ]
                                case 'cos':
                                    expression = [
                                        'sin' if is_minus else '-sin',
                                        expression_1
                                    ]
                                case 'tan':
                                    expression = [
                                        [
                                            '-sec' if is_minus else 'sec',
                                            expression_1
                                        ], '^', '2']
                                case 'cot':
                                    expression = [
                                        [
                                            'csc' if is_minus else '-csc',
                                            expression_1
                                        ], '^', '2']
                                case 'sec': 
                                    expression = [
                                        [
                                            '-sec' if is_minus else 'sec',
                                            expression_1
                                        ],
                                        '*',
                                        ['tan', expression_1]
                                    ]
                                case 'csc':
                                    expression = [
                                        [
                                            'csc' if is_minus else '-csc',
                                            expression_1
                                        ],
                                        '*',
                                        ['cot', expression_1]
                                    ]
                                case 'abs':
                                    expression = ['sgn', expression_1]
                    expression = [expression, '*', self.ordinar_derivate(expression_1)]

            case 3:
                expression_0: Union[str, list] = expression[0]
                expression_2: Union[str, list] = expression[2]
                match expression[1]:
                    case "+" | "-":
                        expression = [
                                self.ordinar_derivate(expression_0), 
                                expression[1], 
                                self.ordinar_derivate(expression_2)
                        ]
                    case "*":
                        expression = [
                            [
                                self.ordinar_derivate(expression_0), 
                                '*',
                                expression_2
                            ], 
                            '+', 
                            [
                                expression_0, 
                                '*', 
                                self.ordinar_derivate(expression_2)
                            ]
                        ]
                    case "/":
                        expression = [
                            [
                                [
                                    self.ordinar_derivate(expression_0), 
                                    '*',
                                    expression_2
                                ], 
                                '+', 
                                [
                                    expression_0, 
                                    '*', 
                                    self.ordinar_derivate(expression_2)
                                ]
                            ],
                            '/',
                            [
                                expression_2,
                                '^',
                                '2'
                            ]
                        ]
                    case "^":
                        expression = [
                            [
                                expression_2,
                                '*',
                                [
                                    expression_0,
                                    '^',
                                    str(Decimal(expression_2) - Decimal(1))
                                ]
                            ],
                            '*',
                            self.ordinar_derivate(expression_0)
                        ]
                    case "|":
                        print(expression, 917)
                        ...
                        

        return expression

    def reverse_derivate(self: Self, expression: [list | str], const: bool = False):
        is_const: bool = False
        if len(expression[0]) == 0:
            expression.pop(0)
        print(len(expression), "er")
        match len(expression):
            case 1:
                print(19)
                index: int
                # Константа или переменная
                if len(expression[0]) == 1:
                    index = 0
                else: 
                    index = 1
                if expression[0].isdigit():
                    expression = [expression[0], "*", "x"]  # C -> C*x
                    is_const = True
                elif expression[0] == 'x':
                    expression = [["x", "^", "2"], "/", "2"]  # x -> x^2/2
                else:
                    expression = "Error"
                
            case 2:
                expression_1: [str | list] = expression[1]
                print(self.reverse_derivate(expression_1, True)[1], expression_1)
                if not self.reverse_derivate(expression_1, True)[1]:
                    is_minus: bool = expression[0][0] == '-'
                    match expression[0][-3:]:
                        case 'sin':
                            expression = ['cos' if is_minus else '-cos', expression_1]
                        case 'cos':
                            expression = ['-sin' if is_minus else 'sin', expression_1]
                        case 'tan':
                            expression_1: list[Any] = ["abs", ["cos", expression_1]]
                            expression = ["ln", expression_1] if is_minus else ["-ln", expression_1]
                        case 'cot':
                            expression_1: list[Any] = ["abs", ["sin", expression_1]]
                            expression = ['-ln', expression_1] if is_minus else ['ln', expression_1]
                        case 'sec':
                            expression_1: list[Any] = ["abs", [["sec", expression_1], "+", ["tan", expression_1]]]
                            expression = ["ln", expression_1] if is_minus else ["-ln", expression_1]
                        case 'csc':
                            expression_1: list[Any] = ["abs", [["sec", expression_1], "+", ["tan", expression_1]]]
                            expression = ["-ln", expression_1] if is_minus else ["ln", expression_1]
                    match expression[0][-2:]:
                        case 'ln':
                            if len(expression) == 2 and isinstance(expression[1], str):
                                # Простая форма ln(x)
                                expression = [
                                    [expression_1, "*", ["ln", expression_1]], "-", expression_1
                                ]
                            elif len(expression) == 2 and isinstance(expression[1], list):
                                # Сложная форма ln(f(x))
                                expression = [
                                    [expression, "*", ["ln", expression]], "-", self.reverse_derivate(expression_1)
                                ]
                else:
                    expression = [
                        expression,
                        '*',
                        'x'
                    ]
                    is_const = True
            case 3:
                expression_0: [str | list] = expression[0]
                expression_2: [str | list] = expression[2]
                match expression[1]:
                    case "+" | "-":
                        expression = [
                            self.reverse_derivate(expression_0),
                            expression[1],
                            self.reverse_derivate(expression_2)
                        ]
                    case "/":
                        expression = [
                            "ln", 
                            [
                                "abs", 
                                expression_0
                            ]
                        ]

                    case "*":
                        if expression_0 == expression_2:
                            expression = '1'
                        elif isinstance(expression_0, str):
                            if expression_0 == 'x':
                                expression_0 = self.reverse_derivate(expression_0)
                            else:
                                expression_2 = self.reverse_derivate(expression_2)
                            expression = [
                                expression_0,
                                '*',
                                expression_2
                            ]
                        elif isinstance(expression_2, str):
                            print('jklg')
                            if expression_2 == 'x':
                                expression_2 = self.reverse_derivate(expression_2)
                            else:
                                expression_0 = self.reverse_derivate(expression_0)

                            expression = [
                                expression_2,
                                '*',
                                expression_0
                            ]
                        """
                        # Если dv - это константа, то интегрируем напрямую
                        func = lambda expression: isinstance(expression, str) and (expression.isdigit() or expression == "x")
                        if func(expression_0) and func(expression_2):
                            if expression_0 == 'x':
                                expression_0, expression_2 = expression_2, expression_0
                            expression = [
                                [[expression_0, "*", [expression_2, "^", "2"]], "/", "2"]
                            ]
                         # Если u - это тангенс или котангенс, интегрируем их напрямую
                        elif isinstance(expression_0, list) and expression_0[0] == "tan":
                            result = [
                                [expression_2, "*", ["-ln", ["abs", ["cos", expression_0[1]]]]]
                            ]
                            return result
                        elif isinstance(expression_0, list) and expression_0[0] == "cot":
                            result = [
                                [expression_2, "*", ["ln", ["abs", ["sin", expression_0[1]]]]]
                            ]
                            return result
                        else:
                            # Интегрирование по частям
                            expression = [
                                [
                                    expression_0,  # Первый множитель
                                    '*',
                                    self.reverse_derivate(expression_2)  # Второй множитель
                                ],
                                '-',
                                [
                                    self.reverse_derivate(expression_2),
                                    '*',
                                    self.ordinar_derivate(expression_0)
                                ]
                            ]
                        """
                    case "^":
                        if isinstance(expression_2, str) and expression_2.isdigit():
                            new_exponent = str(Decimal(expression_2) + Decimal(1))
                            expression = [
                                [
                                    expression_0,
                                    "^",
                                    new_exponent
                                ],
                                "/",
                                new_exponent
                            ]
                        else:
                            expression = [expression, '/', ['ln', expression_0]]
                            #raise NotImplementedError("Integration of non-integer powers requires advanced techniques.")

        return (expression, is_const) if const else expression


            
        
                


@threaded_class
class Integral():
    __a:          Decimal
    __b:          Decimal
    __n:          int
    __EPS:        Decimal
    __result:     str
    __equation:   str
    __new_result: Decimal
    def __init__(self, *, a: str, b: str, EPS: str, equation: str):
        try:
            lst = list(Derivative(equation, True))
            self.__result = str(
                Decimal(str(Calculate(None, self.replace_x(lst, b)))) - 
                Decimal(str(Calculate(None, self.replace_x(lst, a))))
            )
        except Exception as e:
            print(e)
            traceback.print_exc()
            self.__result = "Error"
    def __str__(self):
        return self.__result
    def replace_x(self: Self, lst: list, new_value: str):
        return [
            self.replace_x(item, new_value) if isinstance(item, list) else new_value if item == 'x' else item
            for item in lst
        ]
