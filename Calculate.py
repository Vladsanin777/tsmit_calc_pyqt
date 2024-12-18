from math import *
from decimal import Decimal
from re import sub
import traceback
import threading
from functools import wraps
from typing import Self, Any

class Debuger:
        """
        Класс для обработки математических выражений: очистка, замена операторов и проверка скобок.
        """
        def __init__(self, expression: str):
            if "^*" in expression:
                raise Exception("Two operators in expression \"^*\"")
            if "*^" in expression:
                raise Exception("Two operators in expression \"*^\"")

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




class SimpleExpression():
    result: list[str]
    # Разбиение строки на отдельные элементы (числа и операторы)
    def __init__(self: Self, expression: str):


        # sheach first digit
        hex_i: int = expression[1]=='x' if len(expression) > 1 else False
        positions: list[int] = list() 
        last_element: int = 0
        for element_i in range(len(expression)):
            if "%!-+*/:^|()sincoetanqrlg".find(expression[element_i]) != -1 and not ("-+".find(expression[element_i]) and "Ee".find(expression[element_i-1]) != -1 and not hex_i):
                hex_i = expression[element_i+2] == 'x' if len(expression) > element_i+3 else False
                if last_element != element_i:
                    positions.append(element_i)
                last_element = element_i + 1


                
        print(positions, 78)
        result_list = list()
        index_old: int = -2
        delete_positions: int = 0
        for index in range(len(positions)):
            index -= delete_positions
            print(index)
            if index_old + 1 == positions[index] and expression[positions[index]] in ")":
                print(positions.pop(index), 90)
                delete_positions += 1
            else:
                index_old = positions[index]


        print(positions)
        while positions != []:
            print(positions, 77)
            # wrating first digit in result list
            if (element := expression[(number_operators := positions.pop())+1:]) != "": 
                result_list.append(element)
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
        print(result_list, 33)
        for func in [
                ['sin', ['s', 'i', 'n']], 
                ['cos', ['c', 'o', 's']], 
                ['tan', ['t', 'a', 'n']], 
                ['cot', ['c', 'o', 't']], 
                ['sec', ['s', 'e', 'c']], 
                ['csc', ['c', 's', 'c']]
        ]:
            
            while func[1] in result_list:
                index = result_list.index(func[1])  # Найти подсписок
                result_list[index] = func[0]       # Заменить подсписок на строку

            # Проверяем наличие строки последовательных символов
            for i in range(len(result_list) - len(func[1]) + 1):
                if result_list[i:i + len(func[1])] == func[1]:  # Если срез совпадает
                    result_list[i:i + len(func[1])] = [func[0]]
                    """
                    if len(result_list) > i+2:
                        if result_list[i + 1] != '(':
                            result_list.insert(i+1, '(')
                            if len(result_list) > i+4:
                                result_list.insert(i+3, ')')
                            else:
                                result_list.append(')')
                    """
        print(result_list, 34)
        self.result = result_list
        self.delete_brackets()
        self.result = self.add_lists(self.result)
        print(self.result, 13)
    def __iter__(self):
        return iter(self.result)
    def delete_brackets(self: Self) -> None:
        print(self.result, 579)
        while "(" in self.result:
            print(89)
            index_open_bracket = max(index for index in range(len(self.result)) if self.result[index] == '(')
            index_close_bracket: str
            for index_close_bracket in \
                    [index for index in range(len(self.result)) if self.result[index] == ')']:
                if index_open_bracket < index_close_bracket:
                    break
            self.result.pop(index_open_bracket)
            if index_open_bracket != 0 and self.result[index_open_bracket-1][0] in "0123456789":
                self.result.insert(index_open_bracket, "*")
                index_open_bracket += 1
                index_close_bracket += 1
            sub_result: list[str] = list()
            for _ in range(index_close_bracket - index_open_bracket - 1):
                sub_result.append(self.result.pop(index_open_bracket))
            self.result[index_open_bracket] = sub_result[0] if len(sub_result) == 1 else sub_result
            index_open_bracket += 1
            if len(self.result) > index_open_bracket:
                if self.result[index_open_bracket][0] in "0123456789":
                    self.result.insert(index_open_bracket, "*")
        while len(self.result) == 1 and isinstance(self.result[0], list):
            self.result = self.result[0]
    def add_lists(self: Self, equality) -> None:
        index_priority_operator: int = 0
        priority_operator_func = lambda lst, x: len(lst) - 1 - lst[::-1].index(x)
        if "sin" in equality:
            index_priority_operator = priority_operator_func(equality, "sin")
        if "^" in equality:
            index_priority_operator = priority_operator_func(equality, "^")
        if "*" in equality and "/" in equality:
            index_priority_operator = priority_operator_func(equality, "*")
            if index_priority_operator and index_priority_operator < (t := priority_operator_func(equality, "-")):
                index_priority_operator = t
        elif "*" in equality:
            index_priority_operator = priority_operator_func(equality, "*")
        elif "/" in equality:
            index_priority_operator = priority_operator_func(equality, "/")
        if "+" in equality and "-" in equality:
            index_priority_operator = priority_operator_func(equality, "+")
            if index_priority_operator and index_priority_operator < (t := priority_operator_func(equality, "-")):
                index_priority_operator = t
        elif "+" in equality:
            index_priority_operator = priority_operator_func(equality, "+")
        elif "-" in equality:
            index_priority_operator = priority_operator_func(equality, "-")
        if index_priority_operator:
            first_part = self.add_lists(t) if len(t := equality[:index_priority_operator]) > 1 else t[0]
            last_part = self.add_lists(t) if len(t := equality[index_priority_operator+1:]) > 1 else t[0]
            operator = equality[index_priority_operator]
            equality.clear()
            equality.append(first_part)
            equality.append(operator)
            equality.append(last_part)
        return equality




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
    return ThreadedClass


class Calculate:
    result: str
    def __init__(self, expression):
        expression = expression.replace(" ", "")
        if expression == "": self.result = "0"
        else:
            try:
                expression = str(self.Debuger(expression))
                while (count_brackets := expression.count("(")) != 0:
                    print("1")
                    priority_brackets = self._searching_for_priority_brackets(
                        expression, 
                        count_brackets
                    )
                    inner_expression = expression[priority_brackets[0] + 1:priority_brackets[1]]
                    print(4)
                    add_multiplication = lambda symbol: '' if symbol in "-+*/^:snlgm|" else '*'
                    expression = (
                        (t1 := expression[:priority_brackets[0]]) +
                        add_multiplication(t1[-1])+
                        str(self.SimpleExpressionCalculation(inner_expression)) +
                        (add_multiplication(t2[0]) +
                        t2 if len(t2 := expression[priority_brackets[1] + 1:]) != 0 else '')
                    )
                self.result = self.removing_zeros(str(self.SimpleExpressionCalculation(expression)))
            except Exception as e:
                print(e)
                traceback.print_exc()
                self.result = "Error"

    def __str__(self):
        return self.result
    def float(self):
        return Decimal(self.result)
    
    def _find_nth_occurrence(self, string: str, substring: str, n: int) -> int:
        start = 0
        for _ in range(n):
            start = string.find(substring, start)
            if start == -1:
                return -1  # Если подстрока не найдена
            start += len(substring)
        return start - len(substring)
    
    # Метод для поиска приоритетных скобок
    def _searching_for_priority_brackets(self, expression: str, number_of_bracket: int) -> list[int]:
        return [(number_last_open_brackets := self._find_nth_occurrence(expression, '(', number_of_bracket)), expression.find(')', number_last_open_brackets)]
    def removing_zeros(self, expression: str) -> str:
        if '.' in expression and not 'E' in expression:
            while expression[-1] == '0': expression = expression[:-1]
            if expression and expression[-1] == '.': expression = expression[:-1]
        return expression if expression else '0'


    class Debuger:
        """
        Класс для обработки математических выражений: очистка, замена операторов и проверка скобок.
        """
        def __init__(self, expression: str):
            if "^*" in expression:
                raise Exception("Two operators in expression \"^*\"")
            if "*^" in expression:
                raise Exception("Two operators in expression \"*^\"")

            expression = expression.replace("**", "^").replace(":", "/").replace(",", ".").replace("//", "/").replace("--", "+")
            while "++" in expression:
                expression = expression.replace("++", "+")
            while "//" in expression:
                expression = expression.replace("//", "/")
            while "^^" in expression:
                expression = expression.replace("^^", "^")
            expression = expression.replace("^*", "^")
            # Удаление символов в конце строки
            expression = sub(r'[*/:+\-\^lmngs()]+$', '', expression)
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
    @threaded_class
    class SimpleExpressionCalculation:

        # Разбиение строки на отдельные элементы (числа и операторы)
        def __init__(self, expression: str):
            # sheach first digit
            hex_i: int = expression[1]=='x' if len(expression) > 1 else False
            positions: list[int] = list() 
            for element_i in range(len(expression)):
                if "%!-+*/:^sngolm|".find(expression[element_i]) != -1 and not ("-+".find(expression[element_i]) and "Ee".find(expression[element_i-1]) != -1 and not hex_i):
                    positions.append(element_i)
                    hex_i = expression[element_i+2] == 'x' if len(expression) > element_i+3 else False


                    
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
            self.result = self._calculate_expression_list(result_list)
            print(self.result, 13)
        def __str__(self) -> str:
            return self.result
        def _number_negative(self, number):
            return number[1:] if number[0] == '-' else '-' + number
        def _check_for_negative_numbers(self, tokens: list[str]) -> list[str]:
            if tokens[0] == '-': 
                tokens.pop(0)
                tokens[0] = self._number_negative(tokens[0])
            #print([index for index, value in enumerate(tokens) if value == "-"])
            for index in reversed([index for index, value in enumerate(tokens) if value == "-"]):
                if tokens[index-1] in "-/*+":
                    print(1)
                    tokens.pop(index)
                    tokens[index] = self._number_negative(tokens[index])
            print(tokens, 1234)
            return tokens
            
        def is_not_operator(self, number_or_operator):
            return "%!-+*/:^sngol|m".find(number_or_operator) == -1

        def not_operator(self, tokens):
            print(tokens, len(tokens), "len")
            for index in range(len(tokens)-1):
                if self.is_not_operator(tokens[index]) and self.is_not_operator(tokens[index+1]):
                    return index + 1
            return 0
        
        def adding_multiplication(self, tokens):
            while (index := self.not_operator(tokens)):
                print(index, 102)
                tokens.insert(index, "*")
            return tokens

        # Main method for calculate
        def _calculate_expression_base(self, tokens: list[str]) -> str:
            print(tokens, 73)
            tokens = self._check_for_negative_numbers(tokens)
            print(tokens)
            priority_operator_index = 0
            if len(tokens) > 1:
                tokens = self.adding_multiplication(tokens)
            print(tokens, 89)
            while len(tokens) != 1:
                """ Adding operation mod """
                tokens_index = tokens.index
                if "^" in tokens:
                    tokens.pop(priority_operator_index := tokens_index("^"))
                    tokens[priority_operator_index-1] = str(
                        self.FloatDecimal(tokens[priority_operator_index-1]).float() **
                        self.FloatDecimal(tokens.pop(priority_operator_index)).float()
                    )
                elif "*" in tokens:
                    tokens.pop(priority_operator_index := tokens_index("*"))
                    tokens[priority_operator_index-1] = str(
                        self.FloatDecimal(tokens[priority_operator_index-1]).float() *
                        self.FloatDecimal(tokens.pop(priority_operator_index)).float()
                    )
                elif (verification := ":" in tokens) or "/" in tokens:
                    tokens.pop(priority_operator_index := tokens_index(":" if verification else "/"))
                    tokens[priority_operator_index-1] = str(
                        self.FloatDecimal(tokens[priority_operator_index-1]).float() /
                        self.FloatDecimal(tokens.pop(priority_operator_index)).float()
                    )
                elif "m" in tokens:
                    tokens.pop(priority_operator_index := tokens_index("m"))
                    tokens[priority_operator_index-1] = str(
                        self.FloatDecimal(tokens[priority_operator_index-1]).float() %
                        self.FloatDecimal(tokens.pop(priority_operator_index)).float()
                    )
                elif "-" in tokens:
                    tokens.pop(priority_operator_index := tokens_index("-"))
                    tokens[priority_operator_index-1] = str(
                        self.FloatDecimal(tokens[priority_operator_index-1]).float() -
                        self.FloatDecimal(tokens.pop(priority_operator_index)).float()
                    )
                elif "+" in tokens:
                    tokens.pop(priority_operator_index := tokens_index("+"))
                    tokens[priority_operator_index-1] = str(
                        self.FloatDecimal(tokens[priority_operator_index-1]).float() +
                        self.FloatDecimal(tokens.pop(priority_operator_index)).float()
                    )
                elif '|' in tokens:
                    print(tokens)
                    tokens[0] = "".join(tokens)
                    while len(tokens) > 1:
                        tokens.pop()
            print(tokens[0])
            return str(self.FloatDecimal(tokens[0]).float()+Decimal(0.0))
            
        # Calculate persent and factorial
        def _calculate_expression_list(self, tokens: list[str]) -> str:
            print(tokens)
            position: list[int] = [element_i for element_i in range(len(tokens)) if "%!ngsl".find(tokens[element_i]) != -1]
            print(position)
            step: int = 0
            for index in position:
                index -= step
                step += 1
                if '%' == tokens[index]:
                    print(tokens)
                    tokens.pop(index)
                    print(tokens[index-1])
                    tokens[index-1] = str(self.FloatDecimal(tokens[index-1]).float() / self.FloatDecimal(str(100)).float())
                elif '!' == tokens[index]:
                    tokens.pop(index)
                    tokens[index-1] = str(factorial(int(self.FloatDecimal(tokens[index-1]).float())))
                elif 'n' == tokens[index]:
                    tokens.pop(index)
                    tokens[index] = str(log(float(self.FloatDecimal(tokens[index]).float())))
                    print(tokens, 67)
                elif 'g' == tokens[index]:
                    tokens.pop(index)
                    print(tokens[index], 45)
                    tokens[index] = str(log10(float(self.FloatDecimal(tokens[index]).float())))
                elif 's' == tokens[index]:
                    tokens.pop(index)
                    tokens[index] = str(sqrt(float(self.FloatDecimal(tokens[index]).float())))
                elif 'l' == tokens[index]:
                    tokens.pop(index)
                    if len(tokens) >= index+2:
                        print(True)
                        if tokens[t+1] == '|':
                            tokens.pop(index+1)
                            tokens[index] = str(log(float(self.FloatDecimal(tokens[index]).float()), float(self.FloatDecimal(tokens.pop(index+1) if len(tokens) >= t+2 else str(e)).float())))
                        else:
                            tokens[index] = str(log(float(self.FloatDecimal(tokens[index]).float())))

                    else:
                        print(False)
                        tokens[index] = str(log(float(self.FloatDecimal(tokens[index]).float())))
                    print(890)

            return self._calculate_expression_base(tokens)
        @threaded_class
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

@threaded_class
class Derivative(Calculate):
    result: str
    expression: list[Any] 
    diff: list[Any]
    def __init__(self: Self, expression: str, reverse_derivate: bool = False) -> Self:
        print("reverse_derivate", reverse_derivate)
        expression = expression.replace(" ", "")
        if expression == "": self.result = "0"
        else:
            try:
                expression = str(Debuger(expression))
                self.expression = list(SimpleExpression(expression))
                derivate = self.reverse_derivate if reverse_derivate else self.ordinar_derivate
                self.diff = derivate(self.expression)
                print(self.diff)
            except Exception as e:
                print(e)
                traceback.print_exc()
                self.result = "Error"
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
                    expression_1: [str | list] = expression[1]
                    complex_expression: bool = isinstance(expression_1, list)
                    raising_to_a_power: bool = False
                    sec_or_csc: bool = False
                    is_minus: bool = expression[0][0] == '-'
                    match expression[0][-3:]:
                        case 'sin':
                            expression = '-cos' if is_minus else 'cos'
                        case 'cos':
                            expression = 'sin' if is_minus else '-sin'
                        case 'tan':
                            expression = '-sec' if is_minus else 'sec'
                            raising_to_a_power = True
                        case 'cot':
                            expression = 'csc' if is_minus else '-csc'
                            raising_to_a_power = True
                        case 'sec': 
                            expression = [['-sec' if is_minus else 'sec', expression_1], '*', ['tan', expression_1]]
                            sec_or_csc = True
                        case 'csc':
                            expression = [['csc' if is_minus else '-csc', expression_1], '*', ['cot', expression_1]]
                            sec_or_csc = True
                    if not sec_or_csc:
                        expression = [expression, expression_1]
                        if raising_to_a_power:
                            expression = [expression, '^', '2']
                    if complex_expression:
                        expression = [expression, '*', self.ordinar_derivate(expression_1)]

            case 3:
                expression_0: [str | list] = expression[0]
                expression_2: [str | list] = expression[2]
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
                        

        return expression

    def reverse_derivate(self: Self, expression: [list | str]):
        match len(expression):
            case 1:
                print(19)
                # Константа или переменная
                if len(expression[0]) == 1:
                    if expression[0][0] in "0123456789":
                        expression = [expression[0], "*", "x"]  # C -> C*x
                    else:
                        expression = [["x", "^", "2"], "/", "2"]  # x -> x^2/2
                else:
                    expression = ["x"] if expression[0][1] in "0123456789" else [expression[0], "*", "x"]

            case 2:
                expression_1: [str | list] = expression[1]
                is_minus: bool = expression[0][0] == '-'
                match expression[0][-3:]:
                    case 'sin':
                        expression = ['-cos' if is_minus else 'cos', expression_1]
                    case 'cos':
                        expression = ['sin' if is_minus else '-sin', expression_1]
                    case 'tan':
                        expression = ['-ln|cos' if is_minus else 'ln|cos', expression_1]
                    case 'cot':
                        expression = ['ln|sin' if is_minus else '-ln|sin', expression_1]
                    case 'sec':
                        expression = ['ln|sec+tan' if is_minus else '-ln|sec+tan', expression_1]
                    case 'csc':
                        expression = ['-ln|csc+cot' if is_minus else 'ln|csc+cot', expression_1]
                    case _:
                        if isinstance(expression_1, list):
                            expression = [
                                self.reverse_derivate(expression_1), "*", "x"
                            ]
                        else:
                            expression = [expression[1], "*", "x"]

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
                    case "*":
                        # Интеграл произведения
                        if isinstance(expression_0, str) and expression_0.isdigit():
                            expression = [
                                expression_0,
                                "*",
                                self.reverse_derivate(expression_2)
                            ]
                        elif isinstance(expression_2, str) and expression_2.isdigit():
                            expression = [
                                expression_2,
                                "*",
                                self.reverse_derivate(expression_0)
                            ]
                        else:
                            raise NotImplementedError("Integration by parts is required.")
                    case "/":
                        # Интеграл частного (замена переменной для линейных функций)
                        if isinstance(expression_2, str) and expression_2.isdigit():
                            expression = [
                                "ln|",
                                expression_0
                            ]
                        else:
                            raise NotImplementedError("Complex division requires substitution.")
                    case "^":
                        if isinstance(expression_2, str) and expression_2.isdigit():
                            new_exponent = str(int(expression_2) + 1)
                            expression = [
                                expression_0,
                                "^",
                                new_exponent,
                                "/",
                                new_exponent
                            ]
                        else:
                            raise NotImplementedError("Integration of non-integer powers requires advanced techniques.")

        return expression



            
        
                


@threaded_class
class Integral():
    __a:          Decimal
    __b:          Decimal
    __n:          int
    __EPS:        Decimal
    __result:     Decimal
    __equation:   str
    __new_result: Decimal
    def __init__(self, *, a: str, b: str, EPS: str, equation: str):
        Derivative(equation, True)
    def __str__(self):
        return "0"
    """
        self.__n =          2
        self.__a =          Decimal(a)
        self.__b =          Decimal(b)
        self.__EPS =        Decimal(EPS)
        self.__equation =   equation
        self.__result = Decimal(0)
        self.integral()
        while (abs(self.__result - self.__new_result) >= self.__EPS):
            print(self.__result, self.__new_result)
            self.__n **=    2
            self.__result = self.__new_result
            self.integral()
            print(abs(self.__result - self.__new_result))
    def integral(self):
        
        # Grid spacing
        h: Decimal = (self.__b - self.__a) / Decimal(self.__n)

        
        # Computing sum of first and last terms
        # in above formula
        s: Decimal = (
            Calculate(self.__equation.replace('x', str(self.__a))).float() + 
            Calculate(self.__equation.replace('x', str(self.__b))).float()
        )

        # Adding middle terms in above formula
        i: Decimal = Decimal(1)
        h_i: Decimal = Decimal(1)
        while i < self.__n:
            s += Decimal(2) * Calculate(self.__equation.replace('x', str(self.__a + i * h))).float()
            i += h_i
            
        # h/2 indicates (b-a)/2n. 
        # Multiplying h/2 with s.
        self.__new_result = h / Decimal(2) * s
    def __str__(self):
        return str(self.__new_result)
    """
