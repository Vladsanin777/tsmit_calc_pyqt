from math import *
from decimal import Decimal
from re import sub
import traceback


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
            expression = expression.replace("sqrt", "s").replace("ln", "n").replace("log", "l").replace("lg", "g").replace("**", "^").replace("mod", "m").replace(",", ".")
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

    
