from math import *
from decimal import Decimal
from re import sub


class BracketsError(Exception):
    ...
class NotOperatorError(Exception):
    ...

class Calculate:
    expression: str
    def __init__(self, expression):
        self.expression = expression.replace(" ", "")

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
    def _add_minus(self, result, minus):
        if minus: result = -result
        return result
    def _float(self, number: str) -> Decimal:
        minus: bool = False
        if number[0] == '-':
            number = number[1:]
            minus = True
        print(number, number[0], 34)
        result: Decimal
        match number[:2] :
            case "0x":
                result = Decimal(float.fromhex(number))
            case "0b":
                number = number[2:]
                # Разделяем целую и дробную части
                integer_part, fractional_part = (number.split('.') if '.' in number else (number, ''))

                # Преобразуем целую часть
                integer_value = int(integer_part, base = 2) if integer_part else 0

                # Преобразуем дробную часть
                fractional_value = sum(int(bit) * 2**(-i) for i, bit in enumerate(fractional_part, start=1))

                # Итоговое значение
                result = Decimal(integer_value) + Decimal(fractional_value)
            case "0t":    
                number = number[2:]
                integer_part, fractional_part = (number.split('.') if '.' in number else (number, ''))

                # Преобразуем целую часть
                integer_value = int(integer_part, 8) if integer_part else 0

                # Преобразуем дробную часть
                fractional_value = sum(int(digit) * 8**(-i) for i, digit in enumerate(fractional_part, start=1))

                # Итоговое значение
                result = Decimal(integer_value) + Decimal(fractional_value)
            case _:
                result = Decimal(number)
        return self._add_minus(result, minus)
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
        match number_or_operator:
            case "^" | "*" | "/" | "+" | "_" | "|" | ":":
                return False
        return True

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
            if '^' in tokens:
                priority_operator_index = tokens.index('^')
                b = self._float(tokens.pop(priority_operator_index+1))
                tokens.pop(priority_operator_index)
                a = self._float(tokens[priority_operator_index-1])
                tokens[priority_operator_index-1] = str(a**b)
            elif '*' in tokens:
                priority_operator_index = tokens.index('*')
                b = self._float(tokens.pop(priority_operator_index+1))
                tokens.pop(priority_operator_index)
                a = self._float(tokens[priority_operator_index-1])
                tokens[priority_operator_index-1] = str(a*b)
            elif ':' in tokens:
                priority_operator_index = tokens.index(':')
                b = self._float(tokens.pop(priority_operator_index+1))
                tokens.pop(priority_operator_index)
                a = self._float(tokens[priority_operator_index-1])
                tokens[priority_operator_index-1] = str(a/b)
            elif '/' in tokens:
                priority_operator_index = tokens.index('/')
                b = self._float(tokens.pop(priority_operator_index+1))
                tokens.pop(priority_operator_index)
                a = self._float(tokens[priority_operator_index-1])
                tokens[priority_operator_index-1] = str(a/b)
            elif '-' in tokens:
                priority_operator_index = tokens.index('-')
                b = self._float(tokens.pop(priority_operator_index+1))
                tokens.pop(priority_operator_index)
                a = self._float(tokens[priority_operator_index-1])
                tokens[priority_operator_index-1] = str(a-b)
            elif '+' in tokens:
                priority_operator_index = tokens.index('+')
                b = self._float(tokens.pop(priority_operator_index+1))
                tokens.pop(priority_operator_index)
                a = self._float(tokens[priority_operator_index-1])
                tokens[priority_operator_index-1] = str(a+b)
            elif '|' in tokens:
                print(tokens)
                tokens[0] = "".join([tokens[0], tokens[1], tokens[2] if len(tokens) == 3 else str(e)])
                while len(tokens) > 1:
                    tokens.pop()
            """
            else:
                raise NotOperatorError("_calculate_expression_base")
            """
        tokens[0] = str(self._float(tokens[0]))
        print(tokens[0], 35) 
        return tokens[0]
        
    # Calculate persent and factorial
    def _calculate_expression_list(self, tokens: list[str]) -> str:
        print(tokens)
        t: int
        while '%' in tokens:
            t = tokens.index('%')
            print(tokens)
            print(t)
            tokens.pop(t)
            print(tokens[t-1])
            tokens[t-1] = str(self._float(tokens[t-1])/ self._float(str(100)))
        while '!' in tokens:
            t = tokens.index('!')
            tokens.pop(t)
            tokens[t-1] = str(factorial(int(self._float(tokens[t-1]))))
        while 'n' in tokens:
            t = tokens.index('n')
            tokens.pop(t)
            tokens[t] = str(log(int(self._float(tokens[t]))))
            print(tokens, 67)
        while 'g' in tokens:
            t = tokens.index('g')
            tokens.pop(t)
            print(tokens[t], 45)
            tokens[t] = str(log10(int(self._float(tokens[t]))))
        while 's' in tokens:
            t = tokens.index('s')
            tokens.pop(t)
            tokens[t] = str(sqrt(int(self._float(tokens[t]))))
        while 'l' in tokens:
            t = tokens.index('l')
            tokens.pop(t)
            if len(tokens) >= t+2:
                print(True)
                if tokens[t+1] == '|':
                    tokens.pop(t+1)
                    tokens[t] = str(log(float(self._float(tokens[t])), float(self._float(tokens.pop(t+1) if len(tokens) >= t+2 else str(e)))))
                else:
                    tokens[t] = str(log(float(self._float(tokens[t]))))

            else:
                print(False)
                tokens[t] = str(log(float(self._float(tokens[t]))))
            print(890)

        return self._calculate_expression_base(tokens)
    class Debuger:
        """
        Класс для обработки математических выражений: очистка, замена операторов и проверка скобок.
        """
        def __init__(self, expression: str):
            expression = expression.replace("sqrt", "s").replace("ln", "n").replace("log", "l").replace("lg", "g").replace("**", "^")
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
                    raise BracketsError("verification_brackets")
            return bracket

        def __str__(self) -> str:
            """
            Возвращает обработанное выражение.
            """
            return self.expression
    # Разбиение строки на отдельные элементы (числа и операторы)
    def _tokenize(self, expression: str) -> str:
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
        return self._calculate_expression_list(result_list)
    @staticmethod
    def removing_zeros(expression: str) -> str:
        if '.' in expression and not 'E' in expression:
            while expression[-1] == '0': expression = expression[:-1]
            if expression and expression[-1] == '.': expression = expression[:-1]
        return expression if expression else '0'

    # Основная функция подсчёта
    def calc(self) -> str:
        if not self.expression: return "0"
        try:
            self.expression = str(self.Debuger(self.expression))
            
            expression_1 = self.expression
            while (count_brackets := expression_1.count("(")) != 0:
                print("1")
                priority_brackets = self._searching_for_priority_brackets(
                    expression_1, 
                    count_brackets
                )
                inner_expression = expression_1[priority_brackets[0] + 1:priority_brackets[1]]
                expression_1 = (
                    expression_1[:priority_brackets[0]] +
                    self._tokenize(inner_expression) +
                    expression_1[priority_brackets[1] + 1:]
                )
            return self.removing_zeros(str(self._tokenize(expression_1)))
        except Exception as e:
            print(e)
            return "Error"
