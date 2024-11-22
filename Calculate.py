from math import *
from decimal import Decimal

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
    def _float(self, number: str) -> Decimal:
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
    def _calculate_expression_base(self, tokens: list[str]) -> str:
        print(tokens, 73)
        priority_operator_index = 0
        while len(tokens) != 1:
            if '^' in tokens:
                priority_operator_index = tokens.index('^')
                b = self._float(tokens.pop(priority_operator_index+1))
                tokens.pop(priority_operator_index)
                a = self._float(tokens[priority_operator_index-1])
                tokens[priority_operator_index-1] = str(a**b)
            if '*' in tokens:
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
        else:
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
    def debuger(self):
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

    def removing_zeros(expression: str) -> str:
        if '.' in expression and not 'E' in expression:
            while expression[-1] == '0': expression = expression[:-1]
            if expression and expression[-1] == '.': expression = expression[:-1]
        return expression if expression else '0'

    # Основная функция подсчёта
    def calc(self) -> str:
        if not self.expression: return "0"
        try:
            self.debuger()
            
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
            self.removing_zeros(str(self._tokenize(expression_1)))
        except Exception as e:
            print(e)
            return "Error"
