"""
CountdownRewardModel - Helper class for evaluating and scoring countdown equations
"""

import re
import operator


class CountdownRewardModel:
    def __init__(self, target, numbers):
        """
        Initialize the CountdownRewardModel with a target number and available numbers

        Args:
            target (int): The target number to reach
            numbers (list): List of available numbers to use in the equation
        """
        self.target = target
        self.numbers = numbers

    def extract_equation(self, solution_str):
        """
        Extract the equation from the solution string.

        Args:
            solution_str (str): The full solution string

        Returns:
            str or None: The extracted equation or None if no equation was found
        """
        # Remove everything before the first "Assistant:" if present
        if "Assistant:" in solution_str:
            solution_str = solution_str.split("Assistant:", 1)[1]
        elif "<|im_start|>assistant" in solution_str:
            solution_str = solution_str.split("<|im_start|>assistant", 1)[1]

        # Look for <answer> tags
        answer_pattern = r'<answer>(.*?)</answer>'
        match = re.finditer(answer_pattern, solution_str)
        matches = list(match)
        if matches:
            final_answer = matches[-1].group(1).strip()
            return final_answer
        return None

    def validate_equation(self, equation_str):
        """
        Validate that equation only uses available numbers and each number once.

        Args:
            equation_str (str): The equation string to validate

        Returns:
            bool: True if the equation is valid, False otherwise
        """
        try:
            # Extract all numbers from the equation
            numbers_in_eq = [int(n) for n in re.findall(r'\d+', equation_str)]

            # Check if all numbers in equation are available
            available_numbers = sorted(self.numbers)
            numbers_in_eq = sorted(numbers_in_eq)

            # Each number should be used exactly once
            return numbers_in_eq == available_numbers
        except Exception as e:
            print(f"Error validating equation: {e}")
            return False

    def evaluate_equation(self, equation_str):
        """
        Safely evaluate the arithmetic equation using eval() with precautions.

        Args:
            equation_str (str): The equation string to evaluate

        Returns:
            float or None: The result of the equation or None if evaluation failed
        """
        try:
            # Define a regex pattern that only allows numbers, operators, parentheses, and whitespace
            allowed_pattern = r'^[\d+\-*/().\s]+$'
            if not re.match(allowed_pattern, equation_str):
                print(f"Invalid characters in equation: {equation_str}")
                return None

            # Evaluate the equation with restricted globals and locals
            result = eval(equation_str, {"__builtins__": None}, {})
            return result
        except Exception as e:
            print(f"Error evaluating equation: {e}")
            return None

    def compute_score(self, solution_str, format_score=0.1, success_score=1.0, debug=False):
        """
        Compute a score for the given solution.

        Args:
            solution_str (str): The full solution string
            format_score (float): Score for correct format but wrong answer
            success_score (float): Score for correct answer
            debug (bool): Whether to print debug information

        Returns:
            float: Score between 0.0 and success_score
        """
        equation = self.extract_equation(solution_str)

        if debug:
            print(f"--------------------------------")
            print(f"Target: {self.target} | Numbers: {self.numbers}")
            print(f"Extracted equation: {equation}")
            print(f"Solution string: {solution_str}")

        if equation is None:
            if debug:
                print(f"No equation found")
            return 0.0

        # Validate equation uses correct numbers
        if not self.validate_equation(equation):
            if debug:
                print(f"Invalid equation: numbers don't match available numbers")
            return format_score

        # Evaluate equation
        result = self.evaluate_equation(equation)
        if result is None:
            if debug:
                print(f"Could not evaluate equation")
            return format_score

        if abs(result - self.target) < 1e-5:  # Account for floating point precision
            if debug:
                print(f"Correct equation: {equation} = {result}")
            return success_score
        else:
            if debug:
                print(f"Wrong result: equation = {result}, target = {self.target}")
            return format_score

    def find_solution(self, numbers=None, target=None, max_depth=6):
        """
        Find a solution to the countdown problem using a recursive approach.

        Args:
            numbers (list): List of available numbers (defaults to self.numbers if None)
            target (int): Target number (defaults to self.target if None)
            max_depth (int): Maximum recursion depth

        Returns:
            tuple: (bool, str) - Success status and solution expression
        """
        if numbers is None:
            numbers = self.numbers.copy()
        if target is None:
            target = self.target

        if len(numbers) == 1:
            return abs(numbers[0] - target) < 1e-5, str(numbers[0])

        if max_depth <= 0:
            return False, ""

        # Try all possible pairs of numbers
        for i in range(len(numbers)):
            for j in range(i + 1, len(numbers)):
                a, b = numbers[i], numbers[j]
                remaining = numbers.copy()
                remaining.remove(a)
                remaining.remove(b)

                # Try addition
                remaining.append(a + b)
                found, expr = self.find_solution(remaining, target, max_depth - 1)
                if found:
                    return True, expr.replace(str(a + b), f"({a} + {b})")
                remaining.pop()

                # Try subtraction (both ways)
                if a > b:
                    remaining.append(a - b)
                    found, expr = self.find_solution(remaining, target, max_depth - 1)
                    if found:
                        return True, expr.replace(str(a - b), f"({a} - {b})")
                    remaining.pop()

                if b > a:
                    remaining.append(b - a)
                    found, expr = self.find_solution(remaining, target, max_depth - 1)
                    if found:
                        return True, expr.replace(str(b - a), f"({b} - {a})")
                    remaining.pop()

                # Try multiplication
                remaining.append(a * b)
                found, expr = self.find_solution(remaining, target, max_depth - 1)
                if found:
                    return True, expr.replace(str(a * b), f"({a} * {b})")
                remaining.pop()

                # Try division (both ways, only if divisible)
                if a != 0 and b % a == 0:
                    remaining.append(b // a)
                    found, expr = self.find_solution(remaining, target, max_depth - 1)
                    if found:
                        return True, expr.replace(str(b // a), f"({b} / {a})")
                    remaining.pop()

                if b != 0 and a % b == 0:
                    remaining.append(a // b)
                    found, expr = self.find_solution(remaining, target, max_depth - 1)
                    if found:
                        return True, expr.replace(str(a // b), f"({a} / {b})")
                    remaining.pop()

        return False, ""


if __name__ == "__main__":
    # Example usage
    target = 24
    numbers = [3, 6, 8, 10]
    model = CountdownRewardModel(target, numbers)

    # Test equation validation
    print("Validating equations:")
    print(model.validate_equation("(3 + 6) * (8 - 10)"))  # Should be True
    print(model.validate_equation("(3 + 6) * 8"))  # Should be False (missing 10)

    # Test equation evaluation
    print("\nEvaluating equations:")
    print(model.evaluate_equation("(3 + 6) * (8 - 10)"))  # Should be -18

    # Test scoring
    print("\nScoring solutions:")
    solution1 = "Let me solve this step by step.\n<think>I need to reach 24 using [3, 6, 8, 10].\n\nI'll try different operations:\n- 10 + 8 = 18\n- 18 + 6 = 24\nThis works! I can use 10 + 8 + 6 = 24, but I still have 3 left.\n\nLet me try a different approach:\n- 3 * 8 = 24\nPerfect! This uses 3 and 8, but I still have 6 and 10 left.\n\nLet me try again:\n- 6 * 3 = 18\n- 18 + 6 = 24\nThat still doesn't use all numbers.\n\nOne more try:\n- (10 - 8) * 3 * 6 = 2 * 3 * 6 = 36\nNo, that's not right.\n\nAlright, I see that I can use (3 * 10) - 6 = 30 - 6 = 24. This uses three numbers but not all four.\n\nLet me try: (10 - 6) * (8 - 3) = 4 * 5 = 20. That's close but not 24.\n\nHow about: (10 + 6) * (8 / 4) = 16 * 2 = 32? No, that doesn't use the 3.\n\nLet me try: (10 + 8) * (6 / 3) = 18 * 2 = 36. No, not right.\n\nLet me try: (10 - 3) * (8 - 6) = 7 * 2 = 14. No, not 24.\n\nLet's try: (8 - 6) * 10 + 3 = 2 * 10 + 3 = 23. So close!\n\nLet's try: (8 - 6) * (10 + 3) = 2 * 13 = 26. No, not quite.\n\n(3 * 8) = 24 is the most efficient solution, but it doesn't use all numbers.\n\nAh! I have it: 3 * 8 - 10 + 6 = 24 - 10 + 6 = 14 + 6 = 20. Still not right.\n\nWhat about (3 * 10) - (8 - 6) = 30 - 2 = 28. Nope.\n\nOne more attempt: (3 + 6) * (10 - 8) = 9 * 2 = 18. Not 24.\n\nHow about (6 / 3) * 10 + 8 = 2 * 10 + 8 = 28. No.\n\nLet me try: (6 / 3) * (10 + 8) = 2 * 18 = 36. No.\n\nI think I need to use all four numbers in a specific way.\n\nOh! I see it: (8 / (6 - 3)) * 10 = (8 / 3) * 10. Wait, 8 is not divisible by 3 within integers.\n\nLet me try: ((10 * 6) / 3) + 8 = (60 / 3) + 8 = 20 + 8 = 28. No.\n\nHow about: ((10 + 8) * 6) / 3 = (18 * 6) / 3 = 108 / 3 = 36. No.\n\nI need to find a solution where the four numbers [3, 6, 8, 10] combine to make 24.\n\nI see that 3 * 8 = 24, but that doesn't use 6 and 10.\n\nLet me try: 6 * (8 - 3) - 10 = 6 * 5 - 10 = 30 - 10 = 20. Not quite.\n\nOne more: (6 * 8) - (10 * 3) = 48 - 30 = 18. No.\n\nWhat about: 10 - ((8 - 6) / 3) = 10 - (2/3) = 10 - 0.66... That doesn't work within integers.\n\nLet's try: 10 - 8 + 6 * 3 = 2 + 18 = 20. Not 24.\n\nI realize I might not be able to use all four numbers with integer arithmetic to get exactly 24. Let me provide the closest solution I can find.\n\nThe simplest solution is 3 * 8 = 24, but it only uses two numbers.\n\nUsing three numbers: (10 + 6) + 8 = 24. No, that's actually 16 + 8 = 24. So this works with 10, 6, and 8.</think>\n\n<answer>(10 + 6) + 8</answer>"
    solution2 = "Let me solve this step by step.\n<think>I need to reach 24 using [3, 6, 8, 10].\n\nLet me try different operations:\n\n3 * 8 = 24\nThis works perfectly! I've found a solution using just two of the numbers. Let me see if I can find a solution using all four numbers.\n\n(10 + 6) + 8 = 24\nNo, that's actually 16 + 8 = 24, but it still only uses three numbers.\n\n3 * (10 - 6) = 3 * 4 = 12\nThat's not right.\n\n(10 - 6) * (8 - 3) = 4 * 5 = 20\nClose, but not 24.\n\nOh wait, I can do:\n(3 * 8) + (10 - 6) = 24 + 4 = 28\nThat's not 24 either.\n\nLet me try one more approach:\n(3 * 10) - 6 = 30 - 6 = 24\nThis works and uses 3, 10, and 6, but not 8.\n\nAfter exploring multiple possibilities, I believe 3 * 8 = 24 is the most direct solution, even though it only uses two of the four numbers.</think>\n\n<answer>3 * 8</answer>"
    solution3 = "Let me solve this step by step.\n<think>I need to reach 24 using [3, 6, 8, 10].\n\nLet me try different operations:\n\n3 * 8 = 24\nThis works perfectly! I've found a solution using just two of the numbers. Let me see if I can find a solution using all four numbers.\n\n(10 + 6) + 8 = 24\nNo, that's actually 16 + 8 = 24, but it still only uses three numbers.\n\n3 * (10 - 6) = 3 * 4 = 12\nThat's not right.\n\n(10 - 6) * (8 - 3) = 4 * 5 = 20\nClose, but not 24.\n\nOh wait, I can do:\n(3 * 8) + (10 - 6) = 24 + 4 = 28\nThat's not 24 either.\n\nLet me try one more approach:\n(3 * 10) - 6 = 30 - 6 = 24\nThis works and uses 3, 10, and 6, but not 8.\n\nAfter exploring multiple possibilities, I believe 3 * 8 = 24 is the most direct solution, even though it only uses two of the four numbers.</think>\n\n<answer>3 * (10 - (8 - 6))</answer>"

    print(model.compute_score(solution1, debug=True))
    print(model.compute_score(solution2, debug=True))
    print(model.compute_score(solution3, debug=True))

    # Test solution finder
    print("\nFinding solution:")
    found, solution = model.find_solution()
    print(f"Found solution: {found}, Expression: {solution}")