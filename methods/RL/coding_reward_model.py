"""
Coding Reward Model - Helper class for evaluating and scoring coding problems
"""


import re
import operator
import coding_utils
import datasets






class CodingRewardModel:
    def __init__(self, testcases):
        """

        Args:
            testcases: the testcases in an array of [testcases, solution]
        """
        self.testcases = testcases




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
        valid, output_solution = self.extract_solution(solution_str)

        #TODO: How to handle incorrect formatting case
        if not valid:
            return 0

        
        if debug:
            print(f"--------------------------------")
            print(f"Answer: {self.answer} | Output: {output_solution}")
            print(f"Solution string: {solution_str}")

        try:
            
            results = coding_utils.check_answer(
                completion=output_solution,
                verification=self.testcases,
                task_id="task_id",
            )
            print(f"Percentage correct: {results['output']}")
            
            if float(results['output']) > .99:
                if debug:
                    print(f"Score: {success_score}")
                return success_score
        except TypeError as e:
              print("TypeError during comparison:")
              print(f"  self.answer: {self.answer} (type: {type(self.answer)})")
              print(f"  output_solution: {output_solution} (type: {type(output_solution)})")

        return format_score


    def extract_solution(self, llm_output):

        answer_match = re.search(r'<code>\s*(.*?)\s*</code>', llm_output, re.DOTALL)
        
        if answer_match:
            try:
                answer_content = answer_match.group(1)
                extracted_answer = coding_utils.extract_answer(answer_content)
                if extracted_answer is not None and extracted_answer != "":
                    self.answer = extracted_answer
                    return True, extracted_answer
                
            except (ValueError, AttributeError):
                return False, ""
        
        return False, ""


if __name__ == "__main__":
    # Example usage
    

    ds = datasets.load_dataset("open-r1/codeforces", split="train")

    prob_ds = ds.filter(lambda ex: ex["id"] == "994/A")
    verification = prob_ds[0]["official_tests"]

    model = CodingRewardModel(verification)

    completion = """
<code>
```python
n , m = map(int,input().split())
arrn = list(map(int ,input().split()))
arrm = list(map(int,input().split()))
for i in arrn:
    if i in arrm:
        print(i,end =" ")
```
</code>
    """

    # Test scoring

    print(model.compute_score(completion, debug=True))

