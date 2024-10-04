# https://github.com/FastEval/FastEval/blob/main/evaluation/benchmarks/cot_math_equivalence.py

import re
from fractions import Fraction
import math

def last_boxed_only_string(string):
    idx = string.rfind("\\boxed")
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None
    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1
    if right_brace_idx == None:
        retval = None
    else:
        retval = string[idx : right_brace_idx + 1]
    return retval

def remove_box(s):
    if "\\boxed{" in s:
        left = "\\boxed{"
    elif "\\fbox{" in s:
        left = "\\fbox{"
    try:
        assert s[: len(left)] == left
        assert s[-1] == "}"
        return s[len(left) : -1]
    except:
        return None

# s = last_boxed_only_string("x = \\boxed{\\frac{1}{2}}")
# print(remove_box(s))

def _fix_fracs(string):
    substrs = string.split("\\frac")
    new_str = substrs[0]
    if len(substrs) > 1:
        substrs = substrs[1:]
        for substr in substrs:
            new_str += "\\frac"
            if substr[0] == "{":
                new_str += substr
            else:
                try:
                    assert len(substr) >= 2
                except:
                    return string
                a = substr[0]
                b = substr[1]
                if b != "{":
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}{" + b + "}" + post_substr
                    else:
                        new_str += "{" + a + "}{" + b + "}"
                else:
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}" + b + post_substr
                    else:
                        new_str += "{" + a + "}" + b
    string = new_str
    return string

def _fix_a_slash_b(string):
    if len(string.split("/")) != 2:
        return string
    a = string.split("/")[0]
    b = string.split("/")[1]
    try:
        a = int(a)
        b = int(b)
        assert string == "{}/{}".format(a, b)
        new_string = "\\frac{" + str(a) + "}{" + str(b) + "}"
        return new_string
    except:
        return string

def _remove_right_units(string):
    # "\\text{ " only ever occurs (at least in the val set) when describing units
    if "\\text{ " in string:
        splits = string.split("\\text{ ")
        assert len(splits) == 2
        return splits[0]
    else:
        return string

def _fix_sqrt(string):
    if "\\sqrt" not in string:
        return string
    splits = string.split("\\sqrt")
    new_string = splits[0] 
    for split in splits[1:]:
        if split[0] != "{":
            a = split[0]
            new_substr = "\\sqrt{" + a + "}" + split[1:]
        else:
            new_substr = "\\sqrt" + split
        new_string += new_substr
    return new_string

def _strip_string(string):
    # linebreaks  
    string = string.replace("\n", "")
    #print(string)

    # remove inverse spaces
    string = string.replace("\\!", "")
    #print(string)

    # replace \\ with \
    string = string.replace("\\\\", "\\")
    #print(string)

    # replace tfrac and dfrac with frac
    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")
    #print(string)

    # remove \left and \right
    string = string.replace("\\left", "")
    string = string.replace("\\right", "")
    #print(string)
    
    # Remove circ (degrees)
    string = string.replace("^{\\circ}", "")
    string = string.replace("^\\circ", "")

    # remove dollar signs
    string = string.replace("\\$", "")
    
    # remove units (on the right)
    string = _remove_right_units(string)

    # remove percentage
    string = string.replace("\\%", "")
    string = string.replace("\%", "")

    # " 0." equivalent to " ." and "{0." equivalent to "{." Alternatively, add "0" if "." is the start of the string
    string = string.replace(" .", " 0.")
    string = string.replace("{.", "{0.")
    # if empty, return empty string
    if len(string) == 0:
        return string
    if string[0] == ".":
        string = "0" + string

    # to consider: get rid of e.g. "k = " or "q = " at beginning
    if len(string.split("=")) == 2:
        if len(string.split("=")[0]) <= 2:
            string = string.split("=")[1]

    # fix sqrt3 --> sqrt{3}
    string = _fix_sqrt(string)

    # remove spaces
    string = string.replace(" ", "")

    # \frac1b or \frac12 --> \frac{1}{b} and \frac{1}{2}, etc. Even works with \frac1{72} (but not \frac{72}1). Also does a/b --> \\frac{a}{b}
    string = _fix_fracs(string)

    # manually change 0.5 --> \frac{1}{2}
    if string == "0.5":
        string = "\\frac{1}{2}"

    # NOTE: X/Y changed to \frac{X}{Y} in dataset, but in simple cases fix in case the model output is X/Y
    string = _fix_a_slash_b(string)

    return string

def extract_model_answer(answer):
    if answer.endswith("."):
        answer = answer[:-1]

    if answer.startswith("$$") and answer.endswith("$$"):
        answer = answer[2:-2]
    if answer.startswith("$") and answer.endswith("$"):
        answer = answer[1:-1]
    if answer.startswith("\\[") and answer.endswith("\\]"):
        answer = answer[2:-2]
    if answer.startswith("\\(") and answer.endswith("\\)"):
        answer = answer[2:-2]
    if answer.startswith("\\text{") and answer.endswith("}"):
        answer = answer[6:-1]

    if "=" in answer:
        answer = answer.split("=")[1]

    answer = answer.replace("∞", "\\infty")
    
    return answer

def frac_compare(str1, str2):
    if "\\frac{" in str1:
        str1 = latex_frac_to_simple_frac(str1)
    if "\\frac{" in str2:
        str2 = latex_frac_to_simple_frac(str2)
    equal = None
    try:
        # Convert both strings to fractions.
        # Fractions can handle integer, float, and fraction strings.
        num1 = Fraction(str1)
        num2 = Fraction(str2)
        # Compare the two numbers for equality.
        num1 = float(num1)
        num2 = float(num2)
        if abs(num1 - num2) < 1e-4:
            equal = True
        else:
            equal = False
    except ValueError:
        # If a ValueError is raised, one of the strings might not be a valid number.
        pass

    return equal


def latex_frac_to_simple_frac(latex_str):
    # Pattern to match LaTeX fractions: \frac{numerator}{denominator}
    pattern = r'\\frac\{([^}]+)\}\{([^}]+)\}'
    
    # Replacement function to convert LaTeX fraction to simple fraction
    def repl(m):
        numerator, denominator = m.groups()
        return f"{numerator}/{denominator}"
    
    # Replace all LaTeX fraction patterns in the input string
    return re.sub(pattern, repl, latex_str)

def remove_spaces(s):
    if len(s.split()) == 1:
        s = s.replace(",\\!", "")
    s = s.replace(" ", "").replace("\t", "").replace("\n", "")
    if "\\begin{pmatrix}" in s:
        s = s.replace("\\frac{", "")
        s = s.replace("}{", "/")
        s = s.replace("}", "")
    return s

def is_equiv(str1, str2, verbose=False):
    if remove_spaces(str1).lower() == remove_spaces(str2).lower():
        return True
    if str1 is None and str2 is None:
        print("WARNING: Both None")
        return True
    if str1 is None or str2 is None:
        return False

    if "sqrt(" in str1 or "sqrt(" in str2:
        try:
            str1_v = eval(str1.replace("sqrt(", "math.sqrt("))
            str2_v = eval(str2.replace("sqrt(", "math.sqrt("))
            if abs(str1_v - str2_v) < 1e-3:
                return True
        except:
            pass

    try:
        ss1 = _strip_string(str1)
        ss2 = _strip_string(str2)
        if verbose:
            print(ss1, ss2)
        frac_compare_result = frac_compare(ss1, ss2)
        if frac_compare_result is not None:
            return frac_compare_result
        return ss1 == ss2
    except:
        return str1 == str2

if __name__ == "__main__":
    pair_list = [
        ["$\\frac{4}{3}$", "\\frac34"],
        ["$\\frac{1}{11}$", "\\frac{1}{11}"],
        ["\\(\\frac{1}{11}\\)", "\\frac{1}{11}"],
        ["\\dfrac{1}{2}", "\\frac{1}{2}"],
        ["neither", "\\text{neither}"],
        ["499", "499.0"],
        ["\\(0.7647", "\\frac{13}{17}"],
        ["14400", "14,\!400"]
    ]

    for pair in pair_list:
        print("=====================================")
        model_answer = pair[0]
        correct_answer = pair[1]

        print(model_answer)
        print(correct_answer)

        model_answer = extract_model_answer(model_answer)
        print(model_answer)
        correct_answer = extract_model_answer(correct_answer)
        print(correct_answer)

        print(is_equiv(model_answer, correct_answer))
        print("=====================================")