import io
import sys
import contextlib
import signal
import traceback
import resource
import multiprocessing as mp
from utils import clean_trace_error_message
from tqdm import tqdm


class TimeoutException(Exception):
    pass

@contextlib.contextmanager
def time_limit(seconds):
    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")

    signal.setitimer(signal.ITIMER_REAL, seconds)
    signal.signal(signal.SIGALRM, signal_handler)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)

code_to_run = '''
import math

def calculate_distance(point1, point2):
    """Calculate the distance between two points."""
    return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

def calculate_angle(a, b, c):
    """Calculate the angle between three points a, b, and c. Angle at point b."""
    ba = (a[0] - b[0], a[1] - b[1])
    bc = (c[0] - b[0], c[1] - b[1])
    cosine_angle = (ba[0] * bc[0] + ba[1] * bc[1]) / (math.sqrt(ba[0]**2 + ba[1]**2) * math.sqrt(bc[0]**2 + bc[1]**2))
    return math.acos(cosine_angle) * (180.0 / math.pi)

def analyze_shape(properties):
    """Analyze the geometric properties to confirm the shape's characteristics."""
    points = properties["points"]
    sides_lengths = []
    angles = []
    
    for i in range(len(points)):
        next_index = (i + 1) % len(points)
        sides_lengths.append(calculate_distance(points[i], points[next_index]))
        
        if i > 0:
            prev_index = i - 1
            angles.append(calculate_angle(points[prev_index], points[i], points[next_index]))
    
    # Adding the angle between the last, first, and second points to complete the cycle
    angles.append(calculate_angle(points[-2], points[-1], points[0]))
    
    return {
        "shape": properties["shape"],
        "sides": properties["sides"],
        "closed": properties["closed"],
        "sides_lengths": sides_lengths,
        "angles": angles
    }


input_properties = {
    "shape": "Heptagon",
    "sides": 7,
    "closed": True,
    "points": [(55.57, 80.69), (57.38, 65.80), (48.90, 57.46), (45.58, 47.78), (53.25, 36.07), (66.29, 48.90), (78.69, 61.09)]
}

output_properties = analyze_shape(input_properties)
print(output_properties)
'''

code_to_run2 = """
def count_words(word_list):
    return len(word_list)

def sort_words(word_list):
    count = count_words(word_list)
    return sorted(word_list)


#input_list = input()
input_list = '["banana", "cherry", "apple"]'
sorted_list = sort_words(eval(input_list))
print(sorted_list)
"""

code_to_run3 = """
def count_words(word_list):
    return len(word_list)

def sort_words(word_list):
    count = count_words(word_list)
    return sorted(word_list)


input_list = input()
#input_list = '["banana", "cherry", "apple"]'
sorted_list = sort_words(eval(input_list))
print(sorted_list)
"""

code_to_run4 = """
import ast

def evaluate_expressions(expressions):
    results = []
    for expr in expressions:
        try:
            if eval(expr) == 24:
                results.append(expr)
        except ZeroDivisionError:
            continue
    return results


input_data = input("Enter the list of expressions: ")
expressions = ast.literal_eval(input_data)
valid_expressions = evaluate_expressions(expressions)
print(valid_expressions)
"""

code_to_run5 = """
num_list = []
while True:
    num_list.append(list(range(100000)))
    x = 1234 * 1234
"""

def execute_inner(code_to_run: str, timeout: int) -> str:
    #set_memory_limit(500)  # Set the memory limit for the process
    output_buffer = io.StringIO()
    with time_limit(timeout):
        # Redirect stdout to the buffer
        with contextlib.redirect_stdout(output_buffer):
            # Create a string buffer
            local_scope = {}
            exec(code_to_run, local_scope, local_scope)

            # Get the output from the buffer
            captured_output = output_buffer.getvalue().strip()
            return captured_output


def execute_inner_with_input(code_to_run: str, single_input, timeout: int) -> str:
    #set_memory_limit(500)  # Set the memory limit for the process
    input_stream = io.BytesIO(single_input.encode())
    input_stream.seek(0)

    sys.stdin = io.TextIOWrapper(input_stream)

    output_buffer = io.StringIO()
    with time_limit(timeout):
        # Redirect stdout to the buffer
        with contextlib.redirect_stdout(output_buffer):
            # Create a string buffer
            local_scope = {}
            exec(code_to_run, local_scope, local_scope)

            # Get the output from the buffer
            captured_output = output_buffer.getvalue().strip()
            return captured_output

def get_params_for_mp(n_data, n_cores=50):
    #n_cores = mp.cpu_count()
    
    pool = mp.Pool(n_cores)
    avg = n_data // n_cores

    range_list = []
    start = 0
    for i in range(n_cores):
        num = avg + 1 if i < n_data - avg * n_cores else avg
        range_list.append([start, start + num])
        start += num

    return n_cores, pool, range_list

def call_solver_single_thread(inputs):
    data, pid = inputs
    results = []

    cnt = 0
    for instance in tqdm(data):
        #print('pid %d: %d/%d' % (pid, cnt, len(data)))
        res = execute_inner_with_input(instance["program"], instance["input"], 60)
        results.append(res)
        cnt += 1

    print('pid %d done' % pid)
    return results

def call_solver_multi_thread(data):
    n_cores = 3
    n_cores, pool, range_list = get_params_for_mp(len(data), n_cores)
    results = pool.map(call_solver_single_thread, zip([data[i[0]:i[1]] for i in range_list],
                                                       range(n_cores)))
    merged_result = []
    for res in results:
        merged_result.extend(res)

    return merged_result

if __name__ == "__main__":
    """
    my_program = code_to_run
    try:
        print("=====")
        print(execute_inner(my_program, 2))
        print("=====")
    except Exception as e:
        error_message = traceback.format_exc()
        cleaned_error_message = clean_trace_error_message(error_message)
        print("-----")
        print(cleaned_error_message)
        print("-----")
    
    my_program = code_to_run2

    try:
        print("=====")
        print(execute_inner(my_program, 2))
        print("=====")
    except Exception as e:
        error_message = traceback.format_exc()
        cleaned_error_message = clean_trace_error_message(error_message)
        print("-----")
        print(cleaned_error_message)
        print("-----")

    my_program = code_to_run3

    try:
        print("=====")
        print(execute_inner_with_input(my_program, '["banana", "cherry", "apple"]\n["abc", "xyz", "efg"]\n', 2))
        print("=====")
    except Exception as e:
        error_message = traceback.format_exc()
        cleaned_error_message = clean_trace_error_message(error_message)
        print("-----")
        print(cleaned_error_message)
        print("-----")
    

    my_program = code_to_run3

    try:
        print("=====")
        print(execute_inner_with_input(my_program, '["abc", "xyz", "efg"]\n', 2))
        print("=====")
    except Exception as e:
        error_message = traceback.format_exc()
        cleaned_error_message = clean_trace_error_message(error_message)
        print("-----")
        print(cleaned_error_message)
        print("-----")
    """

    """
    my_program = code_to_run4

    try:
        print("=====")
        print(execute_inner_with_input(my_program, "['(7/7)/4/1', '((7/7)/4)/1', '7/(7/4/1)', '7/((7/4)/1)', '3*7+2+1']\n", 3))
        print("=====")
    except Exception as e:
        error_message = traceback.format_exc()
        cleaned_error_message = clean_trace_error_message(error_message)
        print("-----")
        print(cleaned_error_message)
        print("-----")
    """
    data = [
        {
            "program": code_to_run5,
            "input": ""
        },
        {
            "program": code_to_run5,
            "input": ""
        },
        {
            "program": code_to_run5,
            "input": ""
        },
    ]

    call_solver_multi_thread(data)


    
