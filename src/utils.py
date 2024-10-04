import json
import random
import glob
import re
import multiprocessing as mp

def save_jsonl_file(instanceList, file_name):
    with open(file_name, 'w', encoding='utf8') as f:
        for instance in instanceList:
            try:
                json.dump(instance, f, ensure_ascii=False)
                f.write('\n')
            except:
                print("Error in save_jsonl_file")
                print(instance)
                continue

def load_jsonl_file(file_name):
    instanceList = []
    with open(file_name, 'r', encoding='utf8') as f:
        for line in f:
            #print(line)
            instanceList.append(json.loads(line))
    return instanceList

def extract_score(answer_str):
    # use re to extact the value in "[[3]]"
    results = re.findall(r"\[\[(\d+)\]\]", answer_str)
    if len(results) == 0:
        return None
    else:
        return float(results[-1])

def extract_pattern_content(input_str, pattern_start, pattern_end):
    # change to literal characters
    pattern_start = re.escape(pattern_start)
    pattern_end = re.escape(pattern_end)
    # use re to extact the value between start_str and end_str
    results = re.findall(pattern_start+'(.*?)'+pattern_end, input_str, re.DOTALL)
    if len(results) == 0:
        return [""]
    return results

def clean_result(result_str):
    # remove spaces and newlines in the beginning and end of the string
    while result_str.startswith(" ") or result_str.startswith("\n"):
        result_str = result_str[1:]
    while result_str.endswith(" ") or result_str.endswith("\n"):
        result_str = result_str[:-1]
    if result_str.startswith("```json") and result_str.endswith("```"):
        result_str = result_str[7:-3]
    return result_str


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

def clean_trace_error_message(message):
    max_idx = 0
    lines = message.split('\n')
    for i, line in enumerate(lines):
        if 'File "<string>",' in line:
            max_idx = i
    return '\n'.join(lines[max_idx:])