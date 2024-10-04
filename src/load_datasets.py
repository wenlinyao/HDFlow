from datasets import Dataset, load_dataset, get_dataset_config_names
import copy
import json
import re
import os
import pandas as pd
import random
from math_equivalence import last_boxed_only_string, remove_box
from utils import save_jsonl_file, load_jsonl_file, extract_pattern_content, clean_result


hf_cache_dir = "../data/hf_datasets"
if not os.path.exists(hf_cache_dir):
    os.mkdir(hf_cache_dir)

def load_csv_file(file_name, sep):
    df = pd.read_csv(file_name, sep=sep)
    return df

def read_MathInstruct_dataset(shards):
    data_path = "../data/MathInstruct"
    # load json file
    dataset = {}
    for shard in shards:
        dataset[f"MathInstruct_clean_part{shard}"] = {"description": "", "format": "", "instances": []}
        data = json.load(open(f"{data_path}/MathInstruct_clean_part{shard}.json", "r"))
        for instance in data:
            new_instance = {}
            new_instance["Id"] = f"MathInstruct_clean_part{shard}@" + instance["Id"]
            new_instance["input"] = instance["new_instruction"]
            new_instance["target"] = ""
            dataset[f"MathInstruct_clean_part{shard}"]["instances"].append(new_instance)
    return dataset

    

def read_MetaMathQA_dataset(shards, math_type):
    data_path = "../data/MetaMathQA"
    dataset = {}
    total_count = 0
    for shard in shards:
        data = json.load(open(f"{data_path}/MetaMathQA-395K_{shard}.json", "r"))
        
        dataset[f"MetaMathQA-395K_{shard}"] = {"description": "", "format": "The final answer should be either a number or a math expression.", "instances": []}
        
        for i in range(len(data)):
            if data[i]["type"].startswith(math_type) == False:
                continue
            instance = {}
            instance["Id"] = f"MetaMathQA-395K_{shard}@{i}"
            instance["input"] = data[i]["query"]
            instance["target"] = data[i]["response"].split("The answer is: ")[-1].strip()
            instance["meta_info"] = {
                "type": data[i]["type"],
                "original_question": data[i]["original_question"],
                "response": data[i]["response"],
            }
            dataset[f"MetaMathQA-395K_{shard}"]["instances"].append(instance)
            total_count += 1
    print(f"Total instances: {total_count}")
    input("continue?")
    return dataset


def read_BBH_dataset():
    cache_dir = hf_cache_dir

    bbh_dataset_config_names = get_dataset_config_names("lukaemon/bbh", cache_dir=cache_dir)

    bbh_dataset_name2description = {}
    bbh_dataset_name2format = {}
    
    with open("prompts/BBH_README.md", "r") as f:
        count = -1
        for line in f:
            if not line.strip():
                continue
            if line.startswith("## "):
                count += 1
                subtask_name = line[3:].strip()
            subtask_lower_name = "_".join(subtask_name.split()).lower()
            bbh_dataset_name = bbh_dataset_config_names[count]
            #assert (subtask_lower_name in bbh_dataset_name) or (bbh_dataset_name in subtask_lower_name)
            print(subtask_lower_name, bbh_dataset_name)
            bbh_dataset_name2description[bbh_dataset_name] = line.strip()
    
    with open("prompts/BBH_format.md", "r") as f:
        count = -1
        for line in f:
            if not line.strip():
                continue
            if line.startswith("## "):
                count += 1
                subtask_name = line[3:].strip()
            subtask_lower_name = "_".join(subtask_name.split()).lower()
            bbh_dataset_name = bbh_dataset_config_names[count]
            #assert (subtask_lower_name in bbh_dataset_name) or (bbh_dataset_name in subtask_lower_name)
            print(subtask_lower_name, bbh_dataset_name)
            bbh_dataset_name2format[bbh_dataset_name] = line.strip()
    
    bbh_dataset = {}
    for subtask in bbh_dataset_config_names:
        subtask_data = load_dataset("lukaemon/bbh", subtask, split="test", cache_dir=cache_dir)
        bbh_dataset[subtask] = {"description": bbh_dataset_name2description[subtask], "format": bbh_dataset_name2format[subtask], "instances": []}
        for i in range(len(subtask_data)):
            instance = copy.deepcopy(subtask_data[i])
            instance["Id"] = subtask + "@" + str(i)
            bbh_dataset[subtask]["instances"].append(instance)
    return bbh_dataset

def read_math_reasoning_dataset_train(subtask_max):
    cache_dir = hf_cache_dir
    config_names = get_dataset_config_names("math_dataset", cache_dir=cache_dir)
    dataset_name2format = {}
    with open("prompts/math_reasoning_format.md", "r") as f:
        count = -1
        for line in f:
            if not line.strip():
                continue
            if line.startswith("## "):
                count += 1
                subtask_name = line[3:].strip()
            dataset_name2format[subtask_name] = line.strip()
    
    dataset = {}
    for subtask in config_names:
        subtask_data = load_dataset("math_dataset", subtask, split="train", cache_dir=cache_dir)
        dataset[subtask] = {
            "description": "This is a math reasoning problem.",
            "format": dataset_name2format[subtask],
            "instances": []
        }
        for i in range(subtask_max):
            instance = {}
            instance["Id"] = subtask + "@" + str(i)
            question = subtask_data[i]["question"]
            if question.startswith("b'"):
                question = question[2:]
            if question.endswith("\\n'"):
                question = question[:-3]
            answer = subtask_data[i]["answer"]
            if answer.startswith("b'"):
                answer = answer[2:]
            if answer.endswith("\\n'"):
                answer = answer[:-3]
            # print('###'+question+'###')
            # print('###'+answer+'###')
            # input("continue?")
            instance["input"] = question
            instance["target"] = answer
            dataset[subtask]["instances"].append(instance)
    return dataset

def read_math_reasoning_dataset():
    cache_dir = hf_cache_dir
    config_names = get_dataset_config_names("math_dataset", cache_dir=cache_dir)
    dataset_name2format = {}
    with open("prompts/math_reasoning_format.md", "r") as f:
        count = -1
        for line in f:
            if not line.strip():
                continue
            if line.startswith("## "):
                count += 1
                subtask_name = line[3:].strip()
            dataset_name2format[subtask_name] = line.strip()
    
    dataset = {}
    for subtask in config_names:
        subtask_data = load_dataset("math_dataset", subtask, split="test", cache_dir=cache_dir)
        dataset[subtask] = {
            "description": "This is a math reasoning problem.",
            "format": dataset_name2format[subtask],
            "instances": []
        }
        for i in range(len(subtask_data)):
            instance = {}
            instance["Id"] = subtask + "@" + str(i)
            question = subtask_data[i]["question"]
            if question.startswith("b'"):
                question = question[2:]
            if question.endswith("\\n'"):
                question = question[:-3]
            answer = subtask_data[i]["answer"]
            if answer.startswith("b'"):
                answer = answer[2:]
            if answer.endswith("\\n'"):
                answer = answer[:-3]
            # print('###'+question+'###')
            # print('###'+answer+'###')
            # input("continue?")
            instance["input"] = question
            instance["target"] = answer
            dataset[subtask]["instances"].append(instance)
    return dataset


def read_MATH_dataset(flag="all"):
    cache_dir = hf_cache_dir
    config_names = get_dataset_config_names("lighteval/MATH", cache_dir=cache_dir)
    dataset = {}
    for subtask in config_names:
        if subtask == "all":
            continue
        subtask_data = load_dataset("lighteval/MATH", subtask, split="test", cache_dir=cache_dir)
        
        dataset[subtask] = {"description": "This is a math competition problem.", "format": "The final answer should be either a number or a math expression.", "instances": []}
        
        for i in range(len(subtask_data)):
            instance = {}
            instance["Id"] = subtask + "@" + str(i)
            instance["input"] = subtask_data[i]["problem"]
            # use re to extract the string in "\boxed{xxx}"
            final_answer = last_boxed_only_string(subtask_data[i]["solution"])
            final_answer = remove_box(final_answer)

            if final_answer is None:
                print("Error in extracting target answer...")
                continue

            instance["target"] = final_answer
            
            instance["meta_info"] = {
                "level": subtask_data[i]["level"],
                "type": subtask_data[i]["type"],
                "solution": subtask_data[i]["solution"]
            }
            dataset[subtask]["instances"].append(instance)

    all_instances = []
    for subtask in dataset:
        all_instances.extend(dataset[subtask]["instances"])
    random.seed(11)
    random.shuffle(all_instances)
    sampled_dataset = {}
    for subtask in dataset:
        sampled_dataset[subtask] = {"description": dataset[subtask]["description"], "format": dataset[subtask]["format"], "instances": []}
    for instance in all_instances[:500]:
        subtask = instance["Id"].split("@")[0]
        sampled_dataset[subtask]["instances"].append(instance)
    
    if flag == "sampled":
        for subtask in sampled_dataset:
            print(f"Subtask: {subtask}, instances: {len(sampled_dataset[subtask]['instances'])}")
        return sampled_dataset
    else:
        return dataset



def read_GameOf24_dataset():
    cache_dir = "../data"
    csv_data = pd.read_csv(f"{cache_dir}/GameOf24.csv")
    print(csv_data.columns)
    print(csv_data.shape)
    print(csv_data.head())

    dataset = {}
    subtasks = ["game_of_24_level1", "game_of_24_level2", "game_of_24_level3"]
    for subtask in subtasks:
        dataset[subtask] = {
            "description": "Use numbers and basic arithmetic operations (+ - * /) to obtain 24. It uses each input number exactly once and no other numbers to reach 24.",
            "format": "The final answer should be a math expression that uses the four input numbers to reach 24.", # You only need to show the first one if there are multiple solutions.
            "instances": []
        }
    
    rank_list = csv_data["Rank"].tolist()
    puzzles_list = csv_data["Puzzles"].tolist()
    amt_list = csv_data["AMT (s)"].tolist()
    solved_rate_list = csv_data["Solved rate"].tolist()

    for i in range(len(rank_list)):
        if 0 <= i < 500:
            subtask = subtasks[0]
        elif 500 <= i < 1000:
            subtask = subtasks[1]
        else:
            subtask = subtasks[2]

        instance = {}
        instance["Id"] = subtask + "@" + str(i)
        instance["input"] = f"Here are the four numbers to use: {puzzles_list[i]}"
        instance["target"] = "24"
        instance["meta_info"] = {
            "rank": rank_list[i],
            "amt": amt_list[i],
            "solved_rate": solved_rate_list[i]
        }
        dataset[subtask]["instances"].append(instance)
    return dataset
