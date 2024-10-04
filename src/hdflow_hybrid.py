from __future__ import annotations
import getpass
import os
import json
import copy
from collections import defaultdict, deque
from tqdm import tqdm
import argparse
import time

from utils import extract_pattern_content, clean_result, get_params_for_mp
from experts import extract_expert_list, LLMExpert, ToolExpert, pack_current_trajectory
from model_api import ChatGPTOpenAI, ClaudeAnthropic, ChatVLLM, select_llm_model
from load_datasets import read_BBH_dataset, read_MATH_dataset, read_GameOf24_dataset, read_math_reasoning_dataset


def solve_problem(instance):
    output_trajectory = []
    
    if instance["solution_name"] == "hybrid_thinking_after":
        fast_thinking_trajectory = instance["fast_thinking_instance"]["output_trajectory"]
        
        if fast_thinking_trajectory[-1]["expert"] == "Meta-Expert@FINAL_JUDGEMENT":
            if fast_thinking_trajectory[-1]["output"] in ["NO"]:
                mode_decision = "slow"
            else:
                mode_decision = "fast"
            
            output_trajectory = fast_thinking_trajectory[-1]

            output_trajectory = [
                {
                    "expert": "Meta-Expert@FAST_SLOW",
                    "messages_list": fast_thinking_trajectory[-1]["messages_list"],
                    "entire_output": "",
                    "output": mode_decision
                }
            ]
            
        else:
            llm_chain = select_llm_model(instance["llm_model"])

            hybrid_prompt = open("prompts/hybrid_thinking_after_prompt.txt", "r").read()
            prompt_dict = copy.deepcopy(instance["prompt_dict"])

            prompt_dict["model_answer"] = instance["fast_thinking_instance"]["output_trajectory"][-1]["entire_output"]

            for k in prompt_dict:
                assert "{"+k+"}" in hybrid_prompt
                hybrid_prompt = hybrid_prompt.replace('{'+k+'}', prompt_dict[k])
            
            fast_slow_result = llm_chain.invoke(hybrid_prompt)

            print("=========== Fast or Slow ============")
            print(hybrid_prompt)
            print("-------------------------------------")
            print(fast_slow_result)
            print("=====================================")

            if "FINAL ASSESSMENT: YES" in fast_slow_result or " YES**" in fast_slow_result:
                mode_decision = "fast"
            elif "FINAL ASSESSMENT: NO" in fast_slow_result or " NO**" in fast_slow_result:
                mode_decision = "slow"
            else:
                mode_decision = "fast"

            output_trajectory = [
                {
                    "expert": "Meta-Expert@FAST_SLOW",
                    "messages_list": [[{"role": "user", "content": hybrid_prompt}, {"role": "assistant", "content": fast_slow_result}]],
                    "entire_output": fast_slow_result,
                    "output": mode_decision
                }
            ]
        
        if mode_decision == "fast":
            output_trajectory += instance["fast_thinking_instance"]["output_trajectory"]
        elif mode_decision == "slow":
            output_trajectory += instance["slow_thinking_instance"]["output_trajectory"]


    new_instance = copy.deepcopy(instance)
    new_instance["output_trajectory"] = output_trajectory

    with open(instance["output_path"], "w") as f:
        json.dump(new_instance, f, indent=4)
    
    return new_instance
    
    
def call_solver_single_thread(inputs):
    data, pid = inputs
    results = []

    cnt = 0
    for instance in tqdm(data):
        #print('pid %d: %d/%d' % (pid, cnt, len(data)))
        res = solve_problem(instance)
        results.append(res)
        cnt += 1

    print('pid %d done' % pid)
    return results

def call_solver_multi_thread(data):
    if "claude" in data[0]["llm_model"]:
        n_cores = 2
    elif "_azure" in data[0]["llm_model"]:
        n_cores = 20
    else:
        n_cores = 50
    n_cores, pool, range_list = get_params_for_mp(len(data), n_cores)
    results = pool.map(call_solver_single_thread, zip([data[i[0]:i[1]] for i in range_list],
                                                       range(n_cores)))
    merged_result = []
    for res in results:
        merged_result.extend(res)

    return merged_result

def detect_broken_file(f):
    result = False
    data = json.load(open(f))
    output_trajectory = data["output_trajectory"]
    for item in output_trajectory:
        expert = item["expert"]
        messages_list = item["messages_list"]
        for messages in messages_list:
            assert len(messages) == 2
            assert messages[0]["role"] == "user"
            litgpt_instruction = messages[0]["content"]
            assert messages[1]["role"] == "assistant"
            litgpt_output = messages[1]["content"]
            if len(litgpt_output) == 0:
                result = True
                break
    return result
            
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--api_model", default="gpt-4o-mini-2024-07-18", type=str, help="gpt-4-0125-preview_azure, gpt-4-turbo-2024-04-09, gpt-4o-2024-05-13, claude-3-opus-20240229")
    parser.add_argument("--solution_name", default="hybrid_thinking_after", type=str, help="solution name")
    parser.add_argument("--fast_suffix", default="", type=str, help="fast extra suffix")
    parser.add_argument("--slow_suffix", default="", type=str, help="slow extra suffix")
    parser.add_argument("--suffix", default="", type=str, help="experiment extra suffix")
    parser.add_argument("--max_num", default="1", type=str, help="max instance number of each subtask")
    parser.add_argument("--dataset", default="BBH", type=str, help="BBH, MATH, GameOf24, etc.")
    parser.add_argument("--overwrite", default="False", type=str, help="overwrite existing output files or not")
    args = parser.parse_args()

    if not args.solution_name:
        raise ValueError("Please provide a solution name")

    if args.dataset == "BBH":
        dataset = read_BBH_dataset()
    elif args.dataset == "MATH":
        dataset = read_MATH_dataset()
    elif args.dataset == "MATH_sampled":
        dataset = read_MATH_dataset("sampled")
    elif args.dataset == "math_reasoning":
        dataset = read_math_reasoning_dataset()
    elif args.dataset == "GameOf24":
        dataset = read_GameOf24_dataset()

    if "/" in args.api_model:
        fields = args.api_model.split("/")
        model_name = fields[-3]+"_"+fields[-2]
    else:
        model_name = args.api_model

    output_dir = f"../output/{args.dataset}/{args.solution_name}__{model_name}"

    fast_thinking_output_dir = f"../output/{args.dataset}/fast_thinking__{model_name}"
    slow_thinking_output_dir = f"../output/{args.dataset}/slow_thinking__{model_name}"

    if args.fast_suffix != "":
        fast_thinking_output_dir += f"__{args.fast_suffix}"
    if args.slow_suffix != "":
        slow_thinking_output_dir += f"__{args.slow_suffix}"
    
    if args.suffix != "":
        output_dir += f"__{args.suffix}"
    os.makedirs(output_dir, exist_ok=True)

    for subtask in dataset:
        if args.dataset == "MATH" and subtask == "all":
            continue
        os.makedirs(f"{output_dir}/{subtask}", exist_ok=True)

    instance_list = []

    for subtask in dataset:
        subtask_name = " ".join(subtask.split("_"))

        task_description = dataset[subtask]["description"]
        task_format = dataset[subtask]["format"]
        subtask_instances = dataset[subtask]["instances"]

        print("len(subtask_instances):", len(subtask_instances))

        if args.max_num == "all":
            MAX_NUM = len(subtask_instances)
        else:
            MAX_NUM = int(args.max_num)
        
        #for i in range(0, len(subtask_instances)):
        for i in range(0, MAX_NUM):
            instance = subtask_instances[i]
            path_subtask = instance["Id"].split("@")[0]
            path_idx = instance["Id"].split("@")[1]
            output_path = f"{output_dir}/{path_subtask}/{path_idx}_output_instance.json"
            if os.path.exists(output_path) and detect_broken_file(output_path):
                print(f"Broken file {output_path}...")
                os.remove(output_path)
                #input("continue?")

            fast_thinking_output_file = f"{fast_thinking_output_dir}/{path_subtask}/{path_idx}_output_instance.json"
            slow_thinking_output_file = f"{slow_thinking_output_dir}/{path_subtask}/{path_idx}_output_instance.json"
            fast_thinking_instance = {}
            slow_thinking_instance = {}
            if os.path.exists(fast_thinking_output_file):
                with open(fast_thinking_output_file, "r") as f:
                    fast_thinking_instance = json.load(f)
            else:
                print(f"Fast thinking file {fast_thinking_output_file} does not exist...")
                exit()
            if os.path.exists(slow_thinking_output_file):
                with open(slow_thinking_output_file, "r") as f:
                    slow_thinking_instance = json.load(f)
            else:
                print(f"Slow thinking file {slow_thinking_output_file} does not exist...")
                exit()
            
            assert fast_thinking_instance["target"] == slow_thinking_instance["target"]

            if os.path.exists(output_path) and args.overwrite == "False":
                print(f"Skip {output_path}...")
                continue
            if i == 0:
                example_input = subtask_instances[1]["input"]
                example_target = subtask_instances[1]["target"]
            else:
                example_input = subtask_instances[0]["input"]
                example_target = subtask_instances[0]["target"]
            
            prompt = ""
            if task_description != "":
                prompt += f"Task Name: {subtask_name}\n\nTask Description:\n{task_description}\n\n"
            prompt += f"Problem to answer:\n{instance['input']}"
            if task_format != "":
                prompt += f"\n\n{task_format}"
            
            
            instance = {
                "task_name": subtask_name,
                "task_description": task_description,
                "task_format": task_format,
                "llm_model": args.api_model,
                "solution_name": args.solution_name,
                "Id": instance["Id"], 
                "prompt_dict": {"task_problem": prompt},
                "input": instance["input"],
                "target": instance["target"],
                "example_input": example_input,
                "example_target": example_target,
                "output_path": output_path,
                "fast_thinking_instance": fast_thinking_instance,
                "slow_thinking_instance": slow_thinking_instance
            }
            instance_list.append(instance)
    
    time.sleep(10)
    
    results = call_solver_multi_thread(instance_list)

    

    for result in results:
        Id = result["Id"]
        subtask = Id.split("@")[0]
        idx = Id.split("@")[1]
        output_trajectory = result["output_trajectory"]
        print(output_trajectory[-1]["output"])

        # save to json file
        with open(f"{output_dir}/{subtask}/{idx}_output_instance.json", "w") as f:
            json.dump(result, f, indent=4)
        with open(f"{output_dir}/{subtask}/{idx}_output_trajectory_readable.txt", "w") as f:
            for item in output_trajectory:
                f.write(f"\n====================================== {item['expert']} START ====================\n")
                f.write("----------------------------------------\n\n")
                for messages in item["messages_list"]:
                    for message in messages:
                        f.write(f"{message['role']}:\n{message['content']}\n\n")
                        f.write("----------------------------------------\n\n")
                if "Python Execution" in item["entire_output"]:
                    f.write(f"{item['entire_output']}\n")
                    f.write("----------------------------------------\n\n")
                f.write(f"\n====================================== {item['expert']} END ====================\n\n\n")
            
            f.write(f'Target Answer:\n{result["target"]}\n\n')
