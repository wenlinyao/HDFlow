from __future__ import annotations
import getpass
import os
import json
import copy
from collections import defaultdict, deque
from tqdm import tqdm
import argparse
import time
import random
from datetime import datetime

from utils import extract_pattern_content, clean_result, get_params_for_mp, save_jsonl_file, load_jsonl_file
from experts import extract_expert_list, LLMExpert, ToolExpert, pack_current_trajectory
from model_api import ChatGPTOpenAI, ClaudeAnthropic, ChatVLLM, select_llm_model
from load_datasets import read_BBH_dataset, read_MATH_dataset, read_GameOf24_dataset, read_math_reasoning_dataset, read_MetaMathQA_dataset, read_math_reasoning_dataset_train, read_MathInstruct_dataset


prompt_suffix = "_v2"


def pack_candidate_answers(answers):
    final_string = []
    for i, item in enumerate(answers):
        answer = item["answer"]
        problem_reflection = item["problem_reflection"]
        final_string.append(f"### Candidate {i+1} problem analysis start ###\n{problem_reflection}\n### Candidate {i+1} problem analysis end ###")
        final_string.append(f"### Candidate {i+1} answer start ###\n{answer}\n### Candidate {i+1} answer end ###")
    
    return "\n\n".join(final_string)

def slow_thinking(instance):
    llm_chain = select_llm_model(instance["llm_model"])

    final_output_trajectory = []
    iteration_count = 0
    answer_candidates = []

    iteration_max = instance["iter"]

    while True:
        print("Iteration:", iteration_count)
        prompt_dict = copy.deepcopy(instance["prompt_dict"])
        
        ######################
        # problem reflection #
        ######################
        reflection_prompt = open(f"prompts/reflection_prompt{prompt_suffix}.txt", "r").read()
        for k in prompt_dict:
            reflection_prompt = reflection_prompt.replace('{'+k+'}', prompt_dict[k])
        #llm_chain = llm | StrOutputParser()
        reflection_result = llm_chain.invoke(reflection_prompt, temperature=instance["temperature"])
        print(instance)
        print()
        print(reflection_result)
        problem_reflection_list = extract_pattern_content(reflection_result, "### Problem Reflection start ###\n", "### Problem Reflection end ###")
        problem_reflection_str = problem_reflection_list[0]

        ##################
        # experts design #
        ##################
        prompt_dict["problem_reflection"] = problem_reflection_str
        experts_design_prompt = open(f"prompts/experts_design_prompt{prompt_suffix}.txt", "r").read()
        for k in prompt_dict:
            experts_design_prompt = experts_design_prompt.replace('{'+k+'}', prompt_dict[k])
        
        experts_design_count = 0
        while True:
            experts_design_result = llm_chain.invoke(experts_design_prompt, temperature=instance["temperature"])
            print(experts_design_result)
            experts_design_list = extract_pattern_content(experts_design_result, "### Specialized Experts Design start ###\n", "### Specialized Experts Design end ###")
            experts_design_str = experts_design_list[0]

            expert_card_list, _ = extract_expert_list(experts_design_str, instance)

            if len(expert_card_list) != 0 or experts_design_count >= 5:
                break
            else:
                experts_design_count += 1

        #print(expert_card_list)

        experts = []

        for expert_card in expert_card_list:
            if "Expert_Type" not in expert_card:
                experts.append(LLMExpert(expert_card, instance["llm_model"], problem_reflection_str, f"prompts/expert_llm_prompt{prompt_suffix}.txt"))
            elif expert_card["Expert_Type"] == "LLM":
                experts.append(LLMExpert(expert_card, instance["llm_model"], problem_reflection_str, f"prompts/expert_llm_prompt{prompt_suffix}.txt"))
            elif expert_card["Expert_Type"] == "Tool":
                experts.append(ToolExpert(expert_card, instance["llm_model"], problem_reflection_str, f"prompts/expert_tool_prompt{prompt_suffix}.txt"))
        
        reflection_messages  = [{"role": "user", "content": reflection_prompt}, {"role": "assistant", "content": reflection_result}]
        experts_design_messages = [{"role": "user", "content": experts_design_prompt}, {"role": "assistant", "content": experts_design_result}]
        output_trajectory = [
                {
                    "expert": "Meta-Expert@REFLECTION",
                    "messages_list": [reflection_messages],
                    "entire_output": reflection_result,
                    "output": problem_reflection_str
                },
                {
                    "expert": "Meta-Expert@EXPERTS_DESIGN",
                    "messages_list": [experts_design_messages],
                    "entire_output": experts_design_result,
                    "output": experts_design_str
                }
            ]
        for expert in experts:
            print(expert.name + " processing...")
            messages_list, entire_output, output = expert.run(output_trajectory)
            output_trajectory.append({"expert": expert.name, "messages_list": messages_list, "entire_output": entire_output, "output": output})
        
        final_expert = output_trajectory[-1]["expert"]
        prompt_dict = copy.deepcopy(instance["prompt_dict"])
        prompt_dict["experts_design"] = experts_design_str
        experts_results = pack_current_trajectory(output_trajectory)
        prompt_dict["experts_results"] = experts_results
        prompt_dict["final_expert"] = final_expert
        prompt_dict["problem_reflection"] = problem_reflection_str
        
        final_judgement_prompt = open(f"prompts/final_judgement_prompt{prompt_suffix}.txt", "r").read()

        for k in prompt_dict:
            final_judgement_prompt = final_judgement_prompt.replace('{'+k+'}', prompt_dict[k])
        
        final_judgement_result = llm_chain.invoke(final_judgement_prompt, temperature=instance["temperature"])

        print(final_judgement_result)

        if "FINAL EVALUATION: YES" in final_judgement_result or " YES**" in final_judgement_result:
            final_answer = "YES"
        elif "FINAL EVALUATION: NO" in final_judgement_result or " NO**" in final_judgement_result:
            final_answer = "NO"
        else:
            final_answer = "UNKNOWN"
        

        output_trajectory.append(
            {
                "expert": "Meta-Expert@FINAL_JUDGEMENT",
                "messages_list": [[{"role": "user", "content": final_judgement_prompt}, {"role": "assistant", "content": final_judgement_result}]],
                "entire_output": final_judgement_result,
                "output": final_answer
            }
        )
        
        answer_candidates.append({
            "problem_reflection": problem_reflection_str,
            "answer": output_trajectory[-2]["output"],
        })
        final_output_trajectory += output_trajectory
        iteration_count += 1

        if final_answer == "YES":
            break
        if iteration_count >= iteration_max:
            break
    
    if iteration_max >= 2 and iteration_count >= iteration_max and final_answer != "YES":
        candidate_answers_str = pack_candidate_answers(answer_candidates)
        prompt_dict = copy.deepcopy(instance["prompt_dict"])
        prompt_dict["iteration_max"] = str(iteration_max)
        prompt_dict["candidate_answers"] = candidate_answers_str
        best_answer_prompt = open(f"prompts/best_answer_prompt{prompt_suffix}.txt", "r").read()
        for k in prompt_dict:
            best_answer_prompt = best_answer_prompt.replace('{'+k+'}', prompt_dict[k])
        best_answer_result = llm_chain.invoke(best_answer_prompt, temperature=instance["temperature"])

        print(best_answer_result)

        best_answer_idx = len(answer_candidates) - 1
        for i in range(len(answer_candidates)):
            #if f"BEST ANSWER: {i}" in best_answer_result:
            if f"BEST ANSWER: {i+1}" in best_answer_result:
                best_answer_idx = i
                break

        final_output_trajectory.append(
            {
                "expert": "Meta-Expert@BEST_ANSWER",
                "messages_list": [[{"role": "user", "content": best_answer_prompt}, {"role": "assistant", "content": best_answer_result}]],
                "entire_output": best_answer_result,
                "output": answer_candidates[best_answer_idx]["answer"]
            }
        )
        

    return final_output_trajectory

direct_prompt_template = "=====\n{task_problem}\n=====\n\nPlease answer the above problem and output your final answer starting with \"FINAL ANSWER:\""



def fast_thinking(instance):
    llm_chain = select_llm_model(instance["llm_model"])

    iteration_max = instance["iter"]

    iteration_count = 0

    final_output_trajectory = []

    while True:
        print("Iteration:", iteration_count)
        prompt_dict = copy.deepcopy(instance["prompt_dict"])

        instruction_prompt = direct_prompt_template
        expert_name = "Direct Answer Expert"

        for k in prompt_dict:
            instruction_prompt = instruction_prompt.replace('{'+k+'}', prompt_dict[k])
        
        instruction_result = llm_chain.invoke(instruction_prompt, temperature=instance["temperature"])

        if "**FINAL ANSWER:**" in instruction_result:
            final_answer = instruction_result.split("**FINAL ANSWER:**")[1].strip()
        elif "**FINAL ANSWER**:" in instruction_result:
            final_answer = instruction_result.split("**FINAL ANSWER**:")[1].strip()
        elif "**Final Answer**:" in instruction_result:
            final_answer = instruction_result.split("**Final Answer**:")[1].strip()
        elif "**Final Answer:**" in instruction_result:
            final_answer = instruction_result.split("**Final Answer:**")[1].strip()
        elif "FINAL ANSWER:" in instruction_result:
            final_answer = instruction_result.split("FINAL ANSWER:")[1].strip()
        else:
            final_answer = instruction_result
        
        print("=========== Fast Thinking ============")
        print(instruction_prompt)
        print("-------------------------------------")
        print(instruction_result)
        print("\nTARGET ANSWER:", instance["target"])
        print("=====================================")
        
        final_answer = clean_result(final_answer)

        final_output_trajectory += [
            {
                "expert": expert_name,
                "messages_list": [[{"role": "user", "content": instruction_prompt}, {"role": "assistant", "content": instruction_result}]],
                "entire_output": instruction_result,
                "output": final_answer
            }
        ]

        self_verify_prompt = open("prompts/hybrid_thinking_after_prompt.txt", "r").read()
        prompt_dict["model_answer"] = instruction_result

        for k in prompt_dict:
            self_verify_prompt = self_verify_prompt.replace('{'+k+'}', prompt_dict[k])
        
        self_verify_result = llm_chain.invoke(self_verify_prompt, temperature=instance["temperature"])

        if "FINAL ASSESSMENT: YES" in self_verify_result or " YES**" in self_verify_result:
            final_answer = "YES"
        elif "FINAL ASSESSMENT: NO" in self_verify_result or " NO**" in self_verify_result:
            final_answer = "NO"
        else:
            final_answer = "UNKNOWN"

        final_output_trajectory += [
            {
                "expert": "Meta-Expert@FINAL_JUDGEMENT",
                "messages_list": [[{"role": "user", "content": self_verify_prompt}, {"role": "assistant", "content": self_verify_result}]],
                "entire_output": self_verify_result,
                "output": final_answer
            }
        ]

        iteration_count += 1

        if final_answer == "YES":
            break
        if iteration_count >= iteration_max:
            break

    return final_output_trajectory

def solve_problem(instance):
    output_trajectory = []
    if instance["solution_name"] == "fast_thinking":
        output_trajectory = fast_thinking(instance)
    elif instance["solution_name"] == "slow_thinking":
        output_trajectory = slow_thinking(instance)

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
        n_cores = 10
    else:
        n_cores = 100
    
    n_cores, pool, range_list = get_params_for_mp(len(data), n_cores)
    results = pool.map(call_solver_single_thread, zip([data[i[0]:i[1]] for i in range_list],
                                                       range(n_cores)))
    merged_result = []
    for res in results:
        merged_result.extend(res)

    return merged_result

def detect_broken_file(f):
    result = False
    try:
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
                model_output = messages[1]["content"]
                if len(model_output) == 0 or model_output == "ERROR":
                    result = True
                    break
    # json decode error
    except:
        result = True
    return result
    
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--api_model", default="gpt-4o-mini-2024-07-18", type=str, help="gpt-4-0125-preview_azure, gpt-4-turbo-2024-04-09, gpt-4o-2024-05-13, claude-3-opus-20240229")
    parser.add_argument("--batch_infer", default="False", type=str, help="batch inference or not")
    parser.add_argument("--solution_name", default="", type=str, help="fast_thinking, slow_thinking")
    parser.add_argument("--suffix", default="", type=str, help="experiment extra suffix")
    parser.add_argument("--max_num", default="1", type=str, help="max instance number of each subtask")
    parser.add_argument("--iter", default=1, type=int, help="max iteration number for slow thinking")
    parser.add_argument("--dataset", default="BBH", type=str, help="BBH, MATH, GameOf24, etc.")
    parser.add_argument("--overwrite", default="False", type=str, help="overwrite existing output files or not")
    parser.add_argument("--temperature", default=0.2, type=float, help="temperature for generation")
    args = parser.parse_args()

    random.seed(11)

    print("####### prompt_suffix #######")
    print(f"############# {prompt_suffix} #############")
    print("#############################")
    #input("continue?")

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
    elif args.dataset == "math_reasoning_train":
        dataset = read_math_reasoning_dataset_train(subtask_max=50) # slow_thinking
    elif "MetaMathQA" in args.dataset: # MetaMathQA_0
        shards = args.dataset.split("_")[1:]
        dataset = read_MetaMathQA_dataset(shards=shards, math_type="MATH")
    elif "MathInstruct" in args.dataset:
        shards = args.dataset.split("_")[1:]
        dataset = read_MathInstruct_dataset(shards=shards)
    elif args.dataset == "GameOf24":
        dataset = read_GameOf24_dataset()

    if "/" in args.api_model:
        fields = args.api_model.split("/")
        model_name = fields[-3]+"_"+fields[-2]
    else:
        model_name = args.api_model

    output_dir = f"../output/{args.dataset}/{args.solution_name}__{model_name}"

    if args.suffix != "":
        output_dir += f"__{args.suffix}"
    if dataset != {}:
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
                #input("continue?")
                os.remove(output_path)
    
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
                "iter": args.iter,
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
                "temperature": args.temperature,
                "output_path": output_path
            }
            instance_list.append(instance)
    
    

    if args.batch_infer == "True" and args.solution_name == "fast_thinking":
        batch_instances = []
        for instance in instance_list:
            instruction_prompt = direct_prompt_template
            prompt_dict = instance["prompt_dict"]
            for k in prompt_dict:
                instruction_prompt = instruction_prompt.replace('{'+k+'}', prompt_dict[k])
            batch_instance = {
                "custom_id": instance["Id"],
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": instance["llm_model"],
                    "temperature": 0.1,
                    "messages": [
                        {"role": "system", "content": "You are ChatGPT, a large language model trained by OpenAI, based on the GPT-4 architecture."},
                        {"role": "user", "content": instruction_prompt}
                    ]
                }
            }
            batch_instances.append(batch_instance)
        os.makedirs("../batch_inference", exist_ok=True)
        now = datetime.now()
        date_time_string = now.strftime('%Y-%m-%d_%H-%M-%S')
        save_jsonl_file(instance_list, f"../batch_inference/{args.dataset}_fast_thinking_{date_time_string}_org.jsonl")
        save_jsonl_file(batch_instances, f"../batch_inference/{args.dataset}_fast_thinking_{date_time_string}.jsonl")
        print("Batch inference file saved...")
        exit()
    
    if args.batch_infer == "True" and args.solution_name == "fast_thinking_verify":
        out_data = load_jsonl_file(f"../batch_inference/{args.dataset}")
        org_data = load_jsonl_file(f"../batch_inference/{args.dataset.replace('_out', '_org')}")
        custom_id2out_instance = {}
        for out_instance in out_data:
            custom_id2out_instance[out_instance["custom_id"]] = out_instance
        batch_instances = []
        hybrid_prompt = open("prompts/hybrid_thinking_after_prompt.txt", "r").read()
        instance_list = org_data
        for instance in instance_list:
            instruction_prompt = hybrid_prompt
            prompt_dict = {}
            prompt_dict["task_problem"] = instance["prompt_dict"]["task_problem"]
            if instance["Id"] not in custom_id2out_instance:
                print(f"Instance {instance['Id']} not found in out data...")
                continue
            prompt_dict["model_answer"] = custom_id2out_instance[instance["Id"]]["response"]["body"]["choices"][0]["message"]["content"]
            for k in prompt_dict:
                instruction_prompt = instruction_prompt.replace('{'+k+'}', prompt_dict[k])
            batch_instance = {
                "custom_id": instance["Id"],
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": instance["llm_model"],
                    "temperature": 0.1,
                    "messages": [
                        {"role": "system", "content": "You are ChatGPT, a large language model trained by OpenAI, based on the GPT-4 architecture."},
                        {"role": "user", "content": instruction_prompt}
                    ]
                }
            }
            batch_instances.append(batch_instance)
        os.makedirs("../batch_inference", exist_ok=True)
        now = datetime.now()
        date_time_string = now.strftime('%Y-%m-%d_%H-%M-%S')
        save_jsonl_file(batch_instances, f"../batch_inference/{args.dataset.replace('.jsonl', '')}_verify_{date_time_string}.jsonl")
        print("Batch inference file saved...")
        exit()
    
    time.sleep(5)
    
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