from utils import extract_pattern_content, clean_result, clean_trace_error_message
import json

from local_exec import execute_inner, execute_inner_with_input
from model_api import ChatGPTOpenAI, ClaudeAnthropic, ChatVLLM, select_llm_model
import traceback

def is_data_structure(s):
    try:
        # Evaluate the string as a Python expression
        result = eval(s)
        # Check if the result is one of the common data structures
        if isinstance(result, (list, tuple, dict, set)):
            return True, type(result), len(result)
        else:
            return False, None, None
    except:
        # If eval() fails, the string is not a valid Python expression
        return False, None, None

def extract_expert_list(input_str, instance):
    #print(input_str)
    if "Expert card: " in input_str:
        input_str = input_str.replace("Expert card: ", "Expert card (in JSON format): ")
    elif "**Expert Card**:" in input_str:
        input_str = input_str.replace("**Expert Card**:", "Expert card (in JSON format):")
    #print(input_str)
    try:
        expert_role_list = extract_pattern_content(input_str, "**:", "Expert card (in JSON format):")
        expert_card_list = extract_pattern_content(input_str, "Expert card (in JSON format):", "\n\n")
        #print(expert_role_list)
        #print(expert_card_list)
        if len(expert_role_list) != len(expert_card_list):
            return [], "ERROR: The number of expert roles does not match the number of expert cards. Please check the input."
        expert_list = []
        for i in range(len(expert_role_list)):
            expert_role = expert_role_list[i]
            expert_role = clean_result(expert_role)
            expert_card = expert_card_list[i]
            expert_card = clean_result(expert_card)
            # load into json
            expert_card = json.loads(expert_card)
            if "Input_Type" not in expert_card or "Output_Type" not in expert_card:
                return [], "ERROR: The expert card does not contain the 'Input_Type' or 'Output_Type' field. Please check the expert card."
            expert_card["Role"] = expert_role
            expert_card["Experts_Design"] = input_str
            expert_card["Original_Problem"] = instance["prompt_dict"]["task_problem"]
            expert_list.append(expert_card)
        return expert_list, "SUCCESS"
    except json.JSONDecodeError:
        return [], "ERROR: JSONDecodeError. Could not parse the expert card. Please check the format of the expert card."

def pack_current_trajectory(current_trajectory):
    result_str = ""
    for item in current_trajectory:
        if "Meta-Expert@" in item["expert"]:
            continue
        expert_output = item["output"]
        is_data_structure_flag, data_structure_type, data_structure_length = is_data_structure(expert_output)
        if is_data_structure_flag and data_structure_length >= 100:
            result_str += f"\n{item['expert']} output:\nData in Python `{data_structure_type}`. The data is a large data structure with more than 100 elements.\n\n"
        else:
            result_str += f"\n{item['expert']} output:\n{item['output']}\n\n"
    return result_str

def pack_current_trajectory_replace_last(current_trajectory, last_data_type):
    result_str = ""
    for i, item in enumerate(current_trajectory):
        if "Meta-Expert@" in item["expert"]:
            continue
        if i == len(current_trajectory) - 1:
            result_str += f"\n{item['expert']} output:\nData in Python `{last_data_type}`. The data is a large data structure with more than 100 elements.\n\n"
        else:
            result_str += f"\n{item['expert']} output:\n{item['output']}\n\n"
    return result_str

class LLMExpert:
    def __init__(self, expert_card, llm_model, problem_reflection, expert_prompt_file):
        self.name = expert_card["Name"]
        self.function_name = "_".join(self.name.lower().split())
        self.original_problem = expert_card["Original_Problem"]
        self.role = expert_card["Role"]
        self.experts_design = expert_card["Experts_Design"]
        self.input_type = expert_card["Input_Type"]
        self.output_type = expert_card["Output_Type"]

        self.llm_chain = select_llm_model(llm_model)

        if self.output_type != "str":
            data_type_instruction = f" Your job is to generate the answer in data type '{self.output_type}' based on the query."
        else:
            data_type_instruction = ""

        prompt_dict = {
            "name": self.name,
            "role": self.role,
            "original_problem": self.original_problem,
            "experts_design": self.experts_design,
            "data_type_instruction": data_type_instruction,
        }

        self.expert_prompt = open(expert_prompt_file, "r").read()

        if "{problem_reflection}" in self.expert_prompt:
            prompt_dict["problem_reflection"] = problem_reflection

        for k in prompt_dict:
            assert '{'+k+'}' in self.expert_prompt
        for k in prompt_dict:
            if not isinstance(prompt_dict[k], str):
                prompt_dict[k] = str(prompt_dict[k])
            self.expert_prompt = self.expert_prompt.replace('{'+k+'}', prompt_dict[k])


    def run(self, current_trajectory):
        # Process the input_data based on the expert's role
        # This is a placeholder logic. You would implement the specific logic here.
        input_data = pack_current_trajectory(current_trajectory)
        print(f"====================================== {self.name} START ====================")
        print("===== Input Start =====")
        print(input_data)
        print("===== Input End =====")

        messages_list = []
        
        # Generate and return the output
        run_time_prompt = self.expert_prompt.replace("$input_data$", input_data)
        entire_output = self.llm_chain.invoke(run_time_prompt)
        messages_list.append([{"role": "user", "content": run_time_prompt}, {"role": "assistant", "content": entire_output}])
        print("===== Output Start =====")
        print(entire_output)
        print("===== Output End =====")
        final_output = extract_pattern_content(entire_output, "### My Final Output Start ###", "### My Final Output End ###")
        final_output = clean_result(final_output[0])
        print(f"====================================== {self.name} END ====================")
        return messages_list, entire_output, final_output


class ToolExpert:
    def __init__(self, expert_card, llm_model, problem_reflection, expert_prompt_file):
        self.name = expert_card["Name"]
        self.function_name = "_".join(self.name.lower().split())
        self.original_problem = expert_card["Original_Problem"]
        self.role = expert_card["Role"]
        self.experts_design = expert_card["Experts_Design"]
        self.input_type = expert_card["Input_Type"]
        self.output_type = expert_card["Output_Type"]

        self.llm_chain = select_llm_model(llm_model)

        self.code_generation_prompt = open(expert_prompt_file, "r").read()

        prompt_dict = {
            "name": self.name,
            "role": self.role,
            "original_problem": self.original_problem,
            "experts_design": self.experts_design,
            "input_type": self.input_type,
            "output_type": self.output_type,
        }

        if "{problem_reflection}" in self.code_generation_prompt:
            prompt_dict["problem_reflection"] = problem_reflection

        for k in prompt_dict:
            assert '{'+k+'}' in self.code_generation_prompt
        for k in prompt_dict:
            if not isinstance(prompt_dict[k], str):
                prompt_dict[k] = str(prompt_dict[k])
            self.code_generation_prompt = self.code_generation_prompt.replace('{'+k+'}', str(prompt_dict[k]))
    
    def remove_if_main(self, script: str):
        new_content = script
        if 'if __name__ ==' in script:
            result_lines = script.split('\n')
            start_dedent = False
            for i, line in enumerate(result_lines):
                if 'if __name__ ==' in line:
                    start_dedent = True
                    result_lines[i] = ''
                if start_dedent:
                    result_lines[i] = result_lines[i][4:]
            new_content = '\n'.join(result_lines)
        return new_content


    def run(self, current_trajectory):
        # Process the input_data based on the expert's role
        # This is a placeholder logic. You would implement the specific logic here.
        last_expert_output = current_trajectory[-1]["output"]
        last_expert = current_trajectory[-1]["expert"]
        is_data_structure_flag, data_structure_type, data_structure_length = is_data_structure(last_expert_output)
        if is_data_structure_flag and data_structure_length >= 100:
            input_data = pack_current_trajectory_replace_last(current_trajectory, data_structure_type)
            how_to_read_input = f"The code should use the 'input()' function to capture the data generated by the {last_expert}. If necessary, use 'eval()' to convert the input data into a Python `{data_structure_type}` object."
            execute_with_input_flag = True
        else:
            input_data = pack_current_trajectory(current_trajectory)
            how_to_read_input = 'The code should include input as a variable properly in the "__main__" part.'
            execute_with_input_flag = False
        
        print(f"====================================== {self.name} START ====================")
        print("===== Input Start =====")
        print(input_data)
        print("===== Input End =====")

        messages_list = []
        
        # Generate and return the output
        run_time_prompt = self.code_generation_prompt.replace("$how_to_read_input$", how_to_read_input)
        run_time_prompt = run_time_prompt.replace("$input_data$", input_data)

        output_code = self.llm_chain.invoke(run_time_prompt)
        messages_list.append([{"role": "user", "content": run_time_prompt}, {"role": "assistant", "content": output_code}])
        org_output_code = output_code
        python_code_list = extract_pattern_content(output_code, "```python", "```")
        python_code = "\n".join(python_code_list)
        python_code = self.remove_if_main(python_code)
        org_python_code = python_code
        print("===== Code Start =====")
        print(python_code)
        print("===== Code End =====")

        #python_code += "\nprint(abc + xyz)\n" # test error fixing !!!!!!!!!!!!!!!!!!!!!

        retry = 0
        while retry < 3:
            try:
                if execute_with_input_flag:
                    std_output = execute_inner_with_input(python_code, last_expert_output+'\n', timeout=5)
                else:
                    std_output = execute_inner(python_code, timeout=5)
                break
            except Exception as e:
                previous_code = python_code
                retry += 1
                error_message = traceback.format_exc()
                clean_error_message = clean_trace_error_message(error_message)
                fix_error_prompt = f"{python_code}\n\nWhen I run the above code, I encountered an error:\n{clean_error_message}\n\nPlease fix the error and output the entire Python code again. Output:\n```python"
                output_code = self.llm_chain.invoke(fix_error_prompt)
                messages_list.append([{"role": "user", "content": fix_error_prompt}, {"role": "assistant", "content": output_code}])
                python_code_list = extract_pattern_content(output_code, "```python", "```")
                python_code = "\n".join(python_code_list)
                python_code = self.remove_if_main(python_code)
        if retry == 3:
            try:
                std_output = execute_inner(python_code, timeout=5)
                output_code = python_code
            except Exception as e:
                simulate_run_prompt = f"{python_code}\n\nPlease act as a Python interpreter. Simulate running the above Python code (ignore its syntax errors) and generate the output based on the code logic. Please think step by step and put the final output between ### Final Output Start ### and ### Final Output End ###."
                simulate_output = self.llm_chain.invoke(simulate_run_prompt)
                messages_list.append([{"role": "user", "content": simulate_run_prompt}, {"role": "assistant", "content": simulate_output}])
                std_output = extract_pattern_content(simulate_output, "### Final Output Start ###", "### Final Output End ###")[0]
                output_code = python_code
        
        print("===== Output Start =====")
        std_output = clean_result(std_output)
        print(std_output)
        print("===== Output End =====")

        entire_output = output_code
        entire_output += "\n\n### Python Code Start ###\n" + python_code + "\n### Python Code End ###"
        if retry == 3:
            entire_output += "\n\n### Python Execution Result Start (simulate) ###\n" + std_output + "\n### Python Execution Result End (simulate) ###"
        else:
            entire_output += "\n\n### Python Execution Result Start ###\n" + std_output + "\n### Python Execution Result End ###"

        print(f"====================================== {self.name} END ====================")
        return messages_list, entire_output, std_output