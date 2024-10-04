try:
    import openai
except ImportError:
    print("OpenAI API not installed.")
    pass
try:
    import anthropic
except ImportError:
    print("Anthropic API not installed.")
    pass
import time
import random
import requests
import json
import sys
import os

def select_llm_model(llm_model):
    if "llama-3" in llm_model.lower() or "llama3" in llm_model.lower():
        llm_chain = ChatVLLM(model_name=llm_model)
    elif "claude" in llm_model:
        llm_chain = ClaudeAnthropic(model_name=llm_model)
    elif "gpt" in llm_model:
        llm_chain = ChatGPTOpenAI(model_name=llm_model, api_key_type="openai")
    
    return llm_chain

class ChatGPTOpenAI:
    def __init__(self, model_name: str, api_key_type: str):
        self.model_name = model_name
        if "gpt-3.5" in model_name or "gpt-35" in model_name:
            self.system_prompt = "You are ChatGPT, a large language model trained by OpenAI, based on the GPT-3.5 architecture."
        elif "gpt-4" in model_name:
            self.system_prompt = "You are ChatGPT, a large language model trained by OpenAI, based on the GPT-4 architecture."
        
        
        if "_azure" in model_name:
            openai.api_key = os.environ["AZURE_API_KEY"]
            openai.api_base = os.environ["AZURE_API_BASE"]
        else:
            openai.api_key = os.environ["OPENAI_API_KEY"]

        if "_azure" in model_name:
            self.api_model = model_name.replace("_azure", "")
        else:
            self.api_model = model_name

    def invoke(self, prompt: str, temperature: float = 0.2, frequency_penalty: float = 0.0) -> str:
        final_messages=[{"role": "system", "content": self.system_prompt}, {"role": "user", "content": prompt}]

        if "_azure" in self.model_name:
            RateLimitError_max = 300
            APIError_max = 300
            InvalidRequestError_max = 5
            sleep_time = 2
        else:
            RateLimitError_max = 50
            APIError_max = 5
            InvalidRequestError_max = 3
            sleep_time = 5
        
        RateLimitError_count = 0
        APIError_count = 0
        InvalidRequestError_count = 0

        while True:
            try:
                openai_response = openai.ChatCompletion.create(
                    model=self.api_model,
                    messages=final_messages,
                    max_tokens=4096,
                    temperature=temperature,
                    frequency_penalty=frequency_penalty,
                )
                openai_response.choices[0].message.content
            except Exception as e:
                print('Error occurred' + ', retrying. Error type: ', type(e).__name__)
                if RateLimitError_count >= RateLimitError_max:
                    print(f'RateLimitError_count exceeded {RateLimitError_max}, exiting...')
                    response = None
                    break
                elif APIError_count >= APIError_max:
                    print(f'APIError_count exceeded {APIError_max}, exiting...')
                    response = None
                    break
                elif InvalidRequestError_count >= InvalidRequestError_max:
                    print(f'InvalidRequestError_count exceeded {InvalidRequestError_max}, exiting...')
                    response = None
                    break
                elif type(e).__name__ == 'RateLimitError':
                    time.sleep(random.uniform(sleep_time-0.2, sleep_time+0.2))
                    RateLimitError_count += 1
                elif type(e).__name__ == 'APIError':
                    time.sleep(random.uniform(sleep_time-0.2, sleep_time+0.2))
                    APIError_count += 1
                elif type(e).__name__ in ['InvalidRequestError', 'BadRequestError', 'AttributeError', 'IndexError']:
                    time.sleep(random.uniform(sleep_time-0.2, sleep_time+0.2))
                    InvalidRequestError_count += 1
            else:
                response = openai_response
                break
        
        if response is None or len(response.choices) == 0:
            resp = ""
            finish_reason = "Error during OpenAI inference"
            return resp
        resp = response.choices[0].message.content
        finish_reason = response.choices[0].finish_reason
        return resp

class ClaudeAnthropic:
    def __init__(self, model_name: str):
        self.client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        # "claude-3-opus-20240229",
        # "claude-3-sonnet-20240229",
        # "claude-3-haiku-20240307",
        # "claude-2.1",
        # "claude-2.0",
        # "claude-instant-1.2",
        self.system_prompt = "The assistant is Claude, created by Anthropic."
        self.api_model = model_name

    def invoke(self, prompt: str, temperature: float = 0.2, frequency_penalty: float = 0.0) -> str:
        
        final_messages=[
            {"role": "user", "content": prompt}
        ]

        RateLimitError_max = 20
        APIError_max = 5
        sleep_time = 5
        
        RateLimitError_count = 0
        APIError_count = 0
        while True:
            try:
                claude_response = self.client.messages.create(
                    model= self.api_model,
                    system=self.system_prompt,
                    max_tokens=4096,
                    temperature=temperature,
                    messages=final_messages
                )
            except Exception as e:
                """
                # https://github.com/anthropics/anthropic-sdk-python
                * 400	BadRequestError
                401	AuthenticationError
                403	PermissionDeniedError
                404	NotFoundError
                * 422	UnprocessableEntityError
                * 429	RateLimitError
                * >=500	InternalServerError
                * N/A	APIConnectionError
                """
                print('Error occurred' + ', retrying. Error type: ', type(e).__name__)
                if RateLimitError_count >= RateLimitError_max:
                    print(f'RateLimitError_count exceeded {RateLimitError_max}, exiting...')
                    response = None
                    break
                elif APIError_count >= APIError_max:
                    print(f'APIError_count exceeded {APIError_max}, exiting...')
                    response = None
                    break
                elif type(e).__name__ in ['RateLimitError', 'InternalServerError']:
                    time.sleep(sleep_time)
                    RateLimitError_count += 1
                elif type(e).__name__ == 'APIConnectionError':
                    time.sleep(sleep_time)
                    APIError_count += 1
                elif type(e).__name__ in ['BadRequestError', 'UnprocessableEntityError']:
                    response = None
                    break
                else:
                    response = None
                    break
            else:
                response = claude_response
                break
        
        if response is None or len(response.content) == 0:
            resp = ""
            finish_reason = "Error during Claude inference"
            return resp
        resp = response.content[0].text
        finish_reason = response.stop_reason
        return resp

class ChatVLLM:
    def __init__(self, model_name: str):
        if ":" in model_name:
            self.port = model_name.split(":")[-1]
            self.model_name = model_name.split(":")[0]
        else:
            self.port = 4231
            self.model_name = model_name
        
        if "llama-3" in self.model_name.lower():
            template = [
                "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n",
                "You are a helpful assistant.<|eot_id|>\n",  # The system prompt is optional
                "<|start_header_id|>user<|end_header_id|>\n\n",
                "{prompt}<|eot_id|>\n",
                "<|start_header_id|>assistant<|end_header_id|>\n\n",
            ]
            self.template = "".join(template)
            
        elif "sft_llama3_70b_chat_dp8_tp8_jiping" in self.model_name.lower():
            template = [
                "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are a helpful AI assistant.<|eot_id|>",
                "<|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
            ]
            self.template = "".join(template)

    def invoke(self, prompt: str, temperature: float = 0.2, frequency_penalty: float = 0.0) -> str:
        new_prompt = self.template.format(prompt=prompt)
        json_data={
            "model": self.model_name,
            "prompt": new_prompt,
            "max_tokens": 2048,
            "top_p": 0.9,
            "stop": ["<|end_of_text|>", "<|eot_id|>"],
            "do_sample": True,
            "repetition_penalty": 1.2,
            "temperature": temperature,
        }

        random_number = 0
        """Support load balance of 4 models hosted on 4 gpus. Suppose 4 models are hosted on 4 consecutive ports. Random select a number in [0, 1, 2, 3]"""
        # random_number = random.randint(0, 3)

        try:
            new_port = int(self.port) + random_number
            #response = requests.post(f'http://localhost:{self.port}/v1/completions', json=json_data)
            response = requests.post(f'http://localhost:{new_port}/v1/completions', json=json_data)
            response_content_string = response.text
            completion = json.loads(response_content_string)['choices'][0]['text']
        except KeyboardInterrupt:
            print("Interrupted by user. Exiting...")
            sys.exit(0)
        except Exception as e:
            print(f"Error: {e}")
            completion = ""
        return completion