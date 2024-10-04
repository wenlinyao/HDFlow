This is the code and data repo of paper [HDFlow: Enhancing LLM Complex Problem-Solving with Hybrid Thinking and Dynamic Workflows](https://arxiv.org/abs/2409.17433). If you think this repo is useful, please cite us. Thank you!

```bibtex
@article{yao2024hdflow,
  title={HDFlow: Enhancing LLM Complex Problem-Solving with Hybrid Thinking and Dynamic Workflows},
  author={Yao, Wenlin and Mi, Haitao and Yu, Dong},
  journal={arXiv preprint arXiv:2409.17433},
  year={2024}
}
```


## Set up

```sh
pip install openai==0.28
pip install anthropic
pip install tqdm
pip install pandas
pip install datasets
```

## Set API keys

```sh
export OPENAI_API_KEY=xxxx
```

If you would like to use OpenAI API service on Azure, please set the AZURE_API_KEY and the AZURE_API_BASE:

```sh
export AZURE_API_KEY=xxxx
export AZURE_API_BASE=https://gptproxy.xxx.xxx/v1
```

If you would like to use Anthropic Claude models, please set ANTHROPIC_API_KEY.

```sh
export ANTHROPIC_API_KEY=xxxx
```


## Run fast thinking, slow thinking and hybrid thinking using GPT-4-Turbo

We use gpt-4o-mini as an example model here to reduce costs. hdflow_hybrid.py will depends on the results produced by fast_thinking and slow_thinking.

```sh
python hdflow.py --api_model gpt-4o-mini-2024-07-18 --solution_name fast_thinking --max_num 10 --dataset MATH_sampled --overwrite False
python hdflow.py --api_model gpt-4o-mini-2024-07-18 --solution_name slow_thinking --max_num 10 --dataset MATH_sampled --iter 3 --overwrite False
python hdflow_hybrid.py --api_model gpt-4o-mini-2024-07-18 --solution_name hybrid_thinking_after --max_num 10 --dataset MATH_sampled --overwrite False
```

Here are a few examples to get the results reported in the paper.

```sh
python hdflow.py --api_model gpt-4-0125-preview --solution_name fast_thinking --max_num 50 --dataset BBH --overwrite False
python hdflow.py --api_model gpt-4-0125-preview --solution_name slow_thinking --max_num 50 --dataset BBH --overwrite False --iter 3
python hdflow_hybrid.py --api_model gpt-4-0125-preview --solution_name hybrid_thinking_after --max_num 50 --dataset BBH --overwrite False

python hdflow.py --api_model gpt-4-0125-preview --solution_name fast_thinking --max_num all --dataset MATH_sampled --overwrite False
python hdflow.py --api_model gpt-4-0125-preview --solution_name slow_thinking --max_num all --dataset MATH_sampled --overwrite False --iter 3
python hdflow_hybrid.py --api_model gpt-4-0125-preview --solution_name hybrid_thinking_after --max_num all --dataset MATH_sampled --overwrite False


python hdflow.py --api_model gpt-4-0125-preview --solution_name fast_thinking --max_num 100 --dataset GameOf24 --overwrite False
python hdflow.py --api_model gpt-4-0125-preview --solution_name slow_thinking --max_num 100 --dataset GameOf24 --overwrite False --iter 3
python hdflow_hybrid.py --api_model gpt-4-0125-preview --solution_name hybrid_thinking_after --max_num 100 --dataset GameOf24 --overwrite False
```

## Run fast thinking, slow thinking and hybrid thinking using Llama-3 models.

We host our models using [vllm](https://github.com/vllm-project/vllm), so please follow the link to install vllm first. 

### Running with the original Llama-3-8B-Instruct model.

1. Download Huggingface Meta-Llama-3-8B-Instruct model into the dir /xxxx/checkpoints/meta-llama/Meta-Llama-3-8B-Instruct/. Next, use vllm to host the model service.

```sh
CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server --model /xxxx/checkpoints/meta-llama/Meta-Llama-3-8B-Instruct/ --tensor-parallel-size 1 --max-num-batched-tokens 8192 --dtype bfloat16 --port 4231
```

2. Run fast thinking, slow thinking and hybrid thinking by calling the model service.

```sh
python hdflow.py --max_num all --solution_name fast_thinking --api_model /xxxx/checkpoints/meta-llama/Meta-Llama-3-8B-Instruct/:4231 --dataset BBH --overwrite False

python hdflow.py --max_num all --solution_name fast_thinking --api_model /xxxx/checkpoints/meta-llama/Meta-Llama-3-8B-Instruct/:4231 --dataset MATH --overwrite False

python hdflow.py --max_num all --solution_name fast_thinking --api_model /xxxx/checkpoints/meta-llama/Meta-Llama-3-8B-Instruct/:4231 --dataset GameOf24 --overwrite False

python hdflow.py --max_num 100 --solution_name fast_thinking --api_model /xxxx/checkpoints/meta-llama/Meta-Llama-3-8B-Instruct/:4231 --dataset math_reasoning --overwrite False
```

## Run fast thinking, slow thinking and hybrid thinking using Llama-3 models after hybrid thinking.

1. Download our [model](https://huggingface.co/wenlinyao/HDFlow-Llama-3-8B-Instruct/tree/main) into the dir /xxxx/checkpoints/wenlinyao/HDFlow-Llama-3-8B-Instruct/. Next, use vllm to host the model service. Here provides an example to host two models at the same time (using different ports).

```sh
CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server --model /xxxx/checkpoints/wenlinyao/HDFlow-Llama-3-8B-Instruct/ --tensor-parallel-size 1 --max-num-batched-tokens 8192 --dtype bfloat16 --port 4231

CUDA_VISIBLE_DEVICES=1 python -m vllm.entrypoints.openai.api_server --model /xxxx/checkpoints/wenlinyao/HDFlow-Llama-3-8B-Instruct/ --tensor-parallel-size 1 --max-num-batched-tokens 8192 --dtype bfloat16 --port 4232
```

2. Run fast thinking, slow thinking and hybrid thinking by calling the model service.

```sh
python hdflow.py --max_num all --solution_name fast_thinking --api_model /xxxx/checkpoints/wenlinyao/HDFlow-Llama-3-8B-Instruct/:4231 --dataset GameOf24 --overwrite False

python hdflow.py --max_num all --solution_name slow_thinking --api_model /xxxx/checkpoints/wenlinyao/HDFlow-Llama-3-8B-Instruct/:4231 --dataset GameOf24 --overwrite False --iter 3

python hdflow_hybrid.py --max_num all --solution_name hybrid_thinking_after --api_model /xxxx/checkpoints/wenlinyao/HDFlow-Llama-3-8B-Instruct/:4231 --dataset GameOf24 --overwrite False
```

## Training data of hybrid thinking tuning.

We also released the training data of our hybrid thinking tuning. You can use [LitGPT](https://github.com/Lightning-AI/litgpt) to train your own models. Our data can be found [here](https://huggingface.co/datasets/wenlinyao/HDFlow-train/tree/main).

Full training data of LitGPT: **hdflow_training_data_v10.json**

All synthesized reasoning problems: **reasoning_problems.jsonl**

Fast thinking and slow thinking reasoning trajectories on synthesized problems: **fast_slow_trajectories.jsonl**

