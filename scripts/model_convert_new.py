import os
import json
import argparse

# Convert LitGPT trained model into Huggingface so that we can host model inference using vllm.
# IMPORTANT: Please change the data path accordingly.

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="llama-3-8b_4k_hdflow_v10", type=str, help="model name")
    parser.add_argument("--target_steps", default="final", type=str, help="target checkpoint steps")
    args = parser.parse_args()

    workspace = "/xxxx/HDFlow/litgpt_trained_model"

    os.makedirs(f"{workspace}/{args.model_name}_hf/{args.target_steps}", exist_ok=True)
    command = f"litgpt convert_from_litgpt {workspace}/{args.model_name}/{args.target_steps} {workspace}/{args.model_name}_hf/{args.target_steps}"
    print(command)
    os.system(command)
    
    command = f"cp /xxxx/checkpoints/meta-llama/Meta-Llama-3-8B-Instruct/config.json {workspace}/{args.model_name}_hf/{args.target_steps}/config.json"

    print(command)
    os.system(command)

    command = f"cp {workspace}/{args.model_name}/{args.target_steps}/*.json {workspace}/{args.model_name}_hf/{args.target_steps}/"
    print(command)
    os.system(command)

    command = f"cp {workspace}/{args.model_name}/{args.target_steps}/*.yaml {workspace}/{args.model_name}_hf/{args.target_steps}/"
    print(command)
    os.system(command)

    command = f"mv {workspace}/{args.model_name}_hf/{args.target_steps}/model.pth {workspace}/{args.model_name}_hf/{args.target_steps}/model.bin"
    print(command)
    os.system(command)

    print("Done.")