import os
import sys
sys.path.append('../')

from mllm import GPT4_Rare2Frequent, LLaMA3_Rare2Frequent
import transformers
import torch
import argparse
import json


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="GPT4",
        type=str,
    )
    parser.add_argument(
        "--test_file",
        type=str,
        default="../test/original_prompt/rarebench/rarebench_single_4action.txt",
        help="Test file used for generation",
    )
    parser.add_argument(
        "--out_path",
        type=str,
        nargs="?",
        default="../test/r2f_prompt/rarebench/rarebench_single_4action_gpt4.txt",
        help="output file path",
    )
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    api_key = "sk-proj-IAZ4GP2D8ZiWo9yichrqT3BlbkFJTTlc56ffedaIrc5Y3ytu" # KRAFTON research

    ## User input
    test_file = args.test_file
    test_case = test_file.split('/')[-1].split('.')[0]
    with open(test_file) as f:
        prompts = [line.rstrip() for line in f]

    if os.path.exists(args.out_path):
      with open(args.out_path, 'r') as f:
        result = json.load(f)
    else:
      result = {}
    print(result)

    for i, prompt in enumerate(prompts):
        print(i, prompt)
        
        if prompt not in result:
            # Get GPT responses
            if args.model == "GPT4":
                r2f_prompts = GPT4_Rare2Frequent(prompt, key=api_key)
                result[prompt] = r2f_prompts

            elif args.model == "LLaMA3":
                model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
                access_token = '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'

                ## Get Model
                pipeline = transformers.pipeline(
                    "text-generation",
                    model=model_id,
                    model_kwargs={"torch_dtype": torch.float16},
                    device_map="auto",
                    token=access_token,
                )
                r2f_prompts = LLaMA3_Rare2Frequent(prompt, pipeline)
                result[prompt] = r2f_prompts

            with open(args.out_path, 'w') as f:
                json.dump(result, f, indent=4)


if __name__ == "__main__":
    main()