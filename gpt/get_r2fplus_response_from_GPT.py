import os
import sys
sys.path.append('../')

from mllm import GPT4_Rare2Frequent_plus, LLaMA3_Rare2Frequent_plus
import transformers
import torch
import argparse
import json

from R2Fplus_Diffusion_sd3 import R2FplusDiffusionPrompt


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
        default="../test/original_prompt/rarebench/rarebench_multi_1and.txt",
        help="Test file used for generation",
    )
    parser.add_argument(
        "--out_path",
        type=str,
        nargs="?",
        default="../test/r2fplus_prompt/rarebench/rarebench_multi_1and_gpt4.txt",
        help="output file path",
    )
    parser.add_argument(
        "--max_retries",
        default=10,
        type=int,
    )
    args = parser.parse_args()
    return args


def get_r2fplus_prompt(prompt, api_key, args):
    prev_error = None
    for retry in range(args.max_retries):
        if args.model == "GPT4":
            r2fplus_prompt_raw = GPT4_Rare2Frequent_plus(prompt, key=api_key)

        elif args.model == "LLaMA3":
            model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
            access_token = 'HF-token'

            ## Get Model
            pipeline = transformers.pipeline(
                "text-generation",
                model=model_id,
                model_kwargs={"torch_dtype": torch.float16},
                device_map="auto",
                token=access_token,
            )
            r2fplus_prompt_raw = LLaMA3_Rare2Frequent_plus(prompt, pipeline)

        try:
            r2fplus_prompt_json = json.loads(r2fplus_prompt_raw)
            r2fplus_prompt = R2FplusDiffusionPrompt.from_json(r2fplus_prompt_json)

        except ValueError as e:
            prev_error = e
            print(f"Error: {e}")
            continue

        return r2fplus_prompt_json
    
    raise prev_error


def main():
    args = parse_args()
    api_key = "APIKEY"

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
            try:
                r2fplus_prompt_json = get_r2fplus_prompt(prompt, api_key, args)
                result[prompt] = r2fplus_prompt_json
                with open(args.out_path, 'w') as f:
                    json.dump(result, f, indent=4)

            except ValueError as err:
                print(f"LLM failed to generate R2Fplus prompt for {prompt}. Error: {str(err)}")


if __name__ == "__main__":
    main()