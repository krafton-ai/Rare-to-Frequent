import os
import sys
sys.path.append('../')

from mllm import GPT4_rpg
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
        default="../test/original_prompt/dvmp/dvmp_single100.txt",
        help="Test file used for generation",
    )
    parser.add_argument(
        "--out_path",
        type=str,
        nargs="?",
        default="../test/rpg_prompt/dvmp/dvmp_single100_gpt4.txt",
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
        print(i)

        if prompt not in result:
            # Get GPT responses
            if args.model == "GPT4":
                # FIXME:
                try:
                    rpg_prompts = GPT4_rpg(prompt, key=api_key)
                    result[prompt] = rpg_prompts
                except:
                    print("**error occered, retry")
                    try:
                        rpg_prompts = GPT4_rpg(prompt, key=api_key)
                        result[prompt] = rpg_prompts
                    except:
                        print("**error occered twice, retry")
                        rpg_prompts = GPT4_rpg(prompt, key=api_key)
                        result[prompt] = rpg_prompts

            with open(args.out_path, 'w+') as f:
                json.dump(result, f, indent=4)
        

if __name__ == "__main__":
    main()