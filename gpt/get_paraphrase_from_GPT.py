import os
import sys
sys.path.append('../')

from mllm import local_llm, GPT4_Rare2Frequent, GPT4_Paraphrase
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


    with open(args.out_path, 'w+') as f:
        for i, prompt in enumerate(prompts):
            print(i, prompt)
            
            #if prompt not in result:
            
            # Get GPT responses
            if args.model == "GPT4":
                prompt_paraphrased = GPT4_Paraphrase(prompt, key=api_key)
                f.write(prompt_paraphrased + '\n')


if __name__ == "__main__":
    main()