import requests
import json
import os
import numpy as np
import argparse
import base64

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--eval_data",
        type=str,
        default="rarebench",
        required=True,
        help="data for evaluation",
    )
    parser.add_argument(
        "--out_path",
        type=str,
        nargs="?",
        default="result/",
        help="output file path",
    )
    args = parser.parse_args()
    return args


def eval_GPT4(prompt, key):
    url = "https://api.openai.com/v1/chat/completions"
    api_key = key

    # GPT PROMPT
    eval_prompt = f"You are an assistant to evaluate if the text prompt contains rare concepts that exist infrequently or not in the real world. \
            Evaluate if rare concepts are contained in the text prompt: \"{prompt}\" \
            The answer format should be YES or NO, without any reasoning."

    payload = {
      "model": "gpt-4o",
      "messages": [
        {
          "role": "user",
          "content": [
            {
              "type": "text",
              "text": eval_prompt
            },
          ]
        }
      ],
      "max_tokens": 4096
    }

    headers = {
      "Content-Type": "application/json",
      "Authorization": f"Bearer {api_key}"
    }

    print('waiting for GPT-4 response')
    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
    output=response.json()
    #print(output)
    
    text=output['choices'][0]['message']['content']
    print(text)
    
    return {"rareness": text}


def main():
    args = parse_args()

    # openai
    api_key = "APIKEY"
    

    # Load Dataset
    if 'rarebench' in args.eval_data:
        data_path = '../test/original_prompt/rarebench/'
    if 'dvmp' in args.eval_data:
        data_path = '../test/original_prompt/dvmp/'
    if 'compbench' in args.eval_data:
        data_path = '../test/original_prompt/compbench/'
    
    prompts = []
    files = os.listdir(data_path)
    for file in files:
        if '.txt' not in file:
                continue

        with open(data_path+file, 'r') as f:
            prompt = f.readlines()

            for p in prompt:
                prompts.append(p.rstrip())
    #print(prompts)
    print(len(prompts))

    # Save path
    save_file = args.out_path + 'rareness_' + args.eval_data + '.json'
    if os.path.exists(save_file):
      with open(save_file, 'r') as f:
        result = json.load(f)
    else:
      result = {}
    print(result)


    # Evaluation
    for i, prompt in enumerate(prompts):
        try:
          output = eval_GPT4(prompt, api_key)
        except:
          try:
            output = eval_GPT4(prompt, api_key)
          except:
            output = eval_GPT4(prompt, api_key)
        
        result[f'{i}_{prompt}'] = output

        with open(save_file, 'w+') as f:
          json.dump(result, f, indent=4)
    

if __name__ == "__main__":
    main()