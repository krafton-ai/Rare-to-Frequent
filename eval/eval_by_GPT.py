import requests
import json
import os
import numpy as np
import argparse
import base64

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        required=True,
        help="Model name",
    )
    parser.add_argument(
        "--eval_data",
        type=str,
        default="dvmp_single100",
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


# Function to encode the image
def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')


def eval_GPT4(img_path, prompt, key):
    url = "https://api.openai.com/v1/chat/completions"
    api_key = key

    # GPT PROMPT
    eval_prompt = f"You are my assistant to evaluate the correspondence of the image to a given text prompt. \
            focus on the objects in the image and their attributes (such as color, shape, texture), spatial layout and action relationships. \
            According to the image and your previous answer, evaluate how well the image aligns with the text prompt: \"{prompt}\"  \
            Give a score from 0 to 5, according the criteria: \n\
            5: the image perfectly matches the content of the text prompt, with no discrepancies. \
            4: the image portrayed most of the actions, events and relationships but with minor discrepancies. \
            3: the image depicted some elements in the text prompt, but ignored some key parts or details. \
            2: the image did not depict any actions or events that match the text. \
            1: the image failed to convey the full scope in the text prompt. \
            Provide your score and explanation (within 20 words) in the following format: \
            ###SCORE: score \
            ###EXPLANATION: explanation"

    # Getting the base64 string
    base64_image = encode_image(img_path)

    payload = {
      "model": "gpt-4-turbo",
      "messages": [
        {
          "role": "user",
          "content": [
            {
              "type": "text",
              "text": eval_prompt
            },
            {
              "type": "image_url",
              "image_url": {
                "url": f"data:image/jpeg;base64,{base64_image}"
              }
            }
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
    
    text=output['choices'][0]['message']['content']
    print(text)
    
    score = text.split('###EXPLANATION: ')[0].split("###SCORE: ")[1].lstrip().rstrip()
    explanation = text.split('###EXPLANATION: ')[1].lstrip().rstrip()

    return {"score": int(score), "explanation": explanation}


def main():
    args = parse_args()

    # openai
    api_key = "sk-proj-IAZ4GP2D8ZiWo9yichrqT3BlbkFJTTlc56ffedaIrc5Y3ytu" # KRAFTON research
    #api_key = "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" # KRAFTON NLP"

    # Use the Euler scheduler here instead
    if 'sd3' in args.model:
        model_name = 'stable-diffusion-3-medium'
    elif 'sdxl' in args.model:
        model_name = 'stable-diffusion-xl-base-1.0'
    elif 'sd2' in args.model:
        model_name = 'stable-diffusion-2-base'
    elif 'sd1.5' in args.model:
        model_name = 'stable-diffusion-v1-5'
    elif 'PixArt' in args.model:
        model_name = 'PixArt-XL-2-1024-MS'
    elif 'SynGen' in args.model:
        model_name = 'SynGen-sd1.5'
    elif 'LMD' in args.model:
        model_name = 'LMD-sdxl'
    elif 'R2F' in args.model:
        model_name = 'R2F-sd3'
    
    img_folder = f'../images/{model_name}/{args.eval_data}/'
    print("image_folder: ", img_folder)

    files = os.listdir(img_folder)
    
    idxs = [int(f.split('_')[0]) for f in files]
    idxs = np.argsort(idxs)
    
    # Output path
    save_path = f"{args.out_path}{model_name}/"
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    save_file = save_path + args.eval_data + '_t2i_scores_by_gpt.json'
    if os.path.exists(save_file):
      with open(save_file, 'r') as f:
        result = json.load(f)
    else:
      result = {}
    print(result)

    # Evaluation
    for i, idx in enumerate(idxs):
      print(i)
      file = files[idx]
      file_path = img_folder + file
      prompt = file.split('_')[1].split('.png')[0]

      if file not in result:
        output = eval_GPT4(file_path, prompt, api_key)
        
        result[file] = output

        with open(save_file, 'w+') as f:
          json.dump(result, f, indent=4)
        

    

if __name__ == "__main__":
    main()