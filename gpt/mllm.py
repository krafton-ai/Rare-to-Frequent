import requests
import json
import os
import torch
import re
import numpy as np
import ast

# R2F with GPT4
def GPT4_Rare2Frequent(prompt, key):

    result = GPT4_Rare2Frequent_single(prompt, key)
    print(result)
    
    return result

def GPT4_Rare2Frequent_single(prompt, key):
    print("*** call GPT4_Rare2Frequent_single() ***")

    url = "https://api.openai.com/v1/chat/completions"
    api_key = key

    with open('template/template_r2f_system.txt', 'r') as f:
        template_system=f.readlines()
        prompt_system=' '.join(template_system)

    # FIXME:
    #with open('template/template_r2f_user_past.txt', 'r') as f:
    with open('template/template_r2f_user.txt', 'r') as f:
        template_user=f.readlines()
        template_user=' '.join(template_user)

    prompt_user=f"### Input: {prompt}\n### Output: "
    prompt_user= f"{template_user}\n\n{prompt_user}"
    
    payload = json.dumps({
    # FIXME:
    "model": "gpt-4o", # we suggest to use the latest version of GPT, you can also use gpt-4-vision-preivew, see https://platform.openai.com/docs/models/ for details. 
    "messages": [
        {
            "role": "system",
            "content": prompt_system
        },
        {
            "role": "user",
            "content": prompt_user
        }
    ]
    })
    headers = {
        'Accept': 'application/json',
        'Authorization': f'Bearer {api_key}',
        'User-Agent': 'Apifox/1.0.0 (https://apifox.com)',
        'Content-Type': 'application/json'
    }
    print('waiting for GPT-4 response')
    response = requests.request("POST", url, headers=headers, data=payload)
    obj=response.json()
    
    text=obj['choices'][0]['message']['content']
    print(text)

    #return get_params_dict_r2f(text)
    return get_params_dict_r2f_v2(text, prompt) # FIXME:


# R2F with LLaMA3
def LLaMA3_Rare2Frequent(prompt, pipeline):

    result = LLaMA3_Rare2Frequent_single(prompt, pipeline)
    print(result)
    
    return result

def LLaMA3_Rare2Frequent_single(prompt, pipeline):
    print("*** call LLaMA3_Rare2Frequent_single() ***")

    terminators = [
        pipeline.tokenizer.eos_token_id,
        pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    # Get prompt
    with open('template/template_r2f_system.txt', 'r') as f:
        template_system=f.readlines()
        prompt_system=' '.join(template_system)
    
    with open('template/template_r2f_user.txt', 'r') as f:
        template_user=f.readlines()
        template_user=' '.join(template_user)

    prompt_user=f"### Input: {prompt}\n### Output: "
    prompt_user= f"{template_user}\n\n{prompt_user}"
    
    message = [
                {"role": "system", "content": prompt_system},
                {"role": "user", "content": prompt_user},
    ]

    # inference
    outputs = pipeline(
                    message,
                    max_new_tokens=1024,
                    eos_token_id=terminators,
                    do_sample=True,
                    temperature=0.6,
                    top_p=0.9,
                )
    text = outputs[0]["generated_text"][-1]['content']
    print(text)

    result = get_params_dict_r2f_v2(text, prompt)
    print(result)

    #return get_params_dict_r2f(text)
    return get_params_dict_r2f_v2(text, prompt) # FIXME:

def get_params_dict_r2f(response):
    
    visual_detail_level = response.split("### Visual Detail Level: ")[1].split("### Final Prompt Sequence:")[0].replace(' ', '').replace('\n', '')

    gpt_prompts = response.split("### Final Prompt Sequence: ")
    
    if len(gpt_prompts)>1:
        #print("Final Prompt Sequence:", gpt_prompts[-1])
        
        sep_prompt_sequence = gpt_prompts[-1].split(" SEP ")
        
        final_r2f_prompts = []
        for split_prompt in sep_prompt_sequence:
            split_prompt_sequence = split_prompt.split(" BREAK ")

            final_r2f_prompts.append(split_prompt_sequence)
    else:
        print("No rare concept.")
        final_r2f_prompts = [gpt_prompts]
    
    output = {'visual_detail_level': visual_detail_level, 'r2f_prompt': final_r2f_prompts}
    return output
    
def get_params_dict_r2f_v2(response, prompt):

    visual_detail_level = response.split("### Visual Detail Level: ")[1].split("### Final Prompt Sequence:")[0].replace(' ', '').replace('\n', '').split('AND')
    gpt_prompts = response.split("### Final Prompt Sequence: ")[1]

    sep_prompt_sequence = gpt_prompts.split(" AND ")
    
    rare_prompt = prompt
    decomposed_r2f_prompts = []
    for split_prompt in sep_prompt_sequence:
        split_prompt_sequence = split_prompt.split(" BREAK ")
        print("split_prompt_sequence: ", split_prompt_sequence)

        if len(split_prompt_sequence) > 1:
            rare_prompt = rare_prompt.lower().replace(split_prompt_sequence[1].lower(), split_prompt_sequence[0].lower())
            decomposed_r2f_prompts.append(split_prompt_sequence)
    print("decomposed_r2f_prompts: ", decomposed_r2f_prompts)

    # sorted by Visual Detail Level
    idxs = np.argsort(visual_detail_level)
    visual_detail_level = [visual_detail_level[idx] for idx in idxs]

    final_r2f_prompts = []
    for idx in idxs:
        if visual_detail_level[idx] == '0':
            continue

        if len(decomposed_r2f_prompts[idx]) > 1:
            final_r2f_prompts.append(decomposed_r2f_prompts[idx][0])
            rare_prompt = rare_prompt.lower().replace(decomposed_r2f_prompts[idx][0].lower(), decomposed_r2f_prompts[idx][1].lower())

    final_r2f_prompts.append(prompt)

    output = {'visual_detail_level': visual_detail_level, 'r2f_prompt': [final_r2f_prompts]}
    return output

def LLaMA3_Rare2Frequent_plus(prompt, pipeline):
    print("*** call LLaMA3_Rare2Frequent_plus() ***")

    terminators = [
        pipeline.tokenizer.eos_token_id,
        pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    with open('template/template_r2fplus_system.txt', 'r') as f:
        template = f.readlines()
    
    text_prompt= f"{' '.join(template)}".replace("{INPUT}", prompt)
    
    message = [
        {
            "role": "system",
            "content": text_prompt,
        },
    ]

    # inference
    outputs = pipeline(
        message,
        max_new_tokens=1024,
        eos_token_id=terminators,
        do_sample=True,
        temperature=0.6,
        top_p=0.9,
    )
    text = outputs[0]["generated_text"][-1]['content']
    return text

def GPT4_Rare2Frequent_plus(prompt, key):
    print("*** call GPT4_Rare2Frequent_plus() ***")

    url = "https://api.openai.com/v1/chat/completions"
    api_key = key

    with open('template/template_r2fplus_system.txt', 'r') as f:
        template = f.readlines()
    
    text_prompt= f"{' '.join(template)}".replace("{INPUT}", prompt)
    
    payload = json.dumps({
        "model": "gpt-4",
        "messages": [
            {
                "role": "system",
                "content": text_prompt
            }
        ]
    })

    headers = {
        'Accept': 'application/json',
        'Authorization': f'Bearer {api_key}',
        'User-Agent': 'Apifox/1.0.0 (https://apifox.com)',
        'Content-Type': 'application/json'
    }

    print('waiting for GPT-4 response')
    response = requests.request("POST", url, headers=headers, data=payload)
    obj = response.json()
    text = obj['choices'][0]['message']['content']
    return text

def GPT4_DecomposeObject(prompt,key):
    print("*** call GPT4_DecomposeObject() ***")

    url = "https://api.openai.com/v1/chat/completions"
    api_key = key

    #with open('template/template_r2f_system.txt', 'r') as f:
    #    template_system=f.readlines()
    #    prompt_system=' '.join(template_system)

    with open('template/template_obj_user.txt', 'r') as f:
        template_user=f.readlines()
        template_user=' '.join(template_user)

    prompt_user = template_user.replace("{PROMPT}", prompt)
    
    payload = json.dumps({
    "model": "gpt-4", # we suggest to use the latest version of GPT, you can also use gpt-4-vision-preivew, see https://platform.openai.com/docs/models/ for details. 
    "messages": [
        {
            "role": "user",
            "content": prompt_user
        }
    ]
    })
    headers = {
        'Accept': 'application/json',
        'Authorization': f'Bearer {api_key}',
        'User-Agent': 'Apifox/1.0.0 (https://apifox.com)',
        'Content-Type': 'application/json'
    }
    print('waiting for GPT-4 response')
    response = requests.request("POST", url, headers=headers, data=payload)
    obj=response.json()
    
    text=obj['choices'][0]['message']['content']
    print(text)

    return get_params_dict_obj(text) # TODO: parsing

def get_params_dict_obj(response):

    # TODO: dict -> parsing
    background = response.split("Objects: ")[0].split("Background: ")[1].rstrip()
    obj_list = response.split("Objects: ")[1].rstrip()

    obj_list = ast.literal_eval(obj_list)
    return background, obj_list

# RPG
def GPT4_rpg(prompt,key):
    url = "https://api.openai.com/v1/chat/completions"
    api_key = key
    with open('template/template_rpg.txt', 'r') as f:
        template=f.readlines()
    user_textprompt=f"Caption:{prompt} \n Let's think step by step:"
    
    textprompt= f"{' '.join(template)} \n {user_textprompt}"
    
    payload = json.dumps({
    "model": "gpt-4", # we suggest to use the latest version of GPT, you can also use gpt-4-vision-preivew, see https://platform.openai.com/docs/models/ for details. 
    "messages": [
        {
            "role": "user",
            "content": textprompt
        }
    ]
    })
    headers = {
    'Accept': 'application/json',
    'Authorization': f'Bearer {api_key}',
    'User-Agent': 'Apifox/1.0.0 (https://apifox.com)',
    'Content-Type': 'application/json'
    }
    print('waiting for GPT-4 response')
    response = requests.request("POST", url, headers=headers, data=payload)
    obj=response.json()
    text=obj['choices'][0]['message']['content']
    print(text)
    # Extract the split ratio and regional prompt

    return get_params_dict(text)

def get_params_dict(output_text):
    response = output_text
    # Final split ratio
    split_ratio_match = re.search(r"Final Split Ratio: ([\d.,;]+)", response)
    if split_ratio_match:
        final_split_ratio = split_ratio_match.group(1)
        print("Final Split Ratio:", final_split_ratio)
    else:
        print("Final Split Ratio not found.")
    # Regional Prompt
    prompt_match = re.search(r"Regional Prompt: (.*?)(?=\n\n|\Z)", response, re.DOTALL)
    if prompt_match:
        regional_prompt = prompt_match.group(1).strip()
        print("Regional Prompt:", regional_prompt)
    else:
        print("Regional Prompt not found.")

    image_region_dict = {'Final Split Ratio': final_split_ratio, 'Regional Prompt': regional_prompt}    
    return image_region_dict


# Paraphrase
def GPT4_Paraphrase(prompt,key):
    url = "https://api.openai.com/v1/chat/completions"
    api_key = key
    with open('template/template_paraphrase.txt', 'r') as f:
        template=f.read()
    print(template)

    textprompt= template.replace('###PROMPT', prompt)
    print(textprompt)
    
    payload = json.dumps({
    "model": "gpt-4o", # we suggest to use the latest version of GPT, you can also use gpt-4-vision-preivew, see https://platform.openai.com/docs/models/ for details. 
    "messages": [
        {
            "role": "user",
            "content": textprompt
        }
    ]
    })
    headers = {
    'Accept': 'application/json',
    'Authorization': f'Bearer {api_key}',
    'User-Agent': 'Apifox/1.0.0 (https://apifox.com)',
    'Content-Type': 'application/json'
    }
    print('waiting for GPT-4 response')
    response = requests.request("POST", url, headers=headers, data=payload)
    obj=response.json()
    text=obj['choices'][0]['message']['content']
    print(text)
    # Extract the split ratio and regional prompt

    return text