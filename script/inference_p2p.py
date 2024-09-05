import os
import sys
sys.path.append('../')

import json
import torch
from prompt_to_prompt_pipeline import Prompt2PromptPipeline
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--test_file",
        type=str,
        nargs="?",
        default="A bathroom with green tile and a red shower curtain",
        help="Test file used for generation",
    )
    parser.add_argument(
        "--out_path",
        type=str,
        nargs="?",
        default="images/",
        help="output file path",
    )
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    model_name = "P2P"
    save_path = args.out_path + model_name + '/'

    if not os.path.exists(save_path):
        os.mkdir(save_path)
    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    if device.type == "cuda":
        pipe = Prompt2PromptPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0",
                                                    torch_dtype=torch.float16, use_safetensors=True).to(device)
    else:
        pipe = Prompt2PromptPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0",
                                                    torch_dtype=torch.float32, use_safetensors=False).to(device)

    # Demo
    '''
    prompts = ["a pink bear riding a bicycle on the beach", "a pink dragon riding a bicycle on the beach"]
    cross_attention_kwargs = {"edit_type": "replace",
                               "n_self_replace": 0.4,
                               "n_cross_replace": {"default_": 1.0, "dragon": 0.4},
                               }
    
    prompts = ['a horned animal in a hood', 'a horned frog in a hood']
    cross_attention_kwargs = {"edit_type": "replace",
                               "n_self_replace": 0.4,
                               "n_cross_replace": {"default_": 1.0, "frog": 0.4},
                               }
    '''

    ## User input
    test_file = args.test_file
    test_case = test_file.split('/')[-1].split('.')[0]
    with open(test_file) as f:
        r2f_prompts = json.load(f)
    #print(r2f_prompts)

    save_path = save_path + f'{test_case}/'
    if not os.path.exists(save_path):
        os.mkdir(save_path)


    # Inference
    for i, prompt in enumerate(r2f_prompts):
        r2f_prompt = r2f_prompts[prompt]['r2f_prompt'][0]
        print(r2f_prompt)
        #print(save_path + f"{test_case}_{str(i)}_{r2f_prompt[1].rstrip()}.png")

        rare_object = list(set(r2f_prompt[1].split(' ')) - set(r2f_prompt[0].split(' ')))
        rare_object = ' '.join(rare_object)
        print("rare_object: ", rare_object)

        cross_attention_kwargs = {"edit_type": "replace",
                               "n_self_replace": 0.4,
                               "n_cross_replace": {"default_": 1.0, rare_object: 0.4},
                               }

        # run inference
        generator = torch.Generator().manual_seed(42)
        image = pipe(r2f_prompt, cross_attention_kwargs=cross_attention_kwargs, generator=generator).images

        print(f"Num images: {len(image)}")

        for j, img in enumerate(image):
            if j == 0:
                img.save(save_path + f"frequent_{str(i)}_{r2f_prompt[1].rstrip()}.png")
            else:
                img.save(save_path + f"{str(i)}_{r2f_prompt[1].rstrip()}.png")



if __name__ == "__main__":
    main()