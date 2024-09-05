import os
import sys
sys.path.append('../')

from diffusers import DPMSolverMultistepScheduler
from RegionalDiffusion_xl import RegionalDiffusionXLPipeline
from diffusers import DiffusionPipeline

import torch
import argparse
import json
import math

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="sdxl",
        type=str,
    )
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
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=50,
        help="inference steps for denoising",
    )
    parser.add_argument(
        "--alt-step",
        type=int,
        default=2,
        help="transition step, from frequent (or alternating) to rare",
    )
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    # Save path
    model_name = "RPG-" + args.model
    save_path = args.out_path + model_name + '/' #+ f'_adaptive_{args.num_inference_steps}_alt{args.alt_step}/'

    if not os.path.exists(save_path):
        os.mkdir(save_path)
    
    ## User input
    test_file = args.test_file
    test_case = test_file.split('/')[-1].split('.')[0].replace('_gpt4', '')
    save_path = save_path + test_case + '/'
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    with open(test_file, 'r') as f:
        rpg_prompts = json.loads(f.read())

    # Use the Euler scheduler here instead
    if args.model == 'sdxl':
        pipe_rpg = RegionalDiffusionXLPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0",torch_dtype=torch.float16, use_safetensors=True, variant="fp16")
        pipe_rpg.to("cuda:1")
        pipe_rpg.scheduler = DPMSolverMultistepScheduler.from_config(pipe_rpg.scheduler.config,use_karras_sigmas=True)
        pipe_rpg.enable_xformers_memory_efficient_attention()

        # since rpg does not support non-split prompt
        pipe_sdxl = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0",torch_dtype=torch.float16, use_safetensors=True, variant="fp16")
        pipe_sdxl.to("cuda:1")
        pipe_sdxl.scheduler = DPMSolverMultistepScheduler.from_config(pipe_sdxl.scheduler.config,use_karras_sigmas=True)
        pipe_sdxl.enable_xformers_memory_efficient_attention()


    # Inference
    for i, key in enumerate(rpg_prompts):
        
        if i<10:
            continue

        rpg_prompt = rpg_prompts[key]
        print(rpg_prompt)

        regional_prompt = rpg_prompt['Regional Prompt']
        split_ratio = rpg_prompt['Final Split Ratio']
        
        
        # run inference
        if len(split_ratio.split(';')) == 1:
            print("***Run SDXL!!!")
            image = pipe_sdxl(
                prompt=regional_prompt,
                num_inference_steps=50,
            ).images[0]
        else:
            print("***Run RPG!!!")
            image = pipe_rpg(
                prompt = regional_prompt,
                split_ratio = split_ratio, # The ratio of the regional prompt, the number of prompts is the same as the number of regions, and the number of prompts is the same as the number of regions
                batch_size = 1, #batch size
                base_ratio = 0.5, # The ratio of the base prompt    
                base_prompt= key,       
                num_inference_steps=50, # sampling step
                height = 1024, 
                negative_prompt="", # negative prompt
                width = 1024, 
                seed = 2468,# random seed
                guidance_scale = 7.0
            ).images[0]
        image.save(f"{save_path}{str(i)}_{key.rstrip()}.png")

if __name__ == "__main__":
    main()