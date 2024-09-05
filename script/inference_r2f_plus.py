import os
import sys
sys.path.append('../')

from diffusers import DPMSolverMultistepScheduler
from DynamicDiffusion_xl import DynamicDiffusionXLPipeline

#from DynamicDiffusion_sd3 import DynamicDiffusion3Pipeline
from DynamicDiffusion_sd3_resize import DynamicDiffusion3Pipeline

import torch
import argparse
import json
import math

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--r2f-generator",
        default="gpt",
        type=str,
    )
    parser.add_argument(
        "--model",
        default="sd3",
        type=str,
    )
    parser.add_argument(
        "--test_file",
        type=str,
        nargs="?",
        default="test/rarebench/multi_object_1and.txt",
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
        "--transition_step",
        type=int,
        default=0,
        help="transition step, from frequent (or alternating) to rare",
    )
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    ### Get Model
    if args.model == 'sdxl':
        pipe = DynamicDiffusionXLPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0",torch_dtype=torch.float16, use_safetensors=True, variant="fp16")
    elif args.model == 'sd3':
        pipe = DynamicDiffusion3Pipeline.from_pretrained("stabilityai/stable-diffusion-3-medium", revision="refs/pr/26")
    pipe = pipe.to("cuda")

    if args.model == 'sdxl':
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config, use_karras_sigmas=True)

    ### User input
    test_file = args.test_file
    test_case = test_file.split('/')[-1].split('.')[0]
    with open(test_file, 'r') as f:
        r2f_prompts = json.load(f) # TODO:
    #print(r2f_prompts)

    ### Save path
    if args.r2f_generator == 'gpt':
        save_path = args.out_path + f'R2F_{args.model}/'

    if not os.path.exists(save_path):
        os.mkdir(save_path)

    save_path = save_path + f'{test_case}/'
    if not os.path.exists(save_path):
        os.mkdir(save_path)


    ### Inference
    for i, prompt in enumerate(r2f_prompts):

        r2f_prompt = r2f_prompts[prompt]
        print(save_path + f"{i}_{prompt}.png")

        ### Inference Multi-object
        level_to_transition = [0, 5, 10, 20, 30, 40]
        for key in r2f_prompt:
            if 'obj' in key:
                visual_detail_level = r2f_prompt[key]["transition"] 
                r2f_prompt[key]["transition"] = level_to_transition[int(visual_detail_level)]
        
        # run inference
        image = pipe(
            r2f_prompts = r2f_prompt,
            batch_size = 1, #batch size
            num_inference_steps=args.num_inference_steps, # sampling step
            stop_background = 2,
            stop_local_attn = 30,
            stop_fusion = 40,
            weight_base = 0.5,
            height = 1024, 
            width = 1024, 
            seed = 42,# random seed
        ).images[0]

        # TODO:
        image.save(save_path + f"{i}_{prompt}.png")

if __name__ == "__main__":
    main()