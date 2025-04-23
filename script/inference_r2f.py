import os
import sys
sys.path.append('../')

from diffusers import DPMSolverMultistepScheduler, DDIMScheduler

from R2F_Diffusion_xl import R2FDiffusionXLPipeline
from R2F_Diffusion_sd3 import R2FDiffusion3Pipeline

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
    parser.add_argument(
        "--height",
        type=int,
        default=1024,
        help="height of image",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1024,
        help="width of image",
    )
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    # Save path
    if args.r2f_generator == 'human':
        model_name = "R2F-" + args.model
        save_path = args.out_path + model_name + f'_{args.transition_step}_{args.num_inference_steps}_alt{args.alt_step}/'
    elif args.r2f_generator == 'gpt':
        #model_name = "R2F_gpt4_" + args.model
        model_name = "R2F-" + args.model

        save_path = args.out_path + model_name + '/' #+ f'_adaptive_{args.num_inference_steps}_alt{args.alt_step}/'

    if not os.path.exists(save_path):
        os.mkdir(save_path)
    
    ## User input
    test_file = args.test_file
    test_case = test_file.split('/')[-1].split('.')[0].replace('_gpt4', '')
    save_path = save_path + test_case + '/'
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    if args.r2f_generator == 'human':
        with open(test_file) as f:
            r2f_prompts = [line.replace('\n','').split(', ') for line in f]
            visual_detail_levels = [None for i in r2f_prompts]

    elif args.r2f_generator == 'gpt':
        with open(test_file, 'r') as f:
            r2f_prompts= json.loads(f.read())


    # Use the Euler scheduler here instead
    if args.model == 'sdxl':
        pipe = R2FDiffusionXLPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0",torch_dtype=torch.float16, use_safetensors=True, variant="fp16")
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config, use_karras_sigmas=True)
    elif args.model == 'sd3':
        pipe = R2FDiffusion3Pipeline.from_pretrained("stabilityai/stable-diffusion-3-medium", revision="refs/pr/26")
    pipe = pipe.to("cuda")

    # Inference
    for i, prompt in enumerate(r2f_prompts):

        r2f_prompt = r2f_prompts[prompt]
        
        # run inference
        image = pipe(
            r2f_prompts = r2f_prompt,
            batch_size = 1, #batch size
            num_inference_steps=args.num_inference_steps, # sampling step
            height = args.height, 
            width = args.width, 
            seed = 42,# random seed
        ).images[0]
        image.save(f"{save_path}{str(i)}_{prompt.rstrip()}.png")



if __name__ == "__main__":
    main()