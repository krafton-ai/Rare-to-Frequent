import os
import sys
sys.path.append('../')


from diffusers import DPMSolverMultistepScheduler

from ComposableDiffusion_sd3 import ComposableDiffusion3Pipeline

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
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    # Save path
    if args.r2f_generator == 'human':
        model_name = "Composable-" + args.model
        save_path = args.out_path + model_name + f'_{args.transition_step}_{args.num_inference_steps}_alt{args.alt_step}/'
    elif args.r2f_generator == 'gpt':
        #model_name = "R2F_gpt4_" + args.model
        model_name = "Composable-" + args.model
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
            r2f_prompts_dict = json.loads(f.read())
        #print(r2f_prompts_dict)

        r2f_prompts, visual_detail_levels = [], []
        for prompt in r2f_prompts_dict:
            r2f_prompts += r2f_prompts_dict[prompt]["r2f_prompt"]
            visual_detail_levels.append(r2f_prompts_dict[prompt]["visual_detail_level"])

    # Use the Euler scheduler here instead
    #if args.model == 'sdxl':
    #    pipe = DynamicDiffusionXLPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0",torch_dtype=torch.float16, use_safetensors=True, variant="fp16")
    if args.model == 'sd3':
        pipe = ComposableDiffusion3Pipeline.from_pretrained("stabilityai/stable-diffusion-3-medium", revision="refs/pr/26")
    pipe = pipe.to("cuda")

    if args.model == 'sdxl':
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config, use_karras_sigmas=True)


    # Inference
    for i, r2f_prompt in enumerate(r2f_prompts):

        print(r2f_prompt)
        print(f"{save_path}{str(i)}_{r2f_prompt[-1].rstrip()}.png")

        visual_detail_level = visual_detail_levels[i]
        print("visual_detail_level: ", visual_detail_level)

        level_to_transition = [0, 5, 10, 20, 30, 40]
        transition_steps = [level_to_transition[int(level)] for level in visual_detail_level] + [args.num_inference_steps]
        print("transition_steps: ", transition_steps)
        
        # run inference
        image = pipe(
            r2f_prompts = r2f_prompt,
            batch_size = 1, #batch size
            num_inference_steps=args.num_inference_steps, # sampling step
            transition_steps=transition_steps, # transition step
            alt_step=args.alt_step, # alternating step
            height = 1024, 
            width = 1024, 
            seed = 42,# random seed
        ).images[0]
        image.save(f"{save_path}{str(i)}_{r2f_prompt[-1].rstrip()}.png")

if __name__ == "__main__":
    main()