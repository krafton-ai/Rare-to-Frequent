import os
import sys
sys.path.append('../')

from diffusers import DPMSolverMultistepScheduler, DDIMScheduler

from R2F_Diffusion_flux import R2FFluxPipeline

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
        default="flux",
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
        default=4,
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
    parser.add_argument(
        "--alphas",
        type=str,
        default=None,
        help="alpha list",
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
        model_name = "R2F-" + args.model
        save_path = args.out_path + model_name + '/' #+ f'alphas={args.alphas}/'
        
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
    pipe = R2FFluxPipeline.from_pretrained("black-forest-labs/FLUX.1-schnell", torch_dtype=torch.bfloat16)
    pipe = pipe.to("cuda")

    if 'flux' in args.model:
        for i, r2f_prompt in enumerate(r2f_prompts):
            print(r2f_prompt)
            print(f"{save_path}{str(i)}_{r2f_prompt[-1].rstrip()}.png")

            if not isinstance(visual_detail_levels[i], list):
                visual_detail_level = [int(visual_detail_levels[i])] 
            else:
                visual_detail_level = visual_detail_levels[i]
                visual_detail_level = [int(i) for i in visual_detail_level]

            # alpha list
            if args.alphas is not None:
                alpha_list = [float(x) for x in args.alphas.split(',')]
                assert len(alpha_list) == args.num_inference_steps
            else:
                alpha_list = None

            image = pipe(
                r2f_prompts=r2f_prompt,
                batch_size = 1,
                num_inference_steps = args.num_inference_steps,
                height = args.height, 
                width = args.width,
                alphas = alpha_list,
                visual_level_details = visual_detail_level,
                seed = 42,# random seed
            ).images[0]
            image.save(f"{save_path}{str(i)}_{r2f_prompt[-1].rstrip()}.png")
    else:
        raise NotImplementedError("hi")
    
if __name__ == "__main__":
    main()