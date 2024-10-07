import os
import sys
sys.path.append('../')

from R2Fplus_Diffusion_sd3 import R2FplusDiffusionPrompt, R2FplusDiffusion3Pipeline
from transformers import (
    AutoProcessor,
    AutoModelForZeroShotObjectDetection,
    AutoModelForMaskGeneration
)


import torch
import argparse
import json
import math

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="sd3",
        type=str,
    )
    parser.add_argument(
        "--test_file",
        type=str,
        nargs="?",
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
        "--visual_detail_level_to_transition_step",
        type=int,
        nargs="+",
        default=[0, 5, 10, 20, 30, 40],
        help="visual detail level to transition step",
    )
    parser.add_argument(
        "--alt_step",
        type=int,
        default=2,
        help="transition step, from frequent (or alternating) to rare",
    )
    parser.add_argument(
        "--save_all",
        action='store_true',
    )
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    model_name = "R2Fplus-" + args.model
    save_path = args.out_path + model_name + '/'

    if not os.path.exists(save_path):
        os.mkdir(save_path)
    
    ## User input
    test_file = args.test_file
    test_case = test_file.split('/')[-1].split('.')[0].replace('_gpt4o', '_test')
    save_path = save_path + test_case + '/'
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    
    with open(test_file, 'r') as f:
        r2fplus_prompts_json = json.loads(f.read())
        r2fplus_prompts = [R2FplusDiffusionPrompt.from_json(obj) for obj in r2fplus_prompts_json.values()]

    if args.model == 'sd3':
        detector_id = "IDEA-Research/grounding-dino-tiny"
        detector_model = AutoModelForZeroShotObjectDetection.from_pretrained(detector_id)
        detector_processor = AutoProcessor.from_pretrained(detector_id)
        
        segmentor_id = "facebook/sam-vit-base"
        segmentor_model = AutoModelForMaskGeneration.from_pretrained(segmentor_id)
        segmentor_processor = AutoProcessor.from_pretrained(segmentor_id)

        pipe = R2FplusDiffusion3Pipeline.from_pretrained(
            "stabilityai/stable-diffusion-3-medium",
            detector_model=detector_model,
            detector_processor=detector_processor,
            segmentor_model=segmentor_model,
            segmentor_processor=segmentor_processor,
            revision="refs/pr/26",
            torch_dtype=torch.float16
        )
    else:
        raise Exception(f"Not implemented model: {args.model}")

    pipe = pipe.to("cuda")

    generator = torch.Generator(device="cuda")
    generator.manual_seed(42)

    # Inference
    for i, r2fplus_prompt in enumerate(r2fplus_prompts):
        print(r2fplus_prompt.original_prompt)
        
        # run inference
        output = pipe(
            r2fplus_prompt=r2fplus_prompt,
            num_inference_steps=args.num_inference_steps, # sampling step
            visual_detail_level_to_transition_step=args.visual_detail_level_to_transition_step, # visual detail level to transition step
            alt_step=args.alt_step, # alternating step
            height=1024, 
            width=1024, 
            generator=generator,
        )

        if args.save_all:
            for object_key, obj in r2fplus_prompt.objects.items():
                obj_prompt = obj.prompt.replace(object_key, obj.object)

                save_filename = f"{save_path}{i}_{object_key}_{obj_prompt}.png"
                output.object_images[0][object_key].save(save_filename)
                
                save_filename = f"{save_path}{i}_{object_key}_{obj_prompt}_masked.png"
                output.masked_object_images[0][object_key].save(save_filename)
                
                save_filename = f"{save_path}{i}_{object_key}_{obj_prompt}_bbox.png"
                output.bbox_object_images[0][object_key].save(save_filename)

            save_filename = f"{save_path}{str(i)}_{r2fplus_prompt.original_prompt}.png"
            output.images[0].save(save_filename)

            save_filename = f"{save_path}{str(i)}_{r2fplus_prompt.original_prompt}_bbox.png"
            output.bbox_images[0].save(save_filename)

        else:
            save_filename = f"{save_path}{i}_{r2fplus_prompt.original_prompt}.png"
            output.images[0].save(save_filename)

if __name__ == "__main__":
    main()