import os
import sys
sys.path.append('../')

from R2F_Multi_Diffusion_sd3 import R2FMultiDiffusionPrompt, R2FMultiDiffusion3Pipeline
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

    model_name = "R2F-multi-" + args.model
    save_path = args.out_path + model_name + '/'

    if not os.path.exists(save_path):
        os.mkdir(save_path)
    
    ## User input
    test_file = args.test_file
    test_case = test_file.split('/')[-1].split('.')[0].replace('_gpt4', '')
    save_path = save_path + test_case + '/'
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    
    with open(test_file, 'r') as f:
        r2f_multi_prompts_json = json.loads(f.read())
        r2f_multi_prompts = [R2FMultiDiffusionPrompt.from_json(obj) for obj in r2f_multi_prompts_json.values()]

    if args.model == 'sd3':
        detector_id = "IDEA-Research/grounding-dino-tiny"
        detector_model = AutoModelForZeroShotObjectDetection.from_pretrained(detector_id)
        detector_processor = AutoProcessor.from_pretrained(detector_id)
        
        segmentor_id = "facebook/sam-vit-base"
        segmentor_model = AutoModelForMaskGeneration.from_pretrained(segmentor_id)
        segmentor_processor = AutoProcessor.from_pretrained(segmentor_id)

        pipe = R2FMultiDiffusion3Pipeline.from_pretrained(
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
    for i, r2f_multi_prompt in enumerate(r2f_multi_prompts):
        print(r2f_multi_prompt.base_prompt)
        
        # run inference
        output = pipe(
            r2f_multi_prompt=r2f_multi_prompt,
            num_inference_steps=args.num_inference_steps, # sampling step
            visual_detail_level_to_transition_step=args.visual_detail_level_to_transition_step, # visual detail level to transition step
            alt_step=args.alt_step, # alternating step
            height=1024, 
            width=1024, 
            generator=generator,
        )

        if args.save_all:
            save_filename = f"{save_path}{str(i)}_{r2f_multi_prompt.base_prompt}.png"
            output.images[0].save(save_filename)

            save_filename = f"{save_path}{str(i)}_{r2f_multi_prompt.base_prompt}_bbox.png"
            output.bbox_images[0].save(save_filename)

            for j, (obj, object_image, masked_object_image, bbox_object_image) in enumerate(zip(
                r2f_multi_prompt.objects,
                output.object_images[0],
                output.masked_object_images[0],
                output.bbox_object_images[0]
            )):
                save_filename = f"{save_path}{i}_{j}_{obj.prompt}.png"
                object_image.save(save_filename)
                
                save_filename = f"{save_path}{i}_{j}_{obj.prompt}_masked.png"
                masked_object_image.save(save_filename)
                
                save_filename = f"{save_path}{i}_{j}_{obj.prompt}_bbox.png"
                bbox_object_image.save(save_filename)

        else:
            save_filename = f"{save_path}{i}_{r2f_multi_prompt.base_prompt}.png"
            output.images[0].save(save_filename)

if __name__ == "__main__":
    main()