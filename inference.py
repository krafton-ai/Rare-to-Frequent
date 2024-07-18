import os
from diffusers import DiffusionPipeline, StableDiffusionPipeline, PixArtAlphaPipeline#, StableDiffusion3Pipeline
from StableDiffusion3 import StableDiffusion3Pipeline

import sys
sys.path.append('../../Linguistic-Binding-in-Diffusion-Models/')
from syngen_diffusion_pipeline import SynGenDiffusionPipeline

import torch
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pretrained_model_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model",
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
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    model_id = args.pretrained_model_path
    model_name = model_id.split('/')[-1]
    save_path = args.out_path + model_name + '/'

    print(f"model_id: {model_id}, model_name: {model_name}")

    if not os.path.exists(save_path):
        os.mkdir(save_path)

    # Use the Euler scheduler here instead
    if 'stable-diffusion-3-medium' in model_name:
        pipe = StableDiffusion3Pipeline.from_pretrained(model_id, revision="refs/pr/26")
    elif 'stable-diffusion-xl' in model_name:
        pipe = DiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16, use_safetensors=True, variant="fp16")
    elif 'stable-diffusion' in model_name:
        pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16, use_safetensors=True, variant="fp16")
    elif 'PixArt' in model_name:
        pipe = PixArtAlphaPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    elif 'SynGen' in model_name:
        pipe = SynGenDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16)
    pipe = pipe.to("cuda")

    ## User input
    test_file = args.test_file
    test_case = test_file.split('/')[-1].split('.')[0]
    with open(test_file) as f:
        prompts = [line.rstrip() for line in f]

    # Inference
    for i, prompt in enumerate(prompts):
        print(save_path + f"{test_case}_{str(i)}_{prompt}.png")

        # run inference
        generator = torch.Generator(device="cuda").manual_seed(42)
        if 'SynGen' in model_name:
            pipe(prompt=prompt, generator=generator, syngen_step_size=20,
                  attn_res=(int(math.sqrt(256)), int(math.sqrt(256))))
        else:
            image = pipe(prompt, num_inference_steps=50, generator=generator).images[0]

        image.save(save_path + f"{test_case}_{str(i)}_{prompt}.png")

if __name__ == "__main__":
    main()