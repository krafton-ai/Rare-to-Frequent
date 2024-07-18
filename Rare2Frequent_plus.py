#from RegionalDiffusion_base import RegionalDiffusionPipeline
#from RegionalDiffusion_xl import RegionalDiffusionXLPipeline
from DynamicDiffusion_xl import DynamicDiffusionXLPipeline
from DynamicDiffusion_sd3 import DynamicDiffusion3Pipeline
from DynamicDiffusion_sd3_attn import DynamicDiffusion3Pipeline

from diffusers.schedulers import DPMSolverMultistepScheduler, FlowMatchEulerDiscreteScheduler
from gpt.mllm import local_llm, GPT4, GPT4_Rare2Frequent
import torch
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prompt",
        default='A horned tiger and a spotted monkey',
        type=str,
    )
    parser.add_argument(
        "--stop_background",
        default=1,
        type=int,
    )
    parser.add_argument(
        "--stop_local_attn",
        default=20,
        type=int,
    )
    parser.add_argument(
        "--stop_fusion",
        default=40,
        type=int,
    )
    parser.add_argument(
        "--weight_base",
        default=0.5,
        type=float,
    )
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    ### Get decomposed prompts from GPT
    api_key = "sk-proj-IAZ4GP2D8ZiWo9yichrqT3BlbkFJTTlc56ffedaIrc5Y3ytu" # KRAFTON research

    # Demo
    #prompt= 'a horned frog in a hood'
    #r2f_prompts = ['a horned animal in a hood', 'a horned frog in a hood']

    #prompt= "a yawning crab with a mouth"
    #r2f_prompts = ['a yawning creature with a mouth', 'a yawning crab with a mouth']

    # TODO: fusion test
    #prompt= 'A hairy frog is sitting on top of a wooly lizard, at the desert'
    #prompt= 'a dog sitting on top of a cat, at the desert'
    #prompt= 'four suitcases and two swans'

    if args.prompt == 'a hairy frog is sitting on top of a wooly lizard, at the desert':
        r2f_prompts = { "obj1": {"rare": "a hairy frog, at the desert", "freq": "a hairy animal, at the desert", "transition": 5, "bbox": [300, 350, 300, 200]}, #[390, 500, 112, 132]
                        "obj2": {"rare": "a wooly lizard, at the desert", "freq": "a wooly animal, at the desert", "transition": 4, "bbox": [50, 650, 700, 250]}, #[315, 650, 275, 155]
                        "background": {"freq": "desert"},
                        "base": {"freq": args.prompt}
                        }
    if args.prompt == 'a horned tiger and a spotted monkey, at the forest':
        r2f_prompts = { "obj1": {"rare": "a horned tiger, at the forest", "freq": "a horned animal, at the forest", "transition": 3, "bbox": [192, 300, 292, 412]}, #[192, 290, 292, 412]
                        "obj2": {"rare": "a spotted monkey, at the forest", "freq": "a spotted animal, at the forest", "transition": 3, "bbox": [670, 400, 255, 315]}, #[670, 180, 255, 315]
                        "background": {"freq": "forest"}, # TODO: TODO: TODO:
                        "base": {"rare": "a horned animal and a spotted animal", "freq": args.prompt}
                        }

    #TODO: complete the template!!
    #prompt = args.prompt
    #r2f_prompts = GPT4_Rare2Frequent(prompt, key=api_key)
    #print("r2f_prompts: ", r2f_prompts)

    ### Get model
    model = "sd3" # sdxl, sd3

    if model == "sdxl":
        pipe = DynamicDiffusionXLPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0",torch_dtype=torch.float16, use_safetensors=True, variant="fp16")
        guidance_scale = 7.0
    elif model == 'sd3':
        pipe = DynamicDiffusion3Pipeline.from_pretrained("stabilityai/stable-diffusion-3-medium", revision="refs/pr/26")
        guidance_scale = 7.0
    pipe.to("cuda")

    # what scheduler?
    if model == "sdxl":
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config, use_karras_sigmas=True)
    elif model == "sd3":
        pipe.scheduler = FlowMatchEulerDiscreteScheduler.from_config(pipe.scheduler.config)


    ### Inference Multi-object
    level_to_transition = [0, 5, 10, 20, 30, 40]
    for key in r2f_prompts:
        if 'obj' in key:
            visual_detail_level = r2f_prompts[key]["transition"] 
            r2f_prompts[key]["transition"] = level_to_transition[int(visual_detail_level)]

    # config
    # run inference
    images = pipe(
        r2f_prompts = r2f_prompts,
        negative_prompt="",
        batch_size = 1, 
        num_inference_steps=50,
        stop_background = args.stop_background,
        stop_local_attn = args.stop_local_attn,
        stop_fusion = args.stop_fusion,
        weight_base = args.weight_base,
        alt_step=2,
        height = 1024, 
        width = 1024, 
        seed = 42,
        guidance_scale = guidance_scale,
        save = False
    ).images[0]
    images.save(f"images_fusion_format2/{args.prompt}_R2F_bigbox_localw0_{model}_back{args.stop_background}_local{args.stop_local_attn}_fusion{args.stop_fusion}_w0{int(args.weight_base*10)}.png")


    '''
    ### Inference Single-object
    num_inference_steps, transition_step, alt_step = 50, 0, 50
    images = pipe(
        r2f_prompts = r2f_prompts,
        batch_size = 1, #batch size
        num_inference_steps=num_inference_steps, # sampling step
        transition_step = transition_step,
        alt_step = alt_step,
        height = 1024, 
        negative_prompt="", # negative prompt
        width = 1024, 
        seed = 42,# random seed
        guidance_scale = guidance_scale,
        save = False
    ).images[0]
    images.save(f"images_fusion_format/{prompt}_R2F_{model}.png")
    '''

if __name__ == "__main__":
    main()