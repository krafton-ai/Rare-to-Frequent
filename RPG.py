from RegionalDiffusion_base import RegionalDiffusionPipeline
from RegionalDiffusion_xl import RegionalDiffusionXLPipeline
from diffusers.schedulers import KarrasDiffusionSchedulers,DPMSolverMultistepScheduler
from preprocess.mllm import local_llm,GPT4
import torch

api_key = "sk-proj-IAZ4GP2D8ZiWo9yichrqT3BlbkFJTTlc56ffedaIrc5Y3ytu" # KRAFTON research


# If you want to use load ckpt, initialize with ".from_single_file". 
#pipe = RegionalDiffusionXLPipeline.from_single_file("path to your ckpt", torch_dtype=torch.float16, use_safetensors=True, variant="fp16")

# If you want to use diffusers, initialize with ".from_pretrained".
# SD1.5
#pipe = RegionalDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5",torch_dtype=torch.float16, use_safetensors=True, variant="fp16")
# SDXL
pipe = RegionalDiffusionXLPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0",torch_dtype=torch.float16, use_safetensors=True, variant="fp16")
pipe.to("cuda")

pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config,use_karras_sigmas=True)
pipe.enable_xformers_memory_efficient_attention()
## User input
#prompt= 'a horned frog in a hood'
prompt= 'a horned frog in a hood and a dancing dog'
#para_dict = GPT4(prompt, key=api_key)

#FIXME:
#para_dict = {'Final split ratio': '1', 'Regional Prompt': 'A quirky horned frog peeks out from the coziness of a hood, its textured skin and prominent horns contrasted against the soft fabric, creating an intriguing and whimsical portrait.'}
para_dict = {'Final split ratio': '1;1', 'Regional Prompt': 'Modeled in the top half of the image is a horned frog, its eyes vigilant and skin textured with natural patterns, adorned in a quirky hood that adds a touch of whimsy. BREAK The lower half of the image captures a dancing dog mid-twirl, its fur detailed with life-like movement, paws outstretched and a playful grin suggesting the sheer delight of its dance.'}

split_ratio = para_dict['Final split ratio']
regional_prompt = para_dict['Regional Prompt']
negative_prompt = ""

images = pipe(
    prompt = regional_prompt,
    split_ratio = split_ratio, # The ratio of the regional prompt, the number of prompts is the same as the number of regions, and the number of prompts is the same as the number of regions
    batch_size = 1, #batch size
    base_ratio = 0.5, # The ratio of the base prompt    
    base_prompt= prompt,       
    num_inference_steps=50, # sampling step
    height = 1024, 
    negative_prompt=negative_prompt, # negative prompt
    width = 1024, 
    seed = 2468,# random seed
    guidance_scale = 7.0
).images[0]
images.save(prompt + "_RPG.png")