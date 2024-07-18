#from RegionalDiffusion_base import RegionalDiffusionPipeline
#from RegionalDiffusion_xl import RegionalDiffusionXLPipeline
from DynamicDiffusion_xl import DynamicDiffusionXLPipeline
from DynamicDiffusion_sd3 import DynamicDiffusion3Pipeline
from DynamicDiffusion_sd3_attn import DynamicDiffusion3Pipeline

from diffusers.schedulers import KarrasDiffusionSchedulers,DPMSolverMultistepScheduler, DDPMScheduler, EulerDiscreteScheduler
from preprocess.mllm import local_llm, GPT4, GPT4_Rare2Frequent
import torch


api_key = "sk-proj-IAZ4GP2D8ZiWo9yichrqT3BlbkFJTTlc56ffedaIrc5Y3ytu" # KRAFTON research

# If you want to use load ckpt, initialize with ".from_single_file". 
#pipe = RegionalDiffusionXLPipeline.from_single_file("path to your ckpt", torch_dtype=torch.float16, use_safetensors=True, variant="fp16")
# If you want to use diffusers, initialize with ".from_pretrained".

model = "sd3" #sdxl

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

# Demo
#prompt= 'a horned frog in a hood'
#r2f_prompts = ['a horned animal in a hood', 'a horned frog in a hood']

#prompt= "a yawning crab with a mouth"
#r2f_prompts = ['a yawning creature with a mouth', 'a yawning crab with a mouth']

prompt= 'A hairy frog'
r2f_prompts = ['A hairy animal', 'A hairy frog']

#TODO: complete the template!!
#r2f_prompts = GPT4_Rare2Frequent(prompt, key=api_key)

negative_prompt = ""
#cross_attention_kwargs = {"edit_type": "replace",
#                          "n_self_replace": 0.4,
#                          "n_cross_replace": {"default_": 1.0, "frog": 0.4},
#                          }

num_inference_steps, transition_step, alt_step = 50, 0, 50
images = pipe(
    r2f_prompts = r2f_prompts,
    batch_size = 1, #batch size
    num_inference_steps=num_inference_steps, # sampling step
    transition_step = transition_step,
    alt_step = alt_step,
    height = 1024, 
    negative_prompt=negative_prompt, # negative prompt
    #cross_attention_kwargs=cross_attention_kwargs, # FIXME: remove this
    width = 1024, 
    seed = 42,# random seed
    guidance_scale = guidance_scale,
    save = False
).images[0]
images.save(prompt + f"_R2F_{model}.png")