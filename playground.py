from R2F_Diffusion_xl import R2FDiffusionXLPipeline
from R2F_Diffusion_sd3 import R2FDiffusion3Pipeline

from diffusers import DPMSolverMultistepScheduler

from gpt.mllm import GPT4_Rare2Frequent, LLaMA3_Rare2Frequent
import torch

api_key = "YOUR_API_KEY"

model = "sd3"
if model == "sdxl":
    pipe = R2FDiffusionXLPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0",torch_dtype=torch.float16, use_safetensors=True, variant="fp16")
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config, use_karras_sigmas=True)
elif model == 'sd3':
    pipe = R2FDiffusion3Pipeline.from_pretrained("stabilityai/stable-diffusion-3-medium", revision="refs/pr/26")
pipe.to("cuda")

# Demo
prompt= 'A hairy frog'

# Get r2f prompt from LLMs
r2f_prompt = GPT4_Rare2Frequent(prompt, key=api_key)
#r2f_prompt = LLaMA3_Rare2Frequent(prompt, model_id="meta-llama/Llama-3.1-8B-Instruct")
print(r2f_prompt)

image = pipe(
    r2f_prompts = r2f_prompt,
    seed = 42,# random seed
).images[0]
image.save(f"{prompt}_test.png")