from R2F_Diffusion_xl import R2FDiffusionXLPipeline
from R2F_Diffusion_sd3 import R2FDiffusion3Pipeline
from R2F_Diffusion_flux import R2FFluxPipeline

from diffusers import DPMSolverMultistepScheduler

from gpt.mllm import GPT4_Rare2Frequent, LLaMA3_Rare2Frequent
import torch

api_key = "YOUR_API_KEY"

model = "itercomp"
if model == 'sd3':
    pipe = R2FDiffusion3Pipeline.from_pretrained("stabilityai/stable-diffusion-3-medium", revision="refs/pr/26")
elif model == "sdxl":
    pipe = R2FDiffusionXLPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0",torch_dtype=torch.float16, use_safetensors=True, variant="fp16")
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config, use_karras_sigmas=True)
elif model == "flux":
    pipe = R2FFluxPipeline.from_pretrained("black-forest-labs/FLUX.1-schnell", torch_dtype=torch.bfloat16) # In R2F, we do experiment on FLUX.1-schnell which it requires 4 sampling steps.
elif model == "itercomp":
    pipe = R2FDiffusionXLPipeline.from_pretrained("comin/IterComp",torch_dtype=torch.float16, use_safetensors=True)
pipe.to("cuda")

# Demo
prompt= 'A hairy frog'

# Get r2f prompt from LLMs
llm = "gpt4o"
if llm == "gpt4o":
    r2f_prompt = GPT4_Rare2Frequent(prompt, key=api_key)
elif llm == "llama3.1":
    r2f_prompt = LLaMA3_Rare2Frequent(prompt, model_id="meta-llama/Llama-3.1-8B-Instruct")
print(r2f_prompt)

if model != "flux":
    image = pipe(
        r2f_prompts = r2f_prompt,
        seed = 42,# random seed
    ).images[0]
    image.save(f"{prompt}_test.png")
else:
    # flux
    image = pipe(
        r2f_prompts = r2f_prompt["r2f_prompt"],
        visual_level_details = r2f_prompt["visual_detail_level"],
        alphas = [0.3, 0.15, 0.15, 0.15], # Hyperparam for mixing
        seed = 42,
    ).images[0]
    image.save(f"{prompt}_test.png")