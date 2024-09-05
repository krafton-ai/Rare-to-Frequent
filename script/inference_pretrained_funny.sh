export out_path="../images/"

# Funny
export test_file="../test/original_prompt/funny/funny_visualization.txt"

export pretrained_model_path="stabilityai/stable-diffusion-xl-base-1.0"
CUDA_VISIBLE_DEVICES=1 python inference.py --pretrained_model_path "${pretrained_model_path}" --test_file "${test_file}" --out_path "${out_path}" 

export pretrained_model_path="PixArt-alpha/PixArt-XL-2-1024-MS"
#CUDA_VISIBLE_DEVICES=1 python inference.py --pretrained_model_path "${pretrained_model_path}" --test_file "${test_file}" --out_path "${out_path}" 

export pretrained_model_path="stabilityai/stable-diffusion-3-medium" # 
CUDA_VISIBLE_DEVICES=1 python inference.py --pretrained_model_path "${pretrained_model_path}" --test_file "${test_file}" --out_path "${out_path}"

export pretrained_model_path="black-forest-labs/FLUX.1-schnell" # 
#CUDA_VISIBLE_DEVICES=1 python inference.py --pretrained_model_path "${pretrained_model_path}" --test_file "${test_file}" --out_path "${out_path}"