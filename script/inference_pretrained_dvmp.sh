export out_path="../images/"

#export test_file="../test/dvmp/dvmp_test500.txt"
export test_file="../test/dvmp/dvmp_single100.txt"

export pretrained_model_path="runwayml/stable-diffusion-v1-5"
python ../inference.py --pretrained_model_path "${pretrained_model_path}" --test_file "${test_file}" --out_path "${out_path}" 

export pretrained_model_path="stabilityai/stable-diffusion-2-base"
python ../inference.py --pretrained_model_path "${pretrained_model_path}" --test_file "${test_file}" --out_path "${out_path}" 

export pretrained_model_path="stabilityai/stable-diffusion-xl-base-1.0"
python ../inference.py --pretrained_model_path "${pretrained_model_path}" --test_file "${test_file}" --out_path "${out_path}" 

export pretrained_model_path="PixArt-alpha/PixArt-XL-2-1024-MS"
python ../inference.py --pretrained_model_path "${pretrained_model_path}" --test_file "${test_file}" --out_path "${out_path}" 

export pretrained_model_path="stabilityai/stable-diffusion-3-medium" # 
python ../inference.py --pretrained_model_path "${pretrained_model_path}" --test_file "${test_file}" --out_path "${out_path}" 



export test_file="../test/dvmp/dvmp_multi100.txt"

export pretrained_model_path="runwayml/stable-diffusion-v1-5"
python ../inference.py --pretrained_model_path "${pretrained_model_path}" --test_file "${test_file}" --out_path "${out_path}" 

export pretrained_model_path="stabilityai/stable-diffusion-2-base"
python ../inference.py --pretrained_model_path "${pretrained_model_path}" --test_file "${test_file}" --out_path "${out_path}" 

export pretrained_model_path="stabilityai/stable-diffusion-xl-base-1.0"
python ../inference.py --pretrained_model_path "${pretrained_model_path}" --test_file "${test_file}" --out_path "${out_path}" 

export pretrained_model_path="PixArt-alpha/PixArt-XL-2-1024-MS"
python ../inference.py --pretrained_model_path "${pretrained_model_path}" --test_file "${test_file}" --out_path "${out_path}" 

export pretrained_model_path="stabilityai/stable-diffusion-3-medium" # 
python ../inference.py --pretrained_model_path "${pretrained_model_path}" --test_file "${test_file}" --out_path "${out_path}" 