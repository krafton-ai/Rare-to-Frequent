export out_path="../images/"

### Single-object
export test_file="../test/rpg_prompt/dvmp/dvmp_single100_gpt4.txt"
CUDA_VISIBLE_DEIVCES=1 python inference_rpg.py --test_file "${test_file}" --out_path "${out_path}" 

### Multi-object
#export test_file="../test/rpg_prompt/dvmp/dvmp_multi100_gpt4.txt"
#CUDA_VISIBLE_DEIVCES=1 python inference_rpg.py --test_file "${test_file}" --out_path "${out_path}" 