export out_path="../images/"

### Single-object
#export test_file="../test/r2f_prompt/dvmp/dvmp_single100_gpt4.txt"
#python inference_r2f_single.py --test_file "${test_file}" --out_path "${out_path}"

### Multi-object
export test_file="../test/r2f_prompt/dvmp/dvmp_multi100_gpt4.txt"
CUDA_VISIBLE_DEVICES=1 python inference_r2f_multi.py --test_file "${test_file}" --out_path "${out_path}"