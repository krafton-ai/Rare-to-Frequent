export out_path="../images/"
export model='sdxl'
export resolution=512

### Single-object
export test_file="../test/r2f_prompt/rarebench/rarebench_single_1property_gpt4.txt"
#python inference_r2f.py --test_file "${test_file}" --out_path "${out_path}" --model "${model}" --height "${resolution}" --width "${resolution}"

export test_file="../test/r2f_prompt/rarebench/rarebench_single_2shape_gpt4.txt"
#python inference_r2f.py --test_file "${test_file}" --out_path "${out_path}" --model "${model}" --height "${resolution}" --width "${resolution}"

export test_file="../test/r2f_prompt/rarebench/rarebench_single_3texture_gpt4.txt"
#python inference_r2f.py --test_file "${test_file}" --out_path "${out_path}" --model "${model}" --height "${resolution}" --width "${resolution}"

export test_file="../test/r2f_prompt/rarebench/rarebench_single_4action_gpt4.txt"
#python inference_r2f.py --test_file "${test_file}" --out_path "${out_path}" --model "${model}" --height "${resolution}" --width "${resolution}"

export test_file="../test/r2f_prompt/rarebench/rarebench_single_5complex_gpt4.txt"
#python inference_r2f.py --test_file "${test_file}" --out_path "${out_path}" --model "${model}" --height "${resolution}" --width "${resolution}"


### Multi-objects
# 1_AND
export test_file="../test/r2f_prompt/rarebench/rarebench_multi_1and_gpt4.txt"
#python inference_r2f.py --test_file "${test_file}" --out_path "${out_path}" --model "${model}" --height "${resolution}" --width "${resolution}"

# 2_Relation
export test_file="../test/r2f_prompt/rarebench/rarebench_multi_2relation_gpt4.txt"
CUDA_VISIBLE_DEVICES=1 python inference_r2f.py --test_file "${test_file}" --out_path "${out_path}" --model "${model}" --height "${resolution}" --width "${resolution}"

# 3_Complex
export test_file="../test/r2f_prompt/rarebench/rarebench_multi_3complex_gpt4.txt"
CUDA_VISIBLE_DEVICES=1 python inference_r2f.py --test_file "${test_file}" --out_path "${out_path}" --model "${model}" --height "${resolution}" --width "${resolution}"





### Region-guided
# 1_AND
#export test_file="../test/r2f_prompt/rarebench/rarebench_multi_1and_gpt4_bbox.txt"
#python ../inference_r2f_multi.py --test_file "${test_file}" --out_path "${out_path}" 