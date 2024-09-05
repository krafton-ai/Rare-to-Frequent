export out_path="../images/"

### Rarebench
### Single-object
export test_file="../test/r2f_prompt/rarebench/rarebench_single_1property_gpt4.txt"
#python inference_composable.py --test_file "${test_file}" --out_path "${out_path}" 

export test_file="../test/r2f_prompt/rarebench/rarebench_single_2shape_gpt4.txt"
python inference_composable.py --test_file "${test_file}" --out_path "${out_path}" 

export test_file="../test/r2f_prompt/rarebench/rarebench_single_3texture_gpt4.txt"
#python inference_composable.py --test_file "${test_file}" --out_path "${out_path}" 

export test_file="../test/r2f_prompt/rarebench/rarebench_single_4action_gpt4.txt"
#python inference_composable.py --test_file "${test_file}" --out_path "${out_path}" 

export test_file="../test/r2f_prompt/rarebench/rarebench_single_5complex_gpt4.txt"
#python inference_composable.py --test_file "${test_file}" --out_path "${out_path}" 


### Multi-objects
# 1_AND
export test_file="../test/r2f_prompt/rarebench/rarebench_multi_1and_gpt4.txt"
#python inference_composable.py --test_file "${test_file}" --out_path "${out_path}" 

# 2_Relation
export test_file="../test/r2f_prompt/rarebench/rarebench_multi_2relation_gpt4.txt"
#python inference_composable.py --test_file "${test_file}" --out_path "${out_path}" 

# 3_Complex
export test_file="../test/r2f_prompt/rarebench/rarebench_multi_3complex_gpt4.txt"
#python inference_composable.py --test_file "${test_file}" --out_path "${out_path}" 



### DVMP
### Single-object
#export test_file="../test/r2f_prompt/dvmp/dvmp_single100_gpt4.txt"
#python inference_composable.py --test_file "${test_file}" --out_path "${out_path}"

### Multi-object
#export test_file="../test/r2f_prompt/dvmp/dvmp_multi100_gpt4.txt"
#python inference_composable.py --test_file "${test_file}" --out_path "${out_path}"



### Compbench
#export test_file="../test/r2f_prompt/compbench/compbench_1color_val_gpt4.txt"
#python inference_composable.py --test_file "${test_file}" --out_path "${out_path}" 

#export test_file="../test/r2f_prompt/compbench/compbench_2shape_val_gpt4.txt"
#python inference_composable.py --test_file "${test_file}" --out_path "${out_path}" 

#export test_file="../test/r2f_prompt/compbench/compbench_3texture_val_gpt4.txt"
#python inference_composable.py --test_file "${test_file}" --out_path "${out_path}" 

#export test_file="../test/r2f_prompt/compbench/compbench_4spatial_val_gpt4.txt"
#python inference_composable.py --test_file "${test_file}" --out_path "${out_path}" 

#export test_file="../test/r2f_prompt/compbench/compbench_5non_spatial_val_gpt4.txt"
#python inference_composable.py --test_file "${test_file}" --out_path "${out_path}" 

#export test_file="../test/r2f_prompt/compbench/compbench_6complex_val_gpt4.txt"
#python inference_composable.py --test_file "${test_file}" --out_path "${out_path}" 