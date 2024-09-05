export out_path="../images/"


export test_file="../test/r2f_prompt/compbench/compbench_1color_val_gpt4.txt"
python inference_r2f_multi.py --test_file "${test_file}" --out_path "${out_path}" 

#export test_file="../test/r2f_prompt/compbench/compbench_2shape_val_gpt4.txt"
#python inference_r2f_multi.py --test_file "${test_file}" --out_path "${out_path}" 

#export test_file="../test/r2f_prompt/compbench/compbench_3texture_val_gpt4.txt"
#python inference_r2f_multi.py --test_file "${test_file}" --out_path "${out_path}" 

#export test_file="../test/r2f_prompt/compbench/compbench_4spatial_val_gpt4.txt"
#python inference_r2f_multi.py --test_file "${test_file}" --out_path "${out_path}" 

#export test_file="../test/r2f_prompt/compbench/compbench_5non_spatial_val_gpt4.txt"
#python inference_r2f_multi.py --test_file "${test_file}" --out_path "${out_path}" 

#export test_file="../test/r2f_prompt/compbench/compbench_6complex_val_gpt4.txt"
#python inference_r2f_multi.py --test_file "${test_file}" --out_path "${out_path}" 