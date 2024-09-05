export out_path="../images/"


### Single-object
#export test_file="../test/rpg_prompt/rarebench/rarebench_single_1property_gpt4.txt"
#python inference_rpg.py --test_file "${test_file}" --out_path "${out_path}" 

#export test_file="../test/rpg_prompt/rarebench/rarebench_single_2shape_gpt4.txt"
#python inference_rpg.py --test_file "${test_file}" --out_path "${out_path}" 

#export test_file="../test/rpg_prompt/rarebench/rarebench_single_3texture_gpt4.txt"
#python inference_rpg.py --test_file "${test_file}" --out_path "${out_path}" 

#export test_file="../test/rpg_prompt/rarebench/rarebench_single_4action_gpt4.txt"
#python inference_rpg.py --test_file "${test_file}" --out_path "${out_path}" 

#export test_file="../test/rpg_prompt/rarebench/rarebench_single_5complex_gpt4.txt"
#python inference_rpg.py --test_file "${test_file}" --out_path "${out_path}" 


### Multi-objects
# 1_AND
#export test_file="../test/rpg_prompt/rarebench/rarebench_multi_1and_gpt4.txt"
#python inference_rpg.py --test_file "${test_file}" --out_path "${out_path}" 

#export test_file="../test/rpg_prompt/rarebench/rarebench_multi_2relation_gpt4.txt"
#python inference_rpg.py --test_file "${test_file}" --out_path "${out_path}" 

export test_file="../test/rpg_prompt/rarebench/rarebench_multi_3complex_gpt4.txt"
python inference_rpg.py --test_file "${test_file}" --out_path "${out_path}" 