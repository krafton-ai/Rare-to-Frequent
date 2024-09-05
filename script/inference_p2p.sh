export out_path="../images/"

### Single
# Attributes_1_property
export test_file="../test/p2p_prompt/rarebench/rarebench_single_1property_gpt4.txt"
#python inference_p2p.py --test_file "${test_file}" --out_path "${out_path}" 


# Attributes_2_property
export test_file="../test/p2p_prompt/rarebench/rarebench_single_2shape_gpt4.txt"
#python inference_p2p.py --test_file "${test_file}" --out_path "${out_path}" 


# Attributes_3_texture
export test_file="../test/p2p_prompt/rarebench/rarebench_single_3texture_gpt4.txt"
#python inference_p2p.py --test_file "${test_file}" --out_path "${out_path}" 


# 4_Actions
export test_file="../test/p2p_prompt/rarebench/rarebench_single_4action_gpt4.txt"
#python inference_p2p.py --test_file "${test_file}" --out_path "${out_path}" 


# 5_Complex
export test_file="../test/p2p_prompt/rarebench/rarebench_single_5complex_gpt4.txt"
python inference_p2p.py --test_file "${test_file}" --out_path "${out_path}" 



### Multi-objects
# 1_AND
export test_file="../test/p2p_prompt/rarebench/rarebench_multi_1and_gpt4.txt"
#python inference_p2p.py --test_file "${test_file}" --out_path "${out_path}" 


# 2. Relation
export test_file="../test/p2p_prompt/rarebench/rarebench_multi_2relation_gpt4.txt"
#python inference_p2p.py --test_file "${test_file}" --out_path "${out_path}" 


# 3. Complex
export test_file="../test/p2p_prompt/rarebench/rarebench_multi_3complex_gpt4.txt"
#python inference_p2p.py --test_file "${test_file}" --out_path "${out_path}" 