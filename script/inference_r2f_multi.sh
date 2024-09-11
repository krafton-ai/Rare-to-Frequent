export out_path="../images/"

# 1_AND
export test_file="../test/r2f_prompt/rarebench/rarebench_multi_1and_gpt4_2.txt"
# python inference_r2f_multi.py --test_file "${test_file}" --out_path "${out_path}" --save_all
# 2_RELATION
export test_file="../test/r2f_prompt/rarebench/rarebench_multi_2relation_gpt4_2.txt"
# python inference_r2f_multi.py --test_file "${test_file}" --out_path "${out_path}" --save_all
# 3_COMPLEX
export test_file="../test/r2f_prompt/rarebench/rarebench_multi_3complex_gpt4_2.txt"
python inference_r2f_multi.py --test_file "${test_file}" --out_path "${out_path}" --save_all