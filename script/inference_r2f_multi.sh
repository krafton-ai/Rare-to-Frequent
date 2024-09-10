export out_path="../images/"

# 1_AND
export test_file="../test/r2f_prompt/rarebench/rarebench_multi_1and_gpt4_2.txt"
python inference_r2f_multi.py --test_file "${test_file}" --out_path "${out_path}"