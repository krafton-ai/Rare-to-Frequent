export out_path="../images/"

### Single-object
export test_file="../test_r2f/dvmp/dvmp_single100_gpt4.txt"
python ../inference_r2f_single.py --test_file "${test_file}" --out_path "${out_path}" --visual-detail-aware