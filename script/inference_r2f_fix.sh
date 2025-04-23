export out_path="../images/"
export model='sd3'
export resolution=1024
#export transition_step=5

for transition_step in 5 #10 20 30 40
do
    ### Single-object
    export test_file="../test/r2f_prompt/rarebench/rarebench_single_1property_gpt4.txt"
    python inference_r2f_fix.py --test_file "${test_file}" --out_path "${out_path}" --transition_step "${transition_step}" --r2f-generator 'human'

    export test_file="../test/r2f_prompt/rarebench/rarebench_single_2shape_gpt4.txt"
    #python inference_r2f_fix.py --test_file "${test_file}" --out_path "${out_path}" --transition_step "${transition_step}" --r2f-generator 'human'

    export test_file="../test/r2f_prompt/rarebench/rarebench_single_3texture_gpt4.txt"
    #python inference_r2f_fix.py --test_file "${test_file}" --out_path "${out_path}" --transition_step "${transition_step}" --r2f-generator 'human'

    export test_file="../test/r2f_prompt/rarebench/rarebench_single_4action_gpt4.txt"
    #python inference_r2f_fix.py --test_file "${test_file}" --out_path "${out_path}" --transition_step "${transition_step}" --r2f-generator 'human'
done