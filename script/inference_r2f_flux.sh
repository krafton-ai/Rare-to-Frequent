export out_path="../images/"
export model='flux' #'sdxl'
export resolution=1024

steps=(4)
alphas_list=("0.3,0.15,0.15,0.15")

for step in "${steps[@]}"; do
    for alpha in "${alphas_list[@]}"; do
        ### Single-object
        export test_file="../test/r2f_prompt/rarebench/rarebench_single_1property_gpt4.txt"
        CUDA_VISIBLE_DEVICES=0 python inference_r2f_flux.py --test_file "${test_file_2}" --out_path "${out_path}" --model "${model}" --height "${resolution}" --width "${resolution}" --alphas "${alpha}" --num_inference_steps "${step}"

        # export test_file="../test/r2f_prompt/rarebench/rarebench_single_2shape_gpt4.txt"
        # CUDA_VISIBLE_DEVICES=0 python inference_r2f_flux.py --test_file "${test_file_2}" --out_path "${out_path}" --model "${model}" --height "${resolution}" --width "${resolution}" --alphas "${alpha}" --num_inference_steps "${step}"

        # export test_file="../test/r2f_prompt/rarebench/rarebench_single_3texture_gpt4.txt"
        # CUDA_VISIBLE_DEVICES=0 python inference_r2f_flux.py --test_file "${test_file_2}" --out_path "${out_path}" --model "${model}" --height "${resolution}" --width "${resolution}" --alphas "${alpha}" --num_inference_steps "${step}"

        # export test_file="../test/r2f_prompt/rarebench/rarebench_single_4action_gpt4.txt"
        # CUDA_VISIBLE_DEVICES=0 python inference_r2f_flux.py --test_file "${test_file_2}" --out_path "${out_path}" --model "${model}" --height "${resolution}" --width "${resolution}" --alphas "${alpha}" --num_inference_steps "${step}"
    done
done
    