export out_path="../images/"
export model='sdxl'

### RareBench
data_list=(
        "rarebench_single_1property_gpt4.txt"
        "rarebench_single_2shape_gpt4.txt"
        "rarebench_single_3texture_gpt4.txt"
        "rarebench_single_4action_gpt4.txt"
        "rarebench_single_5complex_gpt4.txt"
        "rarebench_multi_1and_gpt4.txt"
        "rarebench_multi_2relation_gpt4.txt"
        "rarebench_multi_3complex_gpt4.txt"
    )

datalen=${#data_list[@]}
for ((j=0; j<datalen; j++))
do
    data=${data_list[$j]}
    python inference_r2f.py \
        --test_file ../test/r2f_prompt/rarebench/$data \
        --out_path $out_path \
        --model $model
done

### DVMP
data_list=(
        "dvmp_single100_gpt4.txt"
        "dvmp_multi100_gpt4.txt"
    )

datalen=${#data_list[@]}
for ((j=0; j<datalen; j++))
do
    data=${data_list[$j]}
    python inference_r2f.py \
        --test_file ../test/r2f_prompt/dvmp/$data \
        --out_path $out_path \
        --model $model
done


### T2I-CompBench
data_list=(
        "compbench_1color_val_gpt4.txt"
        "compbench_2shape_val_gpt4.txt"
        "compbench_3texture_val_gpt4.txt"
        "compbench_4spatial_val_gpt4.txt"
        "compbench_5non_spatial_val_gpt4.txt"
        "compbench_6complex_val_gpt4.txt"
    )

datalen=${#data_list[@]}
for ((j=0; j<datalen; j++))
do
    data=${data_list[$j]}
    python inference_r2f.py \
        --test_file ../test/r2f_prompt/compbench/$data \
        --out_path $out_path \
        --model $model
done