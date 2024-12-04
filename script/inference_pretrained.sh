export out_path="../images/"
model_list=(
    "runwayml/stable-diffusion-v1-5"
    "stabilityai/stable-diffusion-2-base"
    "stabilityai/stable-diffusion-xl-base-1.0"
    "PixArt-alpha/PixArt-XL-2-1024-MS"
    "stabilityai/stable-diffusion-3-medium"
    "black-forest-labs/FLUX.1-schnell"
)


### RareBench
data_list=(
        "rarebench_single_1property.txt"
        "rarebench_single_2shape.txt"
        "rarebench_single_3texture.txt"
        "rarebench_single_4action.txt"
        "rarebench_single_5complex.txt"
        "rarebench_multi_1and.txt"
        "rarebench_multi_2relation.txt"
        "rarebench_multi_3complex.txt"
    )

modellen=${#model_list[@]}
datalen=${#data_list[@]}
for ((j=0; j<modellen; j++))
do
    for ((j=0; j<datalen; j++))
    do
        model=${model_list[$i]}
        data=${data_list[$j]}
        python inference_pretrained.py \
            --test_file ../test/original_prompt/rarebench/$data \
            --out_path $out_path \
            --pretrained_model_path $model
    done
done



### DVMP
data_list=(
        "dvmp_single100.txt"
        "dvmp_multi100.txt"
    )

modellen=${#model_list[@]}
datalen=${#data_list[@]}
for ((j=0; j<modellen; j++))
do
    for ((j=0; j<datalen; j++))
    do
        model=${model_list[$i]}
        data=${data_list[$j]}
        python inference_pretrained.py \
            --test_file ../test/original_prompt/dvmp/$data \
            --out_path $out_path \
            --pretrained_model_path $model
    done
done


### T2I-CompBench
data_list=(
        "compbench_1color_val.txt"
        "compbench_2shape_val.txt"
        "compbench_3texture_val.txt"
        "compbench_4spatial_val.txt"
        "compbench_5non_spatial_val.txt"
        "compbench_6complex_val.txt"
    )

modellen=${#model_list[@]}
datalen=${#data_list[@]}
for ((j=0; j<modellen; j++))
do
    for ((j=0; j<datalen; j++))
    do
        model=${model_list[$i]}
        data=${data_list[$j]}
        python inference_pretrained.py \
            --test_file ../test/original_prompt/compbench/$data \
            --out_path $out_path \
            --pretrained_model_path $model
    done
done