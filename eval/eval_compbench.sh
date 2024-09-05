for model in sd1.5 sd2 sdxl sd3 PixArt SynGen LMD ELLA R2F  #sd1.5 sd2 sdxl sd3 PixArt SynGen LMD RPG R2F 
do
    for data in compbench_1color_val compbench_2shape_val compbench_3texture_val compbench_4spatial_val compbench_5non_spatial_val compbench_6complex_val
    do
        python eval_by_GPT.py --model $model --eval_data $data
    done
done



# Data
# compbench_1color_val 
# compbench_2shape_val 
# compbench_3texture_val 
# compbench_4spatial_val 
# compbench_5non_spatial_val
# compbench_6complex_val



#python eval_by_GPT.py --model R2F --eval_data 'rarebench_single_2shape'
#python eval_by_GPT.py --model R2F --eval_data 'rarebench_single_4action'
#python eval_by_GPT.py --model R2F --eval_data 'rarebench_multi_1and'