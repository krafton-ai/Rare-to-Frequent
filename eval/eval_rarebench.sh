# Models: sd1.5 sd2 sdxl sd3 PixArt SynGen LMD RPG ELLA P2P Composable Para 
# R2F_sd15 R2F_sdxl R2F_sd3 R2F_sd3_llama3 R2F_sd3_fix5 R2F_sd3_fix10 R2F_sd3_fix20 R2F_sd3_fix30 R2F_sd3_fix40 R2F_sd3_para
for model in Para R2F_sd3_para
do
    for data in rarebench_single_1property rarebench_single_2shape rarebench_single_3texture rarebench_single_4action rarebench_single_5complex rarebench_multi_1and rarebench_multi_2relation rarebench_multi_3complex
    do
        python eval_by_GPT.py --model $model --eval_data $data
    done
done


# Data
# rarebench_single_1property 
# rarebench_single_2shape 
# rarebench_single_3texture 
# rarebench_single_4action 
# rarebench_single_5complex
# rarebench_multi_1and
# rarebench_multi_2relation
# rarebench_multi_3complex



#python eval_by_GPT.py --model R2F --eval_data 'rarebench_single_2shape'
#python eval_by_GPT.py --model R2F --eval_data 'rarebench_single_4action'
#python eval_by_GPT.py --model R2F --eval_data 'rarebench_multi_1and'