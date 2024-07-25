for model in LMD #SynGen R2F sd1.5 sd2 sdxl sd3 PixArt
do
    for data in rarebench_single_1property rarebench_single_2shape rarebench_single_3texture rarebench_single_4action rarebench_multi_1and
    do
        python eval_by_GPT.py --model $model --eval_data $data
    done
done


#python eval_by_GPT.py --model SynGen --eval_data 'rarebench_single_3texture'

#python eval_by_GPT.py --model R2F --eval_data 'rarebench_single_2shape'
#python eval_by_GPT.py --model R2F --eval_data 'rarebench_multi_1and'
#python eval_by_GPT.py --model R2F --eval_data 'rarebench_single_4action'