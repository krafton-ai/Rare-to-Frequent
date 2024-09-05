for model in sd1.5 sd2 sdxl sd3 PixArt SynGen LMD RPG ELLA R2F #sd1.5 sd2 sdxl sd3 PixArt LMD SynGen R2F 
do
    for data in dvmp_single100 dvmp_multi100
    do
        python eval_by_GPT.py --model $model --eval_data $data
    done
done



#python eval_by_GPT.py --model R2F --eval_data 'dvmp_multi100'

#python eval_by_GPT.py --model SynGen --eval_data 'dvmp_multi100'
#python eval_by_GPT.py --model SynGen --eval_data 'dvmp_single100'
#python eval_by_GPT.py --model R2F --eval_data dvmp_single100