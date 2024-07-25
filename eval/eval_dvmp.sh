#for data in dvmp_single100 dvmp_multi100
#do
#    for model in sdxl sd3 PixArt
#    do
#        python eval_by_GPT.py --model $model --eval_data $data
#    done
#done

#sd1.5 sd2 


#python eval_by_GPT.py --model SynGen --eval_data 'dvmp_multi100'
python eval_by_GPT.py --model SynGen --eval_data 'dvmp_single100'
#python eval_by_GPT.py --model R2F --eval_data dvmp_single100