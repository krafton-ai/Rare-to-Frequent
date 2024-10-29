import random
import os 
import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import json

eval_cases = ['rarebench_single_1property', 'rarebench_single_2shape', 'rarebench_single_3texture', \
        'rarebench_single_4action', 'rarebench_single_5complex', 'rarebench_multi_1and', \
        'rarebench_multi_2relation', 'rarebench_multi_3complex', 'dvmp_single100', 'dvmp_multi100']

model_names = ['stable-diffusion-v1-5', 'stable-diffusion-2-base', 'stable-diffusion-xl-base-1.0', \
        'PixArt-XL-2-1024-MS', 'stable-diffusion-3-medium', 'SynGen-sd1.5', \
        'LMD-sdxl', 'RPG-sdxl', 'ELLA', 'R2F-sd3', 'FLUX.1-schnell'] #, 'FLUX.1-schnell'
num_models = len(model_names)



file_names = {}
for eval_case in eval_cases:
    img_folder = f'../images/{model_names[0]}/{eval_case}/'
    files = os.listdir(img_folder)
    
    idxs = [int(file.split('_')[0]) for file in files]

    idx_names = []
    for i in np.argsort(idxs):
        file = files[i]
        idx = int(file.split('_')[0])
        name = file.split('_')[1].replace('.png','')
        
        idx_names.append([idx, name, file])

    file_names[eval_case] = idx_names
#print(file_names)

Done_cases = ['rarebench_single_1property', 'rarebench_single_2shape', 'rarebench_single_3texture', \
        'rarebench_single_4action', 'rarebench_single_5complex', 'rarebench_multi_1and', \
        'rarebench_multi_2relation', 'rarebench_multi_3complex']

# image path
for eval_case in eval_cases:

    if eval_case in Done_cases:
        continue

    model_random_idxs = {}
    for i, file_name in enumerate(file_names[eval_case]):
        print(file_name)
        
        # random shuffling model_idxs
        idxs = list(range(num_models))
        np.random.shuffle(idxs)
        print(idxs)
        model_random_idxs[file_name[2]] = idxs
        
        # save path
        if i % 10 == 0:
            out_file = f'human_eval/images/{eval_case}_{i}to{i+9}.jpg'
            fig, axs = plt.subplots(10, num_models, figsize=(50, 55))

        ### Make image grid!
        imgs = []
        for j, idx in enumerate(idxs):
            model_name = model_names[idx]
            file_path = f'../images/{model_name}/{eval_case}/{file_name[2]}'

            image = Image.open(file_path).resize((512,512))
            
            order = num_models*(i%10) + j+1
            print(order)

            ax = plt.subplot(10, num_models, order)

            # subtitles
            if j == 5:
                ax.set_title(f'{file_name[0]}_{file_name[1]}', fontsize=40) 

            # labels
            plt.xlabel(f'model {j}', fontsize=30)
            #if j == 0:
            #    plt.ylabel(f'image {i}', fontsize=30)

            ax.set_xticks([])
            ax.set_yticks([])
            
            plt.tight_layout()

            plt.imshow(image)
        
        # save
        if i % 10 == 9:
            plt.savefig(out_file)

    # save random idx
    print(model_random_idxs)
    with open(f'human_eval/random_idxs/model_random_idxs_{eval_case}.json', 'w') as out_file:
        json.dump(model_random_idxs, out_file, indent=4)

    #break
    