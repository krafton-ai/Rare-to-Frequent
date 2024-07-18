import pandas as pd
import numpy as np
import os
import sys
import json
from tqdm import tqdm 

imgnet_path = "../../data_preprocess/data/laion400M/imgnet1k_classes.txt"
with open(imgnet_path, "r") as f:
    imgnet_idx_class = f.readlines()

imgnet_classes = []
class_cnt = {}
for idx_class in imgnet_idx_class:
    label = idx_class.split(": ")[1].split(",")[0].replace('\'', '').replace("\"", '').lower()
    imgnet_classes.append(label)

    class_cnt[label] = 0

print(imgnet_classes)
print("# imgnet classes: ", len(imgnet_classes))


laion_path = "../../data_preprocess/data/laion400M/captions.txt"
if not os.path.isfile(laion_path):
    metadata_folder = "../../data_preprocess/data/laion400M/the-eye.eu/public/AI/cah/laion400m-met-release/laion400m-meta/"
    files = os.listdir(metadata_folder)

    cnt = 0
    out_path = "../../data_preprocess/data/laion400M/captions.txt"
    with open(out_path, "w") as f:
        for i, file in enumerate(files):
            if '.parquet' in file:
                print(i, metadata_folder + file)
                data = pd.read_parquet(metadata_folder + file)
                captions = data[['TEXT']].apply(lambda x: x.replace("'", " ").replace('"', " ")).to_numpy()#.reshape(-1)

                for cap in captions:
                    try:
                        f.write(cap[0]+'\n')
                        cnt +=1
                    except Exception as err:
                        print(err)
                        print("cap: ", cap)

    print("Total # of captions: ", cnt)
else:
    print(f"{laion_path} exists !!")
    with open(laion_path, "r") as f:
        captions = f.readlines()
    
    print(len(captions))
    


    for i, cap in tqdm(enumerate(captions), total=len(captions)):
        cap = cap.lower()

        for label in imgnet_classes:
            if ' '+label+' ' in cap:
                class_cnt[label] += 1
    
    print(class_cnt)

    out_path = "../../data_preprocess/data/laion400M/imgnet_class_cnt_in_laion400m.json"
    with open(out_path, "w") as f:
        json.dump(class_cnt, f, indent=4)