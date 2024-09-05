import requests
import json
import os
import numpy as np
import argparse
import base64

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out_file",
        type=str,
        nargs="?",
        default="result/avg_scores_0to4_by_gpt4o.json",
        help="output file path",
    )
    args = parser.parse_args()
    return args

def main():
    args = parse_args()

    folder = "result"
    subfolders = [ f.path + '/' for f in os.scandir(folder) if f.is_dir() ]
    
    # Evaluation
    result = {}
    for subfolder in subfolders:
        files = [f for f in os.listdir(subfolder) if os.path.isfile(os.path.join(subfolder, f))]
        
        for file in files:
            
            # FIXME:
            if not 'gpt4o' in file:
                continue

            with open(subfolder + file, 'r') as f:
                scores = json.load(f)
                
                avg_score = 0
                for key in scores:
                    avg_score += max(scores[key]['score']-1 , 0)

                avg_score = avg_score/len(scores)/4*100
                
                result[subfolder.split("/")[1] + '/' + file.replace(".json","")] = avg_score
    print(result)

    with open(args.out_file, 'w+') as f:
        json.dump(result, f, indent=4)



if __name__ == "__main__":
    main()