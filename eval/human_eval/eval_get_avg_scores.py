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
        default="scores/avg_scores_by_human.json",
        help="output file path",
    )
    args = parser.parse_args()
    return args

def get_random_idxs(data, case):

    if data == 'rarebench':
        if case == '1property':
            with open('random_idxs/model_random_idxs_rarebench_single_1property.json', 'r') as f:
                random_idxs = json.load(f)
        elif case == '2shape':
            with open('random_idxs/model_random_idxs_rarebench_single_2shape.json', 'r') as f:
                random_idxs = json.load(f)
        elif case == '3texture':
            with open('random_idxs/model_random_idxs_rarebench_single_3texture.json', 'r') as f:
                random_idxs = json.load(f)
        elif case == '4action':
            with open('random_idxs/model_random_idxs_rarebench_single_4action.json', 'r') as f:
                random_idxs = json.load(f)
        elif case == '5complex':
            with open('random_idxs/model_random_idxs_rarebench_single_5complex.json', 'r') as f:
                random_idxs = json.load(f)
        elif case == '1concat':
            with open('random_idxs/model_random_idxs_rarebench_multi_1and.json', 'r') as f:
                random_idxs = json.load(f)
        elif case == '2relation':
            with open('random_idxs/model_random_idxs_rarebench_multi_2relation.json', 'r') as f:
                random_idxs = json.load(f)
        elif case == '3complex':
            with open('random_idxs/model_random_idxs_rarebench_multi_3complex.json', 'r') as f:
                random_idxs = json.load(f)

    # TODO:
    if data == 'dvmp':
        if case == '1single':
            with open('random_idxs/model_random_idxs_dvmp_single100.json', 'r') as f:
                random_idxs = json.load(f)
        elif case == '2multi':
            with open('random_idxs/model_random_idxs_dvmp_multi100.json', 'r') as f:
                random_idxs = json.load(f)

    return list(random_idxs.values())


def main():
    args = parse_args()

    folder = "scores/"
    #humans = ['sebin', 'minkyu', 'taehong']
    humans = ['set1', 'set2']

    files = [f for f in os.listdir(folder)]
    
    ### Rarebench Evaluation ###
    #data = 'rarebench'
    #cases = ['1property', '2shape', '3texture', '4action', '5complex', '1concat', '2relation', '3complex']

    ### DVMP Evaluation ###
    data = 'dvmp'
    cases = ['1single', '2multi']



    result = {}
    for file in files:        
        if data not in file:
            continue
        
        if data == 'rarebench':
            human = file.split('_')[3].replace('.txt','')
            case = file.split('_')[2]
        elif data == 'dvmp':
            human = file.split('_')[2].replace('.txt','')
            case = file.split('_')[1]
        print(human, case)

        # Get Random idxs
        random_idxs = get_random_idxs(data, case)
        print(random_idxs)

        # Get Human scores
        scores = []
        with open(folder + file, 'r') as f:
            lines = f.readlines()
            scores = [ line.rstrip('\n').split('\t') for line in lines]

            #print(scores)

        # Averaging
        avg_score = np.zeros(len(scores[0]))
        for i in range(len(scores)):

            idx = random_idxs[i]
            score = np.array(scores[i])

            ordered_score = score[np.argsort(idx)].astype(int)-1
            avg_score += ordered_score

        avg_score = avg_score/len(scores)/4*100
        #print(avg_score)

        if case not in result:
            result[case] = avg_score/len(humans)
        else:
            result[case] += avg_score/len(humans)

    print("GET FINAL SCORE")
    for key in result:
        result[key] = list(result[key])
        print(key)
        print(result[key])


    #with open(args.out_file, 'w+') as f:
    #    json.dump(result, f, indent=4)



if __name__ == "__main__":
    main()