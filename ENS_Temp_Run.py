# This program analyzes the results in each log file produces by running EnsembleSplice 
# Once the log file is produced from training or validating or testing, put the log file 
# in a new folder containing this program, and then run this program.

import ast
import numpy as np
import os
import json
import math
import statistics as st
import pandas as pd
import itertools as it

directory = os.getcwd()

COMPARISONS = [
    'cnn1-cnn2','cnn1-cnn3','cnn1-cnn4','cnn1-dnn1','cnn1-dnn2','cnn1-dnn3','cnn1-dnn4',
    'cnn2-cnn3','cnn2-cnn4','cnn2-dnn1','cnn2-dnn2','cnn2-dnn3','cnn2-dnn4',
    'cnn3-cnn4','cnn3-dnn1','cnn3-dnn2','cnn3-dnn3','cnn3-dnn4',
    'cnn4-dnn1','cnn4-dnn2','cnn4-dnn3','cnn4-dnn4',
    'dnn1-dnn2','dnn1-dnn3','dnn1-dnn4',
    'dnn2-dnn3','dnn2-dnn4',
    'dnn3-dnn4',
]

files = sorted(os.listdir(directory), key=lambda elt: '-'.join(elt.split('_')[-5:]))
files.reverse()
for filename in files:
    in_file = os.path.join(directory, filename)
    # checking if it is a file
    print()
    print()
    try:
        with open(in_file, 'r') as f:
            all_lines = f.read()

            # read the fold dictionaries from the files
            remove_n_lines = list(filter(lambda elt: "Train/Predictions" in elt, all_lines.split('\n')))
            just_dicts = [ast.literal_eval(elt.split("FOLD")[0][18:]) for elt in remove_n_lines]

            # the training data used for each fold (so 5)
            y_trns = [[np.argmax([float(selt) for selt in elt.replace('[','').replace(']','').split(', ')]) for elt in _['Train Data'].split("], [")] for _ in just_dicts]

            # the predictions made by each sub-network for each fold (so 5)
            preds = [ast.literal_eval(elt['Predictions']) for elt in just_dicts]

            models = list(preds[0].keys()) # use first fold
            model_compared = [perm for perm in it.permutations(models, 2) if '-'.join(perm) in COMPARISONS]

            # get correlation, double, dis
            o_corr, o_q, o_doub, o_dis = [], [], [], []
            for fold in range(len(preds)):
                correlation, q_stat, double, disagree = [], [], [], []
                for tup in model_compared:
                    corr, q, doub, dis = 0, 0, 0, 0
                    N_00 = 0 
                    N_11 = 0 
                    N_10 = 0
                    N_01 = 0 

                    # go through the training data and the predictions
                    for index, elt in enumerate(list(zip(preds[fold][tup[0]], preds[fold][tup[1]]))):
                        if (elt[0] == elt[1]) and (elt[0] == y_trns[fold][index]) and (elt[1] ==  y_trns[fold][index]):
                            N_11 += 1
                        if (elt[0] == elt[1]) and (elt[0] != y_trns[fold][index]) and (elt[1] !=  y_trns[fold][index]):
                            N_00 += 1
                        if (elt[0] != elt[1]) and (elt[0] == y_trns[fold][index]) and (elt[1] !=  y_trns[fold][index]):
                            N_10 += 1
                        if (elt[0] != elt[1]) and (elt[0] != y_trns[fold][index]) and (elt[1] == y_trns[fold][index]):
                            N_01 += 1
                    try:
                        q = ((N_11*N_00) - (N_01*N_10))/((N_11*N_00)+(N_01*N_10))
                        corr = ((N_11*N_00) - (N_01*N_10)) / math.sqrt((N_11+N_10)*(N_01+N_00)*(N_11+N_01)*(N_10+N_00))
                        dis = (N_01 + N_10) / (N_11 + N_10 + N_01 + N_00)
                        doub = (N_00 / (N_11 + N_10 + N_01 + N_00))
                    except ZeroDivisionError:
                        q = 0
                    correlation.append(corr)
                    q_stat.append(q)
                    double.append(doub)
                    disagree.append(dis)

                o_corr.append(st.mean(correlation))
                o_q.append(st.mean(q_stat))
                o_doub.append(st.mean(double))
                o_dis.append(st.mean(disagree))
            print(f"Double Fault: {st.mean(o_doub)}")
            print(f"Correlation: {st.mean(o_corr)}")
            print(f"Q-Statistics: {st.mean(o_q)}")
            print(f"Disagreement: {st.mean(o_dis)}")


            ## Sub-model accuracy for each ensemble
            
            if 'hs3d-bal' not in in_file:
                table = ast.literal_eval(all_lines.split('\n')[-1].strip())
                ds = filename.split('_')[3]
                ss = filename.split('_')[4]
                net = filename.split('_')[5]
                print(net, ds, ss)
                for model in table[ds][ss].keys():
                    avg_fold_acc = st.mean([elt[-1] for elt in table[ds][ss][model]['val_bin_acc']])
                    print(model, avg_fold_acc)
            else:
                table = ast.literal_eval(all_lines.split('\n')[-1].strip())
                ds = "hs3d_bal"
                ss = filename.split('_')[4]
                net = filename.split('_')[5]
                print(net, ds, ss)
                for model in table[ds][ss].keys():
                    avg_fold_acc = st.mean([elt[-1] for elt in table[ds][ss][model]['val_bin_acc']])
                    print(model, avg_fold_acc)
            f.close()
            print("=============================================")
    except:
        continue
