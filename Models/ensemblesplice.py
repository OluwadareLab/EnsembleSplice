# -----------------------------------------------------------------------------
# Copyright (c) 2022 Trevor P. Martin. All rights reserved.
# Distributed under the MIT License.
# -----------------------------------------------------------------------------
from Data import encode_data
from Models import sub_models
from Visuals import make_report


from datetime import datetime
import numpy as np
import pandas as pd
import os
import math
import tensorflow as tf
import copy
import statistics as st
import itertools as it

import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.metrics import auc, plot_precision_recall_curve
from sklearn.metrics import plot_precision_recall_curve
from sklearn.metrics import accuracy_score
from sklearn.linear_model import Perceptron
from sklearn.svm import LinearSVC
from scipy.stats.stats import pearsonr # correlation coefficient
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.feature_selection import f_classif


# this is the default random state across all files
RANDOM_STATE = 123432


COMPARISONS = [
    'cnn1-cnn2','cnn1-cnn3','cnn1-cnn4','cnn1-dnn1','cnn1-dnn2','cnn1-dnn3','cnn1-dnn4',
    'cnn2-cnn3','cnn2-cnn4','cnn2-dnn1','cnn2-dnn2','cnn2-dnn3','cnn2-dnn4',
    'cnn3-cnn4','cnn3-dnn1','cnn3-dnn2','cnn3-dnn3','cnn3-dnn4',
    'cnn4-dnn1','cnn4-dnn2','cnn4-dnn3','cnn4-dnn4',
    'dnn1-dnn2','dnn1-dnn3','dnn1-dnn4',
    'dnn2-dnn3','dnn2-dnn4',
    'dnn3-dnn4',
]

# define the classification performance metrics
# first 4 (model history), next 9 (keras classification metrics), last 7 (mine)
METRICS = {
    'loss':[], # train loss,
    'bin_acc':[], # train accuracy
    'val_bin_acc':[], # validation (val) accuracy, (could be test acc too)
    'val_loss':[], # validation loss, (could be test loss too)
    'bin_cross':[], # training binary crossentropy
    'val_bin_cross':[], # val binary crossentropy,
    'auc':[], # Area Under ROC Curve
    'val_auc':[], # Val Area Under ROC Curve
    'pre':[], # Precision
    'val_pre':[], # Val Precision
    'rec':[], # Recall,
    'val_rec':[], # Val Recall,
    'tru_pos':[], # True Positives,
    'tru_neg':[], # True Negatives,
    'fal_pos':[], # False Positives,
    'fal_neg':[], # False Negatives,
    'val_tru_pos':[], # Val True Positives,
    'val_tru_neg':[], # Val True Negatives,
    'val_fal_pos':[], # Val False Positives,
    'val_fal_neg':[], # Val False Negatives,
}


def ensemble(ds, ss, fold_count, model_preds_by_fold, y_trn, y_val, results_table, models, log_file, kwargs):

    # each list entry appears as [(model, trn_pred, val_pred)... x num models]
    # this is done by fold for validation
    if kwargs["operation"] == "validate":

        # instantiate new lists for this fold's ensemble metrics
        correlations = []
        q_statistics = []
        differences = []
        double_faults = []

        # putting the data into pd.DataFrame makes it easier to work with in terms of metrics
        training_data = dict([(elt[0], [np.argmax(x) for x in elt[1]]) for elt in model_preds_by_fold])
        validation_data = dict([(elt[0], [np.argmax(x) for x in elt[2]]) for elt in model_preds_by_fold])

        # ensemble logging
        data = {"Train Data":str(y_trn.tolist()), "Predictions":str(training_data)}
        with open(log_file, 'a') as f:
            f.write(f'Train/Predictions:{str(data)}')
            f.write("\n")
            #f.write(f'FOLD: {fold_count}: touched: {X_train}\nuntouched: {np.array([elt[1] for elt in model_preds_by_fold])}\n')
            f.close()

        # make the values ready for use in the ensemble network
        # each "elt" is a (model, trn_preds, val_preds)
        X_train = np.vstack([np.array([np.argmax(x) for x in elt[1]]) for elt in model_preds_by_fold]).T
        X_valid = np.vstack([np.array([np.argmax(x) for x in elt[2]]) for elt in model_preds_by_fold]).T
        row_length = X_train.shape[1]

        # a new row length needed for the ensemble, which consists of simple logistic regression

        print(f"\n{'-'*5}ENSEMBLE{'-'*5}\n")

        model_history, trained_model = models["ensemble"].build(
            row_length,
            X_train, y_trn,
            X_valid, y_val,
            kwargs,
        )

        # ensemble metrics
        for metric in METRICS.keys():
            results_table[ds][ss]["ensemble"][metric].append(model_history.history[metric])

        # ensemble logging
        with open(log_file, 'a') as f:
            f.write(f'FOLD: {fold_count}; Dataset: {ds}; Splice Site: {ss}; Model: Ensemble; Data: {str(results_table[ds][ss]["ensemble"])}')
            f.write("\n\n\n")
            #f.write(f'FOLD: {fold_count}: touched: {X_train}\nuntouched: {np.array([elt[1] for elt in model_preds_by_fold])}\n')
            f.close()

        return results_table

    if kwargs["operation"] == "train":
        trained_models = '-'.join(kwargs["sub_models"])
        # putting the data into pd.DataFrame makes it easier to work with in terms of metrics
        training_data = dict([(elt[0], [np.argmax(x) for x in elt[1]]) for elt in model_preds_by_fold])
        # make the values ready for use in the ensemble network
        # each "elt" is a (model, trn_preds, val_preds)
        X_train = np.vstack([np.array([np.argmax(x) for x in elt[1]]) for elt in model_preds_by_fold]).T
        row_length = X_train.shape[1]
        # a new row length needed for the ensemble, which consists of simple logistic regression
        print(f"\n{'-'*5}ENSEMBLE{'-'*5}\n")
        model_history, trained_model = models["ensemble"].build(
            row_length,
            X_train, y_trn, [], [],
            kwargs,
        )
        trained_model.save(os.getcwd()+f'/Models/TrainedModels/Ensemble_{trained_models}_{ds}_{ss}')
        #ensemble metrics
        train_metrics = list(filter(lambda elt: "val" not in elt, list(METRICS.keys())))
        for metric in train_metrics:
            results_table[ds][ss]["ensemble"][metric].append(model_history.history[metric])
        # ensemble logging
        with open(log_file, 'a') as f:
            f.write(f'Dataset: {ds}; Splice Site: {ss}; Model: Ensemble; Data: {str(results_table[ds][ss]["ensemble"])}')
            f.write("\n\n\n")
            f.close()
        return results_table

    if kwargs["operation"] == "test":
        trained_models = '-'.join(kwargs["sub_models"])
        testing_data = dict([(elt[0], [np.argmax(x) for x in elt[1]]) for elt in model_preds_by_fold])
        X_test = np.vstack([np.array([np.argmax(x) for x in elt[1]]) for elt in model_preds_by_fold]).T
        row_length = X_test.shape[1]
        print(f"\n{'-'*5}ENSEMBLE{'-'*5}\n")
        loaded_model = tf.keras.models.load_model(
            os.getcwd()+f'/Models/TrainedModels/Ensemble_{trained_models}_{ds}_{ss}/'
        )
        score = loaded_model.evaluate(X_test, y_trn) # really y_tst

        ens_tst_preds = loaded_model.predict(
            X_test,
            batch_size=32,
        )
        tp, tn, fp, fn = 0,0,0,0
        for i in range(len(ens_tst_preds)):
            if np.argmax(y_trn[i])==np.argmax(ens_tst_preds[i])==1:
               tp += 1
            if np.argmax(ens_tst_preds[i])==1 and np.argmax(y_trn[i])!=np.argmax(ens_tst_preds[i]):
               fp += 1
            if np.argmax(y_trn[i])==np.argmax(ens_tst_preds[i])==0:
               tn += 1
            if np.argmax(ens_tst_preds[i])==0 and np.argmax(y_trn[i])!=np.argmax(ens_tst_preds[i]):
               fn += 1

        tpr = tp / (tp + fn)
        fpr = fp / (fp + tn)
        f1 = (2*tp) / ((2*tp) + fp + fn)
        pre = (tp) / (tp + fp)
        sn = (tp) / (tp + fn)
        sp = (tn) / (tn + fp)
        acc = (tp + tn) / (tp + tn + fp + fn)
        err = 1 - acc
        mcc = ((tp * tn) - (fp * fn)) / math.sqrt((tp + fp)*(tp + fn)*(tn + fp)*(tn + fn))

        # train_metrics = list(filter(lambda elt: "val" not in elt, list(METRICS.keys())))
        # for i, metric in enumerate(train_metrics):
        #     results_table[ds][ss]["ensemble"][metric].append(score[i])
        # ensemble logging
        with open(log_file, 'a') as f:
            f.write(str({"Sp":sp, "Sn":sn,"Pre":pre,"Err":err,"Acc":acc,"Mcc":mcc,"F1":f1}))
            f.write("\n\n\n")
            f.close()
        return results_table


def validate(encoded_datasets, results_table, models, kwargs, log_file, graph_folder):
    # create the fold function
    k_fold = KFold(
        n_splits=kwargs['folds'],
        shuffle=True,
        random_state=RANDOM_STATE
    )
    # validation loop
    for ds in kwargs['datasets']:
        for ss in kwargs['splice_sites']:
            # get the encoded data by ds, ss
            X_train, y_train = encoded_datasets[ds][ss]
            # create an ensemble predictions list for each dataset, each splice site
            ens_sub_preds = []
            # iterate through the folded dataset
            fold_count = 1
            for train, test in k_fold.split(X_train, y_train):
                print(f"\n{'#'*30}\nENTERING {ss.upper()} {ds.upper()} FOLD {fold_count}\n{'#'*30}\n")
                X_trn, y_trn = X_train[train], y_train[train]
                X_val, y_val = X_train[test], y_train[test]
                # length of the nucleotide sequences, differs by dataset, needed for NN dimensions
                row_length = len(X_trn[0])
                # training and validation predictions for ensemble
                model_preds_by_fold = []
                # build each of the sub-models NNs
                for i, m in enumerate(kwargs['sub_models']):
                    print(f"\n{'-'*5+m.upper()+'-'*5}\n")
                    model_history, trained_model = models[m].build(
                        row_length,
                        X_trn, y_trn, X_val, y_val,
                        kwargs,
                    )
                    # add a list of length equal to the number of epochs to the metric
                    for metric in METRICS.keys():
                        results_table[ds][ss][m][metric].append(model_history.history[metric])
                    # write the results to the log file
                    with open(log_file, 'a') as f:
                        f.write(f'FOLD: {fold_count}; Dataset: {ds}; Splice Site: {ss}; Model: {m}; Data: {str(results_table[ds][ss][m])}')
                        f.write("\n\n\n")
                        f.close()
                    # if ensembling, get each models predictions on the datasets by fold
                    if kwargs['ensemble']:
                        trn_X = X_trn.astype("float")
                        val_X = X_val.astype("float")
                        trn_preds = trained_model.predict(
                            trn_X,
                            batch_size=32,
                        )
                        val_preds = trained_model.predict(
                            val_X,
                            batch_size=32,
                        )
                        # predictions for train + val of all models on a SINGLE fold
                        model_preds_by_fold.append((m, trn_preds, val_preds))
                # generate a new results table if ensembling is used
                if kwargs['ensemble']:
                    results_table = ensemble(ds, ss, fold_count, model_preds_by_fold, y_trn, y_val, results_table, models, log_file, kwargs)
                fold_count += 1

    # # plot each fold results by ds and ss
    #plotting.all_subs(results_table, graph_folder, kwargs, METRICS)
    #plotting.each_subs(results_table, graph_folder, kwargs, METRICS)
    #plotting.each_subs_avg(results_table, graph_folder, kwargs, METRICS)
    #plotting.each_subs_avg_box(results_table, graph_folder, kwargs, METRICS)

    print(results_table)
    return results_table



def train(encoded_datasets, results_table, models, kwargs, log_file, graph_folder):
    # loop through datasets and splice sites
    trained_models = '-'.join(kwargs["sub_models"])
    for ds in kwargs['datasets']:
        for ss in kwargs['splice_sites']:
            # get the appropriate training data and labels from the encoded datasets
            X_train, y_train = encoded_datasets[ds][ss]
            row_length = len(X_train[0])
            model_preds_by_fold = []
            # loop through the submodels, training each
            for i, m in enumerate(kwargs['sub_models']):
                print(f"\n{'-'*5+m.upper()+'-'*5}\n")
                model_history, trained_model = models[m].build(
                    row_length,
                    X_train, y_train, [], [], # dummy
                    kwargs,
                )

                trained_model.save(os.getcwd()+f'/Models/TrainedModels/{trained_models}_{m}_{ds}_{ss}')
                # add a list of length equal to the number of epochs to the metric
                train_metrics = list(filter(lambda elt: "val" not in elt, list(METRICS.keys())))
                if m != "sfinder" and m != "dsplice":
                    for metric in train_metrics:
                        results_table[ds][ss][m][metric].append(model_history.history[metric])
                if m == "sfinder" or m == "dsplice":
                    results_table[ds][ss][m]["bin_acc"].append(model_history.history["bin_acc"])

                # write the results to the log file
                with open(log_file, 'a') as f:
                    f.write(f'Dataset: {ds}; Splice Site: {ss}; Model: {m}; Data: {str(results_table[ds][ss][m])}')
                    f.write("\n\n\n")
                    f.close()
                # if ensembling, get each models predictions on the datasets by fold
                if kwargs['ensemble']:
                    trn_X = X_train.astype("float")
                    trn_preds = trained_model.predict(
                        trn_X,
                        batch_size=32,
                    )
                    model_preds_by_fold.append((m, trn_preds))
            # generate a new results table if ensembling is used
            if kwargs['ensemble']:
                fold_count, y_val = 5, [] # dummy
                results_table = ensemble(ds, ss, fold_count, model_preds_by_fold, y_train, y_val, results_table, models, log_file, kwargs)

def test(encoded_datasets, results_table, models, kwargs, log_file):
    trained_models = '-'.join(kwargs["sub_models"])
    for ds in kwargs['datasets']:
        for ss in kwargs['splice_sites']:
            X_test, y_test = encoded_datasets[ds][ss]
            row_length = len(X_test[0])
            model_preds_by_fold = []
            # loop through the submodels, training each
            for i, m in enumerate(kwargs['sub_models']):
                print(f"\n{'-'*5+m.upper()+'-'*5}\n")
                loaded_model = tf.keras.models.load_model(
                    os.getcwd()+f'/Models/TrainedModels/{trained_models}_{m}_{ds}_{ss}/'
                )
                if m == 'sfinder':
                    history = loaded_model.evaluate(X_test, y_test)
                    print(history)

                if m == 'dsplice':
                    history = loaded_model.evaluate(X_test, y_test)
                    print(history)
                    
                if kwargs['ensemble']:
                    tst_X = X_test.astype("float")
                    tst_preds = loaded_model.predict(
                        tst_X,
                        batch_size=32,
                    )
                    model_preds_by_fold.append((m, tst_preds))
                # models_score = loaded_model.evaluate(X_test, y_test, verbose=0)
                # print(models_score)
            if kwargs['ensemble']:
                fold_count, y_val = 5, [] # dummy
                results_table = ensemble(ds, ss, fold_count, model_preds_by_fold, y_test, y_val, results_table, models, log_file, kwargs)



def run(**kwargs):

    # create a table to store all results: Datasets, Splice Sites, Models, Metics
    if kwargs["ensemble"]:
        model_list = copy.deepcopy(kwargs['sub_models'])
        model_list.append('ensemble')
    else:
        model_list = copy.deepcopy(kwargs['sub_models'])
    results_table = dict([(ds, dict([(ss, dict([(m, copy.deepcopy(METRICS)) for m in model_list])) for ss in kwargs['splice_sites']])) for ds in kwargs['datasets']])

    # encode the data properly (one hot encoding); the data shape is different for 1D vs. 2D conv layers
    encoded_datasets = dict([(ds, dict([(ss, ("", "")) for ss in kwargs['splice_sites']])) for ds in kwargs['datasets']])
    for ds in kwargs['datasets']:
        for ss in kwargs['splice_sites']:
            encoded_datasets[ds][ss] = encode_data.encode(ds, ss, kwargs)

            # add ensemble metrics as well
            results_table[ds][ss]['correlation'] = [] # correlation between ensemble network sub-models
            results_table[ds][ss]['q-statistic'] = [] # q-statistic
            results_table[ds][ss]['double-fault'] = [] # double fault

    # create a logs file if one doesn't already exist
    now = datetime.now()
    time = now.strftime("%m-%d-%Y_%H") #Note: %M for minutes
    op = kwargs["operation"]
    ds = ''.join(kwargs["datasets"])
    ms = ''.join(kwargs["sub_models"])
    ss = ''.join(kwargs["splice_sites"])
    if kwargs['ensemble']:
        es = "w_ens"
    else:
        es = "wo_ens"

    # main results file
    file = os.getcwd()+'/Logs/'+f'{op}_{time}_{ds}_{ss}_{ms}_{es}.txt'
    if not os.path.exists(file):
        try:
            with open(file, 'w') as f:
                f.write("ENSEMBLE SPLICE RESULTS REPORT\n")
                f.close()
        except FileNotFoundError:
            print(os.getcwd())

    # create folder with same name as log file for the images
    log_file_name = f'{op}_{time}_{ds}_{ss}_{ms}_{es}'
    graph_folder = os.getcwd()+'/Graphs/'+f'{op}_{time}_{ds}_{ss}_{ms}_{es}/'
    if not os.path.exists(graph_folder):
        try:
            os.mkdir(graph_folder)
        except FileNotFoundError:
            print(os.getcwd())

    # attached models to sub_model names
    models = {
        'cnn1': sub_models.CNN01(),
        'cnn2': sub_models.CNN02(),
        'cnn3': sub_models.CNN03(),
        'cnn4': sub_models.CNN04(),
        'dnn1': sub_models.DNN01(),
        'dnn2': sub_models.DNN02(),
        'dnn3': sub_models.DNN03(),
        'dnn4': sub_models.DNN04(),
        'ensemble': sub_models.ENSEMBLE(),
        'sfinder': sub_models.SpliceFinder(),
        'dsplice': sub_models.DeepSplicer(),
        # 'ens': Ensemble(),
    }

    # run validation
    if kwargs["operation"] == "validate":
        print("\nStarting validation process...\n")
        results_table = validate(
            encoded_datasets,
            results_table,
            models,
            kwargs,
            file,
            graph_folder
        )

        with open(file, 'a') as f:
            f.write(str(results_table))
            f.close()

    # run training
    if kwargs["operation"] == "train":
        print("\nStarting training process...\n")
        results_table = train(
            encoded_datasets,
            results_table,
            models,
            kwargs,
            file,
            graph_folder
        )

        with open(file, 'a') as f:
            f.write(str(results_table))
            f.close()
    if kwargs["operation"] == "test":
        print("\nStarting testing process...\n")
        results_table = test(
            encoded_datasets,
            results_table,
            models,
            kwargs,
            file,
        )

        with open(file, 'a') as f:
            f.write(str(results_table))
            f.close()

