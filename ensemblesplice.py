# -----------------------------------------------------------------------------
# Copyright (c) 2021 Trevor P. Martin. All rights reserved.
# Distributed under the MIT License.
# -----------------------------------------------------------------------------
from Data import encode_data
from Models import build_models
from Visuals import make_report

import numpy as np
import pandas as pd
import os
import math
import tensorflow as tf
import copy
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.metrics import auc, plot_precision_recall_curve
from sklearn.metrics import plot_precision_recall_curve
from sklearn.metrics import accuracy_score
from sklearn.linear_model import Perceptron
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.feature_selection import f_classif


def get_metrics(tp, tn, fp, fn):
    # get tn, tp, sn, sp, rec, pre, acc, mcc, f1, auc
    metrics = {
        'TP':tp, 'TN':tn, 'FP':fp, 'FN':fn,
        'Pre':'', 'Mcc':'', 'F1':'',
        'Sn':'', 'Sp':'',
        'Acc':'', 'Err':'',
        'Auc':'',
        'Tpr':'',
        'Fpr':'',
    }
    metrics['Tpr'] = tp / (tp + fn)
    metrics['Fpr'] = fp / (fp + tn)
    metrics['F1'] = (2*tp) / ((2*tp) + fp + fn)
    metrics['Pre'] = (tp) / (tp + fp)
    metrics['Sn'] = (tp) / (tp + fn)
    metrics['Sp'] = (tn) / (tn + fp)
    metrics['Acc'] = (tp + tn) / (tp + tn + fp + fn)
    metrics['Err'] = 1 - metrics['Acc']
    metrics['Mcc'] = ((tp * tn) - (fp * fn)) / math.sqrt((tp + fp)*(tp + fn)*(tn + fp)*(tn + fn))
    return metrics

def build_ensemble(
        hmd,
        ensemble_type,
        batch_size,
        kwargs,
        ss_type
    ):
    # ======================================================================================================
    # function to convert [9.5078093e-01 4.9219094e-02] to 0
    extr = lambda vals: np.argmax(vals, axis=1)
    # ======================================================================================================
    # HMD {'dnn1_1' : [{history, model, X_val, y_val, X_trn, y_trn}, {}, {}, {}, ..., num_folds], ...}
    if kwargs['validate']:
        # aggregate predictions that each model made for each fold, by fold
        val_preds_by_fold = dict( [(f'Fold_{fold}_Aggregate', []) for fold in range(1, kwargs['folds']+1)])
        for sub_arch in hmd.keys():
            for index, model_by_fold in enumerate(hmd[sub_arch]):
                model_preds = model_by_fold['model'].predict(model_by_fold['X_val'].astype('float32'), batch_size=batch_size)
                model_preds_truth_val = extr(model_preds).reshape(-1,1)
                val_preds_by_fold[f'Fold_{index+1}_Aggregate'].append(model_preds_truth_val)

        # concatenated by fold, each (num_samples, num_models)
        for fold_preds in val_preds_by_fold.keys():
            val_preds_by_fold[fold_preds] = np.concatenate(val_preds_by_fold[fold_preds], axis=1)

        # check whether all the models have the same labels for each fold
        first_model = hmd[list(hmd.keys())[0]]
        labels_by_fold = dict([(f'Fold_{fold}_Aggregate', []) for fold in range(1, kwargs['folds']+1)])
        for fold_num_1st, first_model_by_fold in enumerate(first_model):
            for sub_arch in hmd.keys():
                assert np.array_equal(first_model_by_fold['y_val'], hmd[sub_arch][fold_num_1st]['y_val']),'The labels are not the same'
            labels_by_fold[f'Fold_{fold_num_1st+1}_Aggregate'] = extr(first_model_by_fold['y_val'])

        # score the classifiers
        val_scores_by_fold = []
        for index, fold_preds in enumerate(val_preds_by_fold.keys()):
            if ensemble_type == 'esplice_l':
                classifier = LogisticRegression(random_state=None).fit(val_preds_by_fold[fold_preds], labels_by_fold[fold_preds])
            if ensemble_type == 'esplice_p':
                classifier = Perceptron(shuffle=False, random_state=None).fit(val_preds_by_fold[fold_preds], labels_by_fold[fold_preds])
            if ensemble_type == 'esplice_s':
                classifier = LinearSVC(random_state=None).fit(val_preds_by_fold[fold_preds], labels_by_fold[fold_preds])
            clf_val_score = classifier.score(val_preds_by_fold[fold_preds], labels_by_fold[fold_preds])
            val_scores_by_fold.append(clf_val_score)
        return np.mean(np.asarray(val_scores_by_fold))
    # ======================================================================================================
    if kwargs['train']:
        # HMD (dnn1_1 : {history, model, X_val, y_val, X_trn, y_trn })
        # aggregate predictions that each model makes
        predictions =  []
        for sub_arch in hmd.keys():
            model_preds = hmd[sub_arch]['model'].predict(hmd[sub_arch]['X_trn'].astype('float32'), batch_size=batch_size)
            model_preds_truth_trn = extr(model_preds).reshape(-1,1)
            predictions.append(model_preds_truth_trn)
        predictions = np.concatenate(predictions, axis=1)

        # check whether all the models have the same labels for each fold
        first_model = hmd[list(hmd.keys())[0]]
        for sub_arch in hmd.keys():
            assert np.array_equal(first_model['y_trn'], hmd[sub_arch]['y_trn']),'The labels are not the same'
        labels = extr(first_model['y_trn'])
    # ======================================================================================================
    if kwargs['test']:
        # HMD (dnn1_1 : {model, X_test, y_test})
        predictions =  []
        for sub_arch in hmd.keys():
            model_preds = hmd[sub_arch]['model'].predict(hmd[sub_arch]['X_tst'].astype('float32'), batch_size=batch_size)
            model_preds_truth_tst = extr(model_preds).reshape(-1,1)
            predictions.append(model_preds_truth_tst)
        predictions = np.concatenate(predictions, axis=1)

        first_model = hmd[list(hmd.keys())[0]]
        for sub_arch in hmd.keys():
            assert np.array_equal(first_model['y_tst'], hmd[sub_arch]['y_tst']),'The labels are not the same'
        labels = extr(first_model['y_tst'])

        # print(labels)
        # for sub_arch in hmd.keys():
        #     print(hmd[sub_arch]['preds'])
        #     print(extr(hmd[sub_arch]['preds']))
        #     print(extr(hmd[sub_arch]['preds']).tolist())

        if ensemble_type == 'esplice_l':
            classifier = LogisticRegression(random_state=None).fit(predictions, labels)
            preds = classifier.predict(predictions)
        if ensemble_type == 'esplice_p':
            classifier = Perceptron(shuffle=False, random_state=None).fit(predictions, labels)
            preds = classifier.predict(predictions)
        if ensemble_type == 'esplice_s':
            classifier = LinearSVC(random_state=None).fit(predictions, labels)
            preds = classifier.predict(predictions)

        tp = 0
        tn = 0
        fn = 0
        fp = 0

        for index, pred in enumerate(preds):
            if (pred == 1) and (labels[index] == 1):
                tp += 1
            if (pred == 1) and (labels[index] == 0):
                fp += 1
            if (pred == 0) and (labels[index] == 1):
                fn += 1
            if (pred == 0) and (labels[index] == 0):
                tn += 1

        metrics = get_metrics(tp, tn, fp, fn)
        print(f'ESPLICE TYPE: {ensemble_type} for {ss_type} splice sites')
        for key, value in metrics.items():
            print(f'{key}:{value}')

        clf_val_score = classifier.score(predictions, labels)

        # AUC thresholds
        threshold_list = [0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,.7,.75,.8,.85,.9,.95,.99]
        for threshold in threshold_list:
            clf_pred_score = (classifier.predict_proba(predictions)[:,1] >= threshold)
            # Auc curve
            precision, recall, thresholds = precision_recall_curve(labels, clf_pred_score)
            auc_precision_recall = auc(recall, precision)
            print('AUC',auc_precision_recall,'Threshold',threshold)
            plt.plot(recall, precision, color='blue',linewidth=0.5)

        # clf_pred_score = classifier.predict_proba(predictions)[:,1]
        # # Auc curve
        # precision, recall, thresholds = precision_recall_curve(labels, clf_pred_score)
        # auc_precision_recall = auc(recall, precision)
        # plt.plot(recall, precision, color='blue',linewidth=0.5)


        plt.title('AUC-Thresholding for HS2 Acceptor Data')
        plt.savefig('foop.png')

        disp = plot_precision_recall_curve(classifier, predictions, labels)

        return clf_val_score
    # ======================================================================================================
    if ensemble_type == 'esplice_l':
        classifier = LogisticRegression(random_state=None).fit(predictions, labels)
        # preds = classifier.predict()
    if ensemble_type == 'esplice_p':
        classifier = Perceptron(shuffle=False, random_state=None).fit(predictions, labels)
    if ensemble_type == 'esplice_s':
        classifier = LinearSVC(random_state=None).fit(predictions, labels)
    clf_val_score = classifier.score(predictions, labels)
    return clf_val_score

def ensemble_validation(kwargs, encoded_data, results, models_used):
    average_best_vals = lambda val_list: np.mean(np.asarray([max(x) for x in val_list]))

    # ss_by_fold = {
    #     'true_acceptor':[],
    #     'false_acceptor':[],
    #
    # }

    for ss_type in kwargs['splice_sites']:
        histories_models_data = dict([(f'{model[0]}_{iter}', []) for model in kwargs['models'] for iter in range(1, model[1]+1)])
        for model in models_used:
            for iter in results[ss_type][model].keys():
                for dataset in kwargs['datasets']:
                    # this loop inspired by https://www.machinecurve.com/
                    #index.php/2020/02/18/how-to-use-k-fold-cross-validation-with-keras/
                    k_fold = KFold(n_splits=kwargs['folds'], shuffle=False, random_state=kwargs['state'])
                    fold_num = 1
                    folds = kwargs['folds']
                    X_train, y_train = encoded_data[dataset][model][ss_type]
                    for train, test in k_fold.split(X_train, y_train):
                        print()
                        print('#'*60)
                        print(f'\nTraining {model.upper()} instance {iter} for fold ({fold_num}/{folds}) on {ss_type.upper()} {dataset.upper()} data...\n')
                        print('#'*60)
                        print()
                        X_trn, y_trn = X_train[train], y_train[train]
                        X_val, y_val = X_train[test], y_train[test]
                        history, trained_model = build_models.build_single_model(
                            kwargs['rows'][ss_type][dataset],
                            ss_type, model,
                            X_trn, y_trn, X_val, y_val,
                            kwargs,
                        )
                        results[ss_type][model][iter][dataset]['val_acc'].append(history.history['val_accuracy'])
                        results[ss_type][model][iter][dataset]['val_loss'].append(history.history['val_loss'])
                        results[ss_type][model][iter][dataset]['trn_acc'].append(history.history['accuracy'])
                        results[ss_type][model][iter][dataset]['trn_loss'].append(history.history['loss'])
                        results[ss_type][model][iter][dataset]['auc'].append(history.history['val_auc'])
                        results[ss_type][model][iter][dataset]['pre'].append(history.history['val_pre'])
                        results[ss_type][model][iter][dataset]['rec'].append(history.history['val_rec'])
                        histories_models_data[f'{model}_{iter}'].append(
                            {
                                'history':history, 'model':trained_model,
                                'X_trn':X_trn, 'y_trn':y_trn, 'X_val':X_val, 'y_val':y_val,
                            }
                        )
                        fold_num += 1
                    # average out metric across epochs
                    results[ss_type][model][iter][dataset]['val_acc'] = average_best_vals(results[ss_type][model][iter][dataset]['val_acc'])
                    results[ss_type][model][iter][dataset]['val_loss'] = average_best_vals(results[ss_type][model][iter][dataset]['val_loss'])
                    results[ss_type][model][iter][dataset]['trn_acc'] = average_best_vals(results[ss_type][model][iter][dataset]['trn_acc'])
                    results[ss_type][model][iter][dataset]['trn_loss'] = average_best_vals(results[ss_type][model][iter][dataset]['trn_loss'])
                    results[ss_type][model][iter][dataset]['auc'] = average_best_vals(results[ss_type][model][iter][dataset]['auc'])
                    results[ss_type][model][iter][dataset]['pre'] = average_best_vals(results[ss_type][model][iter][dataset]['pre'])
                    results[ss_type][model][iter][dataset]['rec'] = average_best_vals(results[ss_type][model][iter][dataset]['rec'])
        # get results for each chosen ensemble_type
        for e_arch in kwargs['esplicers']:
            for dataset in kwargs['datasets']:
                results[ss_type][e_arch][1][dataset] = build_ensemble(
                    histories_models_data,
                    e_arch,
                    32,
                    kwargs
                )
    # if kwargs['report']:
    #     for ss_type in kwargs['splice_sites']:
    #         for dataset in kwargs['datasets']:
    #             utils.single_dataset_val_acc_plot(ss_type, dataset, results, kwargs, models_used)
    #     make_report.report_results(results, kwargs, models_used)
    return results


def ensemble_train(kwargs, encoded_data, results, models_used):
    for ss_type in kwargs['splice_sites']:
        histories_models_data = dict([(f'{model[0]}_{iter}', '') for model in kwargs['models'] for iter in range(1, model[1]+1)])
        for model in models_used:
            for iter in results[ss_type][model].keys():
                for dataset in kwargs['datasets']:
                    X_train, y_train = encoded_data[dataset][model][ss_type]
                    print()
                    print('#'*60)
                    print(f'\nTraining {model.upper()} instance {iter} on {ss_type.upper()} {dataset.upper()} data...\n')
                    print('#'*60)
                    print()
                    history, trained_model = build_models.build_single_model(
                        kwargs['rows'][ss_type][dataset],
                        ss_type, model,
                        X_train,y_train,'','',
                        kwargs,
                    )
                    trained_model.save(os.getcwd()+f'/Models/TrainedModels/TrainedOn{dataset.upper()}/{model.upper()}_Instance_{iter}_{dataset}_{ss_type}')
                    results[ss_type][model][iter][dataset]['trn_acc'] = history.history['accuracy']
                    results[ss_type][model][iter][dataset]['trn_loss'] = history.history['loss']
                    histories_models_data[f'{model}_{iter}'] = {
                        'history':history, 'model':trained_model,
                        'X_trn':X_train, 'y_trn':y_train, 'X_val':'', 'y_val':'',
                    }
                    results[ss_type][model][iter][dataset]['trn_acc'] = max(results[ss_type][model][iter][dataset]['trn_acc'])
                    results[ss_type][model][iter][dataset]['trn_loss'] = min(results[ss_type][model][iter][dataset]['trn_loss'])
        for e_arch in kwargs['esplicers']:
            for dataset in kwargs['datasets']:
                results[ss_type][e_arch][1][dataset] = build_ensemble(
                    histories_models_data,
                    e_arch,
                    32,
                    kwargs
                )
    return results

def ensemble_test(kwargs, encoded_data, results, models_used):
    best_vals = lambda val_list: max(val_list)
    for ss_type in kwargs['splice_sites']:
        histories_models_data = dict([(f'{model[0]}_{iter}', '') for model in kwargs['models'] for iter in range(1, model[1]+1)])
        for model in models_used:
            for iter in results[ss_type][model].keys():
                for dataset in kwargs['datasets']:
                    X_test, y_test = encoded_data[dataset][model][ss_type]
                    loaded_model = tf.keras.models.load_model(
                        os.getcwd()+f'/Models/TrainedModels/TrainedOn{dataset.upper()}/{model.upper()}_Instance_{iter}_{dataset}_{ss_type}'
                        #os.getcwd()+f'/Models/TrainedModels/TrainedOnAR/{model.upper()}_Instance_{iter}_ar_{ss_type}'
                    )
                    models_score = loaded_model.evaluate(X_test, y_test, verbose=0)
                    print(f'{model.upper()}_{iter}\'s ACC {models_score[1]}')
                    histories_models_data[f'{model}_{iter}'] = {
                        'model':loaded_model,
                        'X_tst':X_test, 'y_tst':y_test,
                        'preds':loaded_model.predict(X_test),
                    }
        for e_arch in kwargs['esplicers']:
            for dataset in kwargs['datasets']:
                results[ss_type][e_arch][1][dataset] = build_ensemble(
                    histories_models_data,
                    e_arch,
                    32,
                    kwargs,
                    ss_type,
                )
    return results

def run(**kwargs):
    # ======================================================================================================
    metrics = {
        'val_acc':[],'val_loss':[],'trn_acc':[],'trn_loss':[],
        'auc':[],'pre':[],'rec':[],'seATsp':[],'spATsn':[],
    }
    datasets = dict([(dataset, {}) for dataset in kwargs['datasets']])
    sub_models = [(model[0], dict([(num, {}) for num in range(1, model[1]+1)])) for model in kwargs['models']]
    esplicing = [(e_arch, {1:{}}) for e_arch in kwargs['esplicers']]
    models = dict(sub_models+esplicing)
    results = {}
    for ss_type in kwargs['splice_sites']:
        results[ss_type] = copy.deepcopy(models)
        for model in results[ss_type].keys():
            for iter in results[ss_type][model].keys():
                results[ss_type][model][iter] = copy.deepcopy(datasets)
                for dataset in results[ss_type][model][iter].keys():
                    results[ss_type][model][iter][dataset] = copy.deepcopy(metrics)
    # Example of a dictionary produced by running python3 exec.py --val --nn269 --k 2 --donor --dnn1 --dnn2 --all_esplice
    # {'donor': {'dnn1': {1: {'nn269': {'val_acc': [], 'val_loss': [], 'trn_acc': [], 'trn_loss': [], 'auc': [], 'pre': [],
    # 'rec': [], 'seATsp': [], 'spATsn': []}}}, 'dnn2': {1: {'nn269': {'val_acc': [], 'val_loss': [], 'trn_acc': [],
    # 'trn_loss': [], 'auc': [], 'pre': [], 'rec': [], 'seATsp': [], 'spATsn': []}}}, 'esplice_d': {1: {'nn269': {'val_acc':
    # [], 'val_loss': [], 'trn_acc': [], 'trn_loss': [], 'auc': [], 'pre': [], 'rec': [], 'seATsp': [], 'spATsn': []}}},
    # 'esplice_l': {1: {'nn269': {'val_acc': [], 'val_loss': [], 'trn_acc': [], 'trn_loss': [], 'auc': [], 'pre': [],
    # 'rec': [], 'seATsp': [], 'spATsn': []}}}, 'esplice_p': {1: {'nn269': {'val_acc': [], 'val_loss': [], 'trn_acc': [],
    # 'trn_loss': [], 'auc': [], 'pre': [], 'rec': [], 'seATsp': [], 'spATsn': []}}}, 'esplice_s': {1: {'nn269': {'val_acc':
    #  [], 'val_loss': [], 'trn_acc': [], 'trn_loss': [], 'auc': [], 'pre': [], 'rec': [], 'seATsp': [], 'spATsn': []}}}}}
    # ======================================================================================================
    models_used = dict([(model[0], '') for model in kwargs['models']])
    encoded_data = {}
    for dataset in kwargs['datasets']:
        encoded_data[dataset] = models_used
        conv2d_dict = encode_data.encode(dataset, 'cnn1', kwargs) # could also be cnn5
        other_dict = encode_data.encode(dataset, 'cnn2', kwargs) # could also be not cnn1/cnn5
        for model in encoded_data[dataset].keys():
            if (model == 'cnn1') or (model=='cnn5'):
                encoded_data[dataset][model] = copy.deepcopy(conv2d_dict)
            else:
                encoded_data[dataset][model] = copy.deepcopy(other_dict)
    # ======================================================================================================
    if kwargs['validate']:
        run_results = ensemble_validation(kwargs, encoded_data, results, models_used)
    if kwargs['train']:
        run_results = ensemble_train(kwargs, encoded_data, results, models_used)
    if kwargs['test']:
        run_results = ensemble_test(kwargs, encoded_data, results, models_used)
    # ======================================================================================================


    print(results)
