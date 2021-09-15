from Models import ensemblesplice
from Models import sub_models
from Models import build_models

import matplotlib.pyplot as plt # plotting
import matplotlib.font_manager as font_manager # plot fonts
import numpy as np # x,y axes values
from sklearn.model_selection import KFold # cross validation
import copy # splice site results copy
import tensorflow as tf


tmfont = {'fontname': "serif"}
font = font_manager.FontProperties(family='serif', size=12)



def single_dataset_val_acc_plot(ss_type, dataset, results, kwargs, models_used):

    fig, axs = plt.subplots(figsize=(10, 10))

    for model in models_used:
        for iter in results[ss_type][model].keys():
            axs.bar(f'{model}_{iter}', results[ss_type][model][iter][dataset]['val_acc'])
    for e_arch in kwargs['esplicers']:
        axs.bar(f'{e_arch}', results[ss_type][e_arch][1][dataset])

    # y-axis ticks, labels
    # x-axis ticks, labels
    # bar colors
    # standard error bars
    # title
    # ?legend
    # fonts
    # getting metrics for the other things

    plt.show()


    #
    #
    # # GOOD
    # sites = {'acceptor':0, 'donor':1}
    # index_set = dict([(ds, index) for index, ds in enumerate(datasets)])
    #
    # # for each splice site type
    # for ss_type, ss_vals in results.items():
    #     # for each dataset
    #     for dataset, d_vals in ss_vals.items():
    #         axs[sites[ss_type]].bar('el_per', results[ss_type][dataset]['esplice']['per_val_acc_avg'])
    #         axs[sites[ss_type]].bar('el_log', results[ss_type][dataset]['esplice']['log_val_acc_avg'])
    #         axs[sites[ss_type]].bar('el_svc', results[ss_type][dataset]['esplice']['svc_val_acc_avg'])
    #         # for each submodel
    #         for sub_model, s_vals in d_vals.items():
    #             if sub_model != 'esplice':
    #                 axs[sites[ss_type]].bar(sub_model, results[ss_type][dataset][sub_model]['val_acc'])
    #
    #
    # plt.savefig(f'./Graphs/f.png')
    # if bal:
    #     axs[0].set_title(f'{sub_models} Acc/Loss for Balanced {dataset} Acceptor Data, {num_folds}-Folds, ',  fontsize=18,**tmfont)
    #     axs[1].set_title(f'{sub_models} Acc/Loss for Balanced {dataset} Acceptor Data, {num_folds}-Folds, ',  fontsize=18,**tmfont)
    #     fig.legend(loc='best', fontsize=12)
    #     #fig.savefig(f'./Graphs/{arch}_bal_{dataset}_{splice_site}_{num_folds}-folds.png')
    # else:
    #     axs[0].set_title(f'{sub_models} Acc/Loss for Imbalanced {dataset} Acceptor Data, {num_folds}-Folds, ',  fontsize=18,**tmfont)
    #     axs[1].set_title(f'{sub_models} Acc/Loss for Imbalanced {dataset} Acceptor Data, {num_folds}-Folds, ',  fontsize=18,**tmfont)
    #     fig.legend(loc='best', fontsize=12)
    #     #plt.savefig(f'./Graphs/{arch}_imbal_{dataset}_{splice_site}_{num_folds}-folds.png')







def errorbar(x, y, yerr, color, label, linestyle):
    plt.errorbar(
        x=x,
        y=y,
        yerr=yerr,
        elinewidth=2.0,
        barsabove=False,
        capsize=4.0,
        ecolor=color,
        linestyle=linestyle,
        linewidth=4.0,
        marker='o',
        markersize=6.0,
        markerfacecolor='black',
        color=color,
        label=label,
    )


def avg_loss_acc_by_fold_esplice():
    pass

def loss_acc_by_fold_esplice(
    values,
    arch,
    dataset,
    splice_site,
    num_folds,
    num_epochs,
    balanced,
    ):
    # plt.plot(list(range(1, num_folds+1)), values[0], label='val acc')
    # plt.plot(list(range(1, num_folds+1)), values[1], label='trn acc')
    # names = 'cnn_mva,cnn_sva,cnn_mvl,cnn_svl,cnn_mta,cnn_sta,cnn_mtl,cnn_stl,dnn_mva,dnn_sva,dnn_mvl,dnn_svl,dnn_mtl,dnn_stl,dnn_mta,dnn_sta,rnn_mva,rnn_sva,rnn_mvl,rnn_svl,rnn_mta,rnn_sta,rnn_mtl,rnn_stl'
    # names = names.split(',')
    # values = values[2:]
    # # colors = ['blue','blue','red','red','green','green','orange','orange','brown','brown','black','black','yellow','yellow','cyan','cyan']
    # for i in range(len(values), 2):
    #     # print(values[i], values[i+1])
    #     errorbar(fig, list(range(1, num_folds+1)), values[i], values[i+1], names[i])
    #names = 'e_val_acc,e_trn_accs,rnn_va,rnn_vl,rnn_ta,rnn_tl,cnn_vl,cnn_va,cnn_tl,cnn_ta,dnn_vl,dnn_va,dnn_tl,dnn_ta'
    names = 'e_val_acc,e_trn_accs,rnn_va,rnn_ta,cnn_va,cnn_ta,dnn_va,dnn_ta'
    names = names.split(',')

    # for index, val in enumerate(values):
    #     plt.bar(list(range(1, num_folds+1)), [elt*100 for elt in val], label=names[index])
    # move = np.arange(start=-(num_folds-2)//100, stop=(num_folds-2)//100, step=0.2)
    # for index, val in enumerate(values):
    #     plt.bar([elt+move[index] for elt in list(range(1, num_folds+1))], [elt*100 for elt in val], width=0.2, align='center', label=names[index])


    plt.bar(np.asarray(list(range(1, num_folds+1)))-((0.3/4)*4), [elt*100 for elt in values[0]], width=0.3/4, align='center', label=names[0])
    plt.bar(np.asarray(list(range(1, num_folds+1)))-((0.3/4)*3), [elt*100 for elt in values[1]], width=0.3/4, align='center', label=names[1])
    plt.bar(np.asarray(list(range(1, num_folds+1)))-((0.3/4)*2), [elt*100 for elt in values[2]], width=0.3/4, align='center', label=names[2])
    plt.bar(np.asarray(list(range(1, num_folds+1)))-((0.3/4)*1), [elt*100 for elt in values[3]], width=0.3/4, align='center', label=names[3])
    plt.bar(np.asarray(list(range(1, num_folds+1))), [elt*100 for elt in values[4]], width=0.3/4, align='center', label=names[4])
    plt.bar(np.asarray(list(range(1, num_folds+1)))+((0.3/4)*1), [elt*100 for elt in values[5]], width=0.3/4, align='center', label=names[5])
    # plt.bar(np.asarray(list(range(1, num_folds+1)))+((0.3/4)*2), [elt*100 for elt in values[6]], width=0.3/4, align='center', label=names[6])
    # plt.bar(np.asarray(list(range(1, num_folds+1)))+((0.3/4)*3), [elt*100 for elt in values[7]], width=0.3/4, align='center', label=names[7])


    # # annotate the sides of the graph with the final epoch values
    # for var in values:
    #     plt.annotate(
    #         '%0.2f' % var[-1],
    #         xy=(1, var[-1]),
    #         xytext=(7, 0),
    #         xycoords=('axes fraction', 'data'),
    #         textcoords='offset points',
    #         **tmfont,
    #     )

    # calculate y-ticks by max values, x-ticks is epochs
    # yticks = np.arange(start=0.0, stop=1.0, step=2/20)
    yticks = np.arange(start=80.0, stop=100.0, step=0.5)

    plt.yticks(ticks=yticks, fontsize=12, **tmfont)
    plt.ylim(80.0, 100.0)

    #plt.xticks(list(range(0, num_folds+1)), fontsize=12, **tmfont)
    #plt.xlim(0, num_folds+2)

    # x,y axis labels and enable grid
    arch = arch.upper()
    dataset = dataset.upper()
    splice_site = splice_site[0].upper()+splice_site[1:]
    plt.xlabel('Folds', fontsize=20, **tmfont)
    plt.ylabel('Accuracy/Loss', fontsize=20, **tmfont)
    plt.grid(True)
    # title depends on balanced or imbalanced dataset state
    if balanced:
        plt.title(f'{arch} Acc/Loss for Balanced {dataset} {splice_site} Data, {num_folds}-Folds',  fontsize=18,**tmfont)
        plt.legend(loc='best', fontsize=12)
        plt.savefig(f'./Graphs/{arch}_bal_{dataset}_{splice_site}_{num_folds}-folds.png')
    else:
        plt.title(f'{arch} Acc/Loss for Imbalanced {dataset} {splice_site} Data, {num_folds}-Folds', fontsize=18,**tmfont)
        plt.legend(loc='best', fontsize=12)
        plt.savefig(f'./Graphs/{arch}_imbal_{dataset}_{splice_site}_{num_folds}-folds.png')





# used in sub_models.py, ensemblesplice.py
def cross_validation(
    num_folds,
    model_type,
    splice_site,
    dataset,
    data,# encoded data for dataset (ds)
    rows, # donor, acceptor rows for ds
    evals,
    summary,
    config,
    batch_size,
    epochs,
    save
    ):

    # pairs (acc_x, acc_y), (don_x, don_y) by site type
    train_val_pairs = dict([('acceptor',(data[0], data[1])), ('donor',(data[2], data[3]))])

    # results
    means_stds = {
        'val_acc':[],
        'val_loss':[],
        'trn_acc':[],
        'trn_loss':[],
    }

    site_results = {
        'acceptor': copy.deepcopy(means_stds),
        'donor': copy.deepcopy(means_stds)}

    # function for extracting mean_over_epoch
    mean_by_epoch =  lambda seq: np.apply_along_axis(lambda row: np.mean(row), 1, np.asarray(seq).T)
    std_by_epoch = lambda seq: np.apply_along_axis(lambda row: np.std(row), 1, np.asarray(seq).T)

    # model builds
    builds = {'cnn':build_cnn, 'dnn':build_dnn, 'rnn':build_rnn}
    build_model = builds[model_type]

    # run cross validation
    for site_type, site_data in train_val_pairs.items():

        X_train, y_train = train_val_pairs[site_type]

        # this loop inspired by https://www.machinecurve.com/
        #index.php/2020/02/18/how-to-use-k-fold-cross-validation-with-keras/

        k_fold = KFold(n_splits=num_folds, shuffle=False)
        fold_num = 1
        for train, test in k_fold.split(X_train, y_train):

            # partition data accordingly
            X_trn, y_trn = X_train[train], y_train[train]
            X_val, y_val = X_train[test], y_train[test]

            # build model and get history
            history, model = build_models.build_single_model(
                rows[site_type][dataset],
                dataset,
                model_type,
                site_type,
                summary,
                X_trn,
                y_trn,
                batch_size,
                epochs,
                X_val,
                y_val,
                fold_num,
                num_folds,
                save
            )

            # accumulate val, trn loss, acc
            site_results[site_type]['val_acc'].append(history.history['val_accuracy'])
            site_results[site_type]['val_loss'].append(history.history['val_loss'])
            site_results[site_type]['trn_acc'].append(history.history['accuracy'])
            site_results[site_type]['trn_loss'].append(history.history['loss'])
            # accumulate dataset per fold and models


            # iterate fold
            fold_num += 1

        # print new lines
        print()
        print('#'*135)
        print()

        # means over epochs
        for key, value in site_results[site_type].items():
            site_results[site_type][key] = (mean_by_epoch(site_results[site_type][key]), std_by_epoch(site_results[site_type][key]))

    return (site_results)


# def group_metrics():
#     # metrics for all four models
#     pass
#
# def group_auc():
#     # metrics for all four models
#     pass
#
# def indiv_auc():
#     # metrics for all four models
#     pass


def loss_acc_sub_models(
    results,
    datasets,
    sub_models,
    num_epochs,
    num_folds,
    bal
    ):

    # line graphs of all models on a dataset
    datasets = dict([(ds, '') for ds in datasets])
    figs = {'acceptor': copy.deepcopy(datasets), 'donor': copy.deepcopy(datasets)}

    # x axis values
    x = list(range(1, num_epochs+1))

    # graph colors
    colors = {'cnn':'blue', 'dnn':'orange','rnn':'red'}

    # make blank plot
    fig, ax = plt.subplots(figsize=(10, 10))
    plt.xlabel('Epochs', fontsize=20, **tmfont)
    plt.ylabel('Accuracy/Loss', fontsize=20, **tmfont)
    yticks = np.arange(start=85.0, stop=100.0, step=0.5)
    plt.yticks(ticks=yticks, fontsize=12, **tmfont)
    plt.ylim(85.0, 100.0)
    plt.xticks(x, fontsize=12, **tmfont)
    plt.xlim(0, num_epochs+1)
    plt.grid(True)

    # plot on a different figure each time
    for site_type, data_figs in figs.items():
        for index, dataset in enumerate(datasets):
            for sub_model, m_values in results.items():
                if results[sub_model][dataset] == '':
                    pass
                else:
                    errorbar(
                        ax,
                        x,
                        results[sub_model][dataset][site_type]['val_acc'][0],
                        results[sub_model][dataset][site_type]['val_acc'][1],
                        colors[sub_model],
                        f'{sub_model} Val Acc',
                        '-',
                    )
                    errorbar(
                        ax,
                        x,
                        results[sub_model][dataset][site_type]['val_loss'][0],
                        results[sub_model][dataset][site_type]['val_loss'][1],
                        colors[sub_model],
                        f'{sub_model} Val Loss',
                        '--',
                    )
                    errorbar(
                        ax,
                        x,
                        results[sub_model][dataset][site_type]['trn_acc'][0],
                        results[sub_model][dataset][site_type]['trn_acc'][1],
                        colors[sub_model],
                        f'{sub_model} Trn Acc',
                        '-.',
                    )
                    errorbar(
                        ax,
                        x,
                        results[sub_model][dataset][site_type]['trn_loss'][0],
                        results[sub_model][dataset][site_type]['trn_loss'][1],
                        colors[sub_model],
                        f'{sub_model} Trn Loss',
                        ':',
                    )

            if len(sub_models)==1:
                arch=sub_models[0]
            if len(sub_models)!=1:
                arch = '-'.join([arch.upper() for arch in sub_models])
            ds = dataset.upper()
            splice_site = site_type[0].upper()+site_type[1:]
            if bal:
                plt.title(f'{arch} Acc/Loss for Balanced {ds} {splice_site} Data, {num_folds}-Folds',  fontsize=16,**tmfont)
                plt.legend(loc='best', fontsize=12)
                plt.savefig(f'./Graphs/{arch}_bal_{ds}_{splice_site}_{num_folds}-folds.png')
            else:
                plt.title(f'{arch} Acc/Loss for Imbalanced {ds} {splice_site} Data, {num_folds}-Folds', fontsize=16,**tmfont)
                plt.legend(loc='best', fontsize=12)
                plt.savefig(f'./Graphs/{arch}_imbal_{ds}_{splice_site}_{num_folds}-folds.png')
            plt.clf()


            # make_fig(
            #     site_type,
            #     dataset,
            #     results,
            #     sub_models,
            #     num_epochs,
            #     num_folds,
            #     bal,
            # )



            # ax.errorbar(
            #     x=x,
            #     y=results['cnn'][dataset][site_type]['val_acc'][0],
            #     yerr=results['cnn'][dataset][site_type]['val_acc'][1],
            #     elinewidth=2.0,
            #     barsabove=False,
            #     capsize=4.0,
            #     ecolor=colors['cnn'],
            #     linestyle='-',
            #     linewidth=4.0,
            #     marker='o',
            #     markersize=6.0,
            #     markerfacecolor='black',
            #     color=colors['cnn'],
            #     label=f'Cnn Val Acc',
            # )





            # setup_acc_loss(
            #     ax,
            #     x,
            #     results,
            #     dataset,
            #     colors,
            #     sub_models,
            #     site_type,
            #     num_folds
            #     num_epochs,
            # )



                    # for name, tup_values in results[sub_model][dataset][site_type].items():
                    #     print(name)
                    #     print(tup_values)
                    #     plt.annotate(
                    #         '%0.2f' % tup_values[][-1],
                    #         xy=(1, tup_values[-1]),
                    #         xytext=(7, 0),
                    #         xycoords=('axes fraction', 'data'),
                    #         textcoords='offset points',
                    #         **tmfont,
                    #     )
                    #     plt.annotate(
                    #         '%0.2f' % tup_values[-1],
                    #         xy=(1, tup_values[-1]),
                    #         xytext=(7, 0),
                    #         xycoords=('axes fraction', 'data'),
                    #         textcoords='offset points',
                    #         **tmfont,
                    #     )

            # title depends on balanced or imbalanced dataset state


def loss_acc_single_model(
    values,
    arch,
    dataset,
    splice_site,
    num_folds,
    num_epochs,
    balanced,
    ):
    fig, ax = plt.subplots(figsize=(10, 10))
    # x, y, yerr, color, label, linestyle

    x = list(range(1, num_epochs+1))

    errorbar(x , values[2], values[3], 'blue', 'Trn Acc')
    errorbar(fig, list(range(1, num_epochs+1)), values[0], values[1], 'red', 'Val Acc')
    errorbar(fig, list(range(1, num_epochs+1)), values[6], values[7], 'green', 'Trn Loss')
    errorbar(fig, list(range(1, num_epochs+1)), values[4], values[5], 'brown', 'Val Loss')
    # annotate the sides of the graph with the final epoch values
    for var in values:
        plt.annotate(
            '%0.2f' % var[-1],
            xy=(1, var[-1]),
            xytext=(7, 0),
            xycoords=('axes fraction', 'data'),
            textcoords='offset points',
            **tmfont,
        )
    # calculate y-ticks by max values, x-ticks is epochs
    yticks = np.arange(start=0.0, stop=1.0, step=2/20)
    plt.yticks(ticks=yticks, fontsize=12, **tmfont)
    plt.ylim(0.0, 1.0)
    plt.xticks(list(range(1, num_epochs+1)), fontsize=12, **tmfont)
    plt.xlim(0, num_epochs+1)
    # x,y axis labels and enable grid
    arch = arch.upper()
    dataset = dataset.upper()
    splice_site = splice_site[0].upper()+splice_site[1:]
    plt.xlabel('Epochs', fontsize=20, **tmfont)
    plt.ylabel('Accuracy/Loss', fontsize=20, **tmfont)
    plt.grid(True)
    # title depends on balanced or imbalanced dataset state
    if balanced:
        plt.title(f'{arch} Acc/Loss for Balanced {dataset} {splice_site} Data, {num_folds}-Folds',  fontsize=18,**tmfont)
        plt.legend(loc='best', fontsize=12)
        plt.savefig(f'./Graphs/{arch}_bal_{dataset}_{splice_site}_{num_folds}-folds.png')
    else:
        plt.title(f'{arch} Acc/Loss for Imbalanced {dataset} {splice_site} Data, {num_folds}-Folds', fontsize=18,**tmfont)
        plt.legend(loc='best', fontsize=12)
        plt.savefig(f'./Graphs/{arch}_imbal_{dataset}_{splice_site}_{num_folds}-folds.png')



# accuracy bars, line charts, axis text fonts,
# kwargs option, errorbar  option

def plot_point(plot, x, y):
    pass

# make this in same order as other thing
