# -----------------------------------------------------------------------------
# Copyright (c) 2021 Trevor P. Martin. All rights reserved.
# Distributed under the MIT License.
# -----------------------------------------------------------------------------
from Models import sub_models
from Models import ensemblesplice
from Models import tune_models
import argparse


def main():

    # initialize argument parser
    parser = argparse.ArgumentParser(description='EnsembleSplice commands')
    # dataset options
    d_opts = parser.add_argument_group('Dataset Options')
    d_opts.add_argument("--nn269", action='store_const', const='nn269', default='', help='Use this to choose the NN269 (DeepSS) dataset.')
    d_opts.add_argument("--ce", action='store_const',const='ce', default='', help='Use this to choose the CE (DeepSS) dataset.')
    d_opts.add_argument("--hs3d", action='store_const',const='hs3d', default='', help='Use this to choose the HS3D (DeepSS) dataset.')
    d_opts.add_argument("--hs2", action='store_const',const='hs2', default='', help='Use this to choose the Homo Sapiens (Splice2Deep) dataset.')
    d_opts.add_argument("--ce2", action='store_const', const='ce2', default='', help='Use this to choose the Caenorhabditis elegans (Splice2Deep) dataset.')
    d_opts.add_argument("--oy", action='store_const', const='oy', default='', help='Use this to choose the Oryza Sativa japonica (Splice2Deep) dataset.')
    d_opts.add_argument("--ar", action='store_const', const='ar', default='', help='Use this to choose the Arabidopsis thaliana (Splice2Deep) dataset.')
    d_opts.add_argument("--dm", action='store_const', const='dm', default='', help='Use this to choose the Drosophila melanogaster (Splice2Deep) dataset.')
    d_opts.add_argument("--all_ds", action='store_const',  const='all_ds', default=False, help='Run on every available dataset individually (not combined)')
    d_opts.add_argument('--combine', action='store_true', help='Whether to make the datasets into one large dataset.')
    d_opts.add_argument('--random_state', type=int, default=123432, help='Whether to make the datasets into one large dataset.')
    d_opts.add_argument('--bal', action='store_true', default=False, help='Balance the datasets; this takes a random sample of the largest class equal in size to the smallest class.')
    d_opts.add_argument('--imbal', nargs='?', type=float, const=0.2, default=False, help='Make the TRUE splice site data make up x% of the total data.')
    # model options
    m_opts = parser.add_argument_group('Model Options')
    m_opts.add_argument('--cnn1', nargs='?', type=int, const=1, default=0, help='Use this to choose the CNN1 architecture.')
    m_opts.add_argument('--cnn2',  nargs='?',type=int, const=1, default=0, help='Use this to choose the CNN2 architecture.')
    m_opts.add_argument('--cnn3',  nargs='?', type=int, const=1, default=0, help='Use this to choose the CNN3 architecture.')
    m_opts.add_argument('--cnn4',  nargs='?', type=int, const=1, default=0, help='Use this to choose the CNN4 architecture.')
    m_opts.add_argument('--cnn5',  nargs='?', type=int, const=1, default=0, help='Use this to choose the CNN5 architecture.')
    m_opts.add_argument('--rnn1',  nargs='?', type=int, const=1, default=0, help='Use this to choose the RNN1 architecture.')
    m_opts.add_argument('--rnn2',  nargs='?', type=int, const=1, default=0, help='Use this to choose the RNN2 architecture.')
    m_opts.add_argument('--rnn3',  nargs='?', type=int, const=1, default=0, help='Use this to choose the RNN3 architecture.')
    m_opts.add_argument('--dnn1',  nargs='?', type=int, const=1, default=0, help='Use this to choose the DNN1 architecture.')
    m_opts.add_argument('--dnn2',  nargs='?', type=int, const=1, default=0, help='Use this to choose the DNN2 architecture.')
    m_opts.add_argument('--dnn3', nargs='?', type=int, const=1, default=0, help='Use this to choose the DNN3 architecture.')
    m_opts.add_argument('--all_m', action='store_const', const='all_m', default=False, help='Use this to choose all model architectures (not including EnsembleSplice).')
    m_opts.add_argument('--esplice_p', action='store_const', const='esplice_p', default='', help='Use this to choose the sklearn perceptron classifier.')
    m_opts.add_argument('--esplice_l', action='store_const', const='esplice_l', default='', help='Use this to choose the sklearn logistic regression classifier.')
    m_opts.add_argument('--esplice_s', action='store_const', const='esplice_s', default='', help='Use this to choose the sklearn linear support vector classifier.')
    m_opts.add_argument('--esplice_d', action='store_const', const='esplice_d', default='', help='Use this to choose the keras logistic regression classifier.')
    m_opts.add_argument('--all_esplice', action='store_const', const='all_esplice', default=False, help='Use this to choose the EnsembleSplice architecture.')
    m_opts.add_argument('--anova', action='store_true', default=False, help='Use this perform anova between predictions from all model architecture in the ensemble.')
    m_opts.add_argument('--report', action='store_true', default=False, help='Generate a report of the results for the given paradigm.')
    m_opts.add_argument('--val', action='store_true', default=False, help='Perform validation training.')
    m_opts.add_argument('--test', action='store_true', default=False, help='Perform testing.')
    m_opts.add_argument('--train', action='store_true', default=False, help='Perform training and saving.')
    m_opts.add_argument('--store_config', action='store_true', default=False, help='Use this option to print keras config')
    m_opts.add_argument("--tune", action='store_true', default=False, help='Overrides everything else; tune the models')
    # splice site options
    s_opts = parser.add_argument_group('Splice Site Options')
    s_opts.add_argument("--acceptor", action='store_const', const='acceptor', default='', help='Use this option to only select acceptor splice sites')
    s_opts.add_argument("--donor", action='store_const', const='donor', default='', help='Use this option to only select donor splice sites ')
    s_opts.add_argument("--both_ss", action='store_const', const='both_ss', default=False, help='Use this option to only select donor splice sites ')
    # cross validation
    c_opts = parser.add_argument_group('Cross Validation Options')
    c_opts.add_argument("--k", type=int, required=False, default=5, help='The value of k to use for cross validation')
    # command line arguments
    args = parser.parse_args()

    # dataset asserts
    if type(args.imbal) != bool:
        assert 0.1<=(args.imbal)<1.0,'The value of --imbal must be a fraction and greater than or equal to 0.1.'
    datasets = list(filter(lambda dataset: dataset != '', [args.nn269, args.ce, args.hs3d, args.hs2, args.ce2, args.oy, args.ar, args.dm]))
    if args.combine:
        datasets = ['combine']
    if type(args.all_ds)==str:
        assert len(datasets)==0,'You are using all datasets, dont specify other datasets.'
        datasets = ['nn269', 'ce', 'hs3d', 'hs2', 'ce2', 'oy', 'ar', 'dm']
    assert len(datasets)>=1,'You have not chosen a dataset to use.'
    network_rows = {
        'acceptor':{
            'nn269':90, 'ce':141,
            'hs3d':140, 'hs2':602,
            'ce2':602, 'dm':602,
            'ar':602, 'or':602,'combine':602,},
        'donor':{
            'nn269':15, 'ce':141,
            'hs3d':140, 'hs2':602,
            'ce2':602, 'dm':602,
            'ar':602, 'or':602,'combine':602,}
    }
    # model asserts
    assert (args.train+args.test+args.val)==1,'You should choose between doing model validation, model training and saving, and testing.'
    sub_models = [
        ('cnn1', args.cnn1),('cnn2', args.cnn2),('cnn3',args.cnn3),('cnn4', args.cnn4),('cnn5', args.cnn5),
        ('rnn1', args.rnn1),('rnn2', args.rnn2),('rnn3',args.rnn3),
        ('dnn1', args.dnn1),('dnn2', args.dnn2),('dnn3',args.dnn3),
    ]
    if type(args.all_m)==str:
        assert sum([model[1] for model in sub_models])==0,'You are using all models, dont specify other models.'
        sub_models = [(model[0], 1) for model in sub_models]
    assert sum([model[1] for model in sub_models])>0,'You have not chosen any models to use.'
    sub_models = list(filter(lambda m: m[1] != 0, sub_models))
    esplice_archs = list(filter(lambda classif: classif != '', [args.esplice_d, args.esplice_l, args.esplice_p, args.esplice_s]))
    if type(args.all_esplice)==str:
        assert len(esplice_archs)==0,'You dont need to specify other esplice archs if youve chosen all'
        esplice_archs = ['esplice_l','esplice_p','esplice_s'] # add esplice_d later
    if args.tune:
        for model in sub_models:
            assert (model[1]==0 or model[1]==1),'For tuning, the model count must be either 0 or 1'
        assert len(datasets)==1,'For tuning, you must choose only one dataset to tune on'
        assert args.train,'For tuning, you must use the training data'
        assert len(esplice_archs)==0,'EnsembleSplice cannot currently be tuned.'
    # splice site asserts
    splice_sites = list(filter(lambda splice_site: splice_site != '', [args.acceptor, args.donor]))
    assert (len(splice_sites)==0) or (len(splice_sites)==1),'If you want both of the splice sites, just use --both_ss'
    if type(args.both_ss)==str:
        assert len(splice_sites)==0,'Use --both_ss for both splice sites'
        splice_sites = ['acceptor', 'donor']
    assert len(splice_sites)>0,'You have not chosen a splice site option to use.'
    # cross validation assert
    assert 2<=(args.k)<=20,'Dont make the k in the Kfold cross validation too high or too low please!'

    # CONSTRUCTION
    # tune models?
    if args.tune:
        print('\nTuning Submodels...\n')
        tune_models.run(
            models=sub_models,
            datasets=datasets,
            splice_sites=splice_sites,
            rows=network_rows,
            state=args.random_state,
            train=args.train,
            validate=args.val,
            folds=args.k,
            test=args.test,
            combine=args.combine,
            balance=args.bal,
            imbalance=args.imbal,
        )

    if not args.tune:
        # run esplice, with appropriate submodels
        if len(esplice_archs) != 0:
            print('\nRunning EnsembleSplice...\n')
            ensemblesplice.run(
                models=sub_models,
                esplicers=esplice_archs,
                datasets=datasets,
                splice_sites=splice_sites,
                rows=network_rows,
                state=args.random_state,
                train=args.train,
                validate=args.val,
                folds=args.k,
                test=args.test,
                combine=args.combine,
                balance=args.bal,
                imbalance=args.imbal,
                report=args.report,
                anova=args.anova,
                store_config=args.store_config,
            )

        # run the sub_models alone
        if len(esplice_archs) == 0:
            print('\nRunning Sub Models...\n')
            ensemblesplice.run(
                models=sub_models,
                datasets=datasets,
                splice_sites=splice_sites,
                rows=network_rows,
                state=args.random_state,
                train=args.train,
                validate=args.val,
                folds=args.k,
                test=args.test,
                combine=args.combine,
                balance=args.bal,
                imbalance=args.imbal,
                report=args.report,
                anova=args.anova,
                store_config=args.store_config,
            )

if __name__ == '__main__':
    main()
