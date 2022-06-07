# -----------------------------------------------------------------------------
# Copyright (c) 2021 Trevor P. Martin. All rights reserved.
# Distributed under the MIT License.
# -----------------------------------------------------------------------------
from Models import ensemblesplice
from Models import tune_sub_models
import argparse
import sys


def main():

    # initialize argument parser
    parser = argparse.ArgumentParser(description='EnsembleSplice commands')

    # dataset options
    d_opts = parser.add_argument_group('Dataset Options')
    # NN269, CE, HS3D, Homo Sapiens, Caenorhabditis elegans, Oryza Sativa japonica, Arabidopsis thaliana, Drosophila melanogaster
    for d_set in ["nn269", "ce", "hs3d", "hs2", "ce2", "oy", "ar", "dm"]:
        d_opts.add_argument(f"--{d_set}", action='store_const', const=f'{d_set}', default='', help=f'Use this to choose the {d_set.upper()} dataset.')
    d_opts.add_argument("--all_ds", action='store_const',  const='all_ds', default=False, help='Run on every available dataset individually (not combined)')

    # model options
    m_opts = parser.add_argument_group('Model Options')
    for name, number in {"cnn":4, "dnn":4}.items():
        for _ in range(1, number+1):
            m_opts.add_argument(f'--{name}{str(_)}', nargs='?', type=int, const=1, default=0, help=f'Use this to choose the {name.upper()}{str(_)} architecture.')
    m_opts.add_argument('--all_models', action='store_const', const='all_m', default=False, help='Use this to choose all model architectures.')
    m_opts.add_argument('--report', action='store_true', default=False, help='Generate a report of the results for the given paradigm.')
    m_opts.add_argument('--validate', action='store_true', default=False, help='Perform validation training.')
    m_opts.add_argument('--test', action='store_true', default=False, help='Perform testing.')
    m_opts.add_argument('--train', action='store_true', default=False, help='Perform training and saving.')
    m_opts.add_argument("--esplice", action='store_true', default=False, help='Use EnsembleSplice on the sub-models')
    m_opts.add_argument("--tune", action='store_true', default=False, help='Tune sub-models')

    # splice site options
    s_opts = parser.add_argument_group('Splice Site Options')
    s_opts.add_argument("--acceptor", action='store_const', const='acceptor', default='', help='Use this option to only select acceptor splice sites')
    s_opts.add_argument("--donor", action='store_const', const='donor', default='', help='Use this option to only select donor splice sites')
    s_opts.add_argument("--both_ss", action='store_const', const='both_ss', default=False, help='Use this option to only select donor splice sites')

    # cross validation
    c_opts = parser.add_argument_group('Cross Validation Options')
    c_opts.add_argument("--k", type=int, required=False, default=5, help='The value of k (number of folds) to use for cross validation')

    # collect all command line arguments
    args = parser.parse_args()

    #-------

    # for tuning, make sure nothing else is specified
    if args.tune:
        tune_sub_models.run()
        sys.exit(0)

    # collect all datasets given by the user
    datasets = list(filter(lambda dataset: dataset != '', [args.nn269, args.ce, args.hs3d, args.hs2, args.ce2, args.oy, args.ar, args.dm]))

    # check if the all-datasets option is being used, and command user not to write all_ds and other datasets
    if type(args.all_ds)==str:
        assert len(datasets)==0,'You are using all datasets, don\'t specify other datasets.'
        datasets = ['nn269', 'ce', 'hs3d', 'hs2', 'ce2', 'oy', 'ar', 'dm']

    # make sure at least 1 dataset is being used
    assert len(datasets)>=1,'You have not chosen a dataset to use.'

    # make sure only 1 operation (training, validation, testing) is being done
    assert (args.train+args.test+args.validate)==1,'You should choose between doing model validation, model training and saving, and testing.'
    operation = list(filter(lambda op: op[0] != 0, [(args.validate, 'validate'), (args.train, 'train'), (args.test, 'test')]))

    # a list of available sub-networks to use for ensembling
    sub_models = [
        ('cnn1', args.cnn1),('cnn2', args.cnn2),('cnn3',args.cnn3),('cnn4',args.cnn4),
        ('dnn1', args.dnn1),('dnn2', args.dnn2),('dnn3',args.dnn3),('dnn4',args.dnn4)
    ]

    # make sure if user specifies all models
    if type(args.all_models)==str:
        assert sum([model[1] for model in sub_models])==0,'You are using all models, dont specify other models.'
        sub_models = [(model[0], 1) for model in sub_models]

    # make sure, if all_models hasn't been selected, that at least 1 sub-model is selected
    assert sum([model[1] for model in sub_models])>0,'You have not chosen any models to use.'

    # the sub_models used is some command-line selection of available sub_models
    sub_models = [elt[0] for elt in list(filter(lambda m: m[1] != 0, sub_models))]

    # make sure both_ss is used instead of --acceptor and --donor
    splice_sites = list(filter(lambda splice_site: splice_site != "", [args.acceptor, args.donor]))
    assert (len(splice_sites)==0) or (len(splice_sites)==1),'If you want both of the splice sites, just use --both_ss'

    # if both_ss is specified, make sure --acceptor or --donor is not also used
    if type(args.both_ss)==str:
        assert len(splice_sites)==0,'Use --both_ss for both splice sites'
        splice_sites = ['acceptor', 'donor']

    # make sure at least one splice site is selected
    assert len(splice_sites)>0,'You have not chosen a splice site option to use.'

    # make sure the k for cross validation is a reasonable value
    assert 2<=(args.k)<=15,'Dont make the k in the Kfold cross validation too high or too low please!'

    print("\nYou\'ve decided to do the following, based on your command-line selection...\n")
    if args.esplice:
        print(f"{operation[0][1].upper()} with EnsembleSplice, using {sub_models} on {splice_sites} data from {datasets}\n")
    else:
        print(f"{operation[0][1].upper()} {sub_models} on {splice_sites} data from {datasets}\n")

    #-------

    # run ensemble splice
    ensemblesplice.run(
        ensemble=args.esplice,
        sub_models=sub_models,
        datasets=datasets,
        splice_sites=splice_sites,
        operation=operation[0][1],
        folds=args.k,
        report=args.report,
    )

if __name__ == '__main__':
    main()

    # # network_rows = {
    # #     'acceptor':{
    # #         'nn269':90, 'ce':141,
    # #         'hs3d':140, 'hs2':602,
    # #         'ce2':602, 'dm':602,
    # #         'ar':602, 'or':602,'combine':602,},
    # #     'donor':{
    # #         'nn269':15, 'ce':141,
    # #         'hs3d':140, 'hs2':602,
    # #         'ce2':602, 'dm':602,
    # #         'ar':602, 'or':602,'combine':602,}
    # # }




# Notes
# Some examples and their explanations for this program might be as follows:
# python3 exec.py --train --cnn4 --acceptor --hs3d
# Train CNN4 sub-network on Acceptor splice site data from HS3D
# python3 exec.py --train --cnn4 --cnn5 --donor --ns269 --esplice
# Train CNN4 and CNN5 sub-networks on Donor NN269 data and perform Ensembling
