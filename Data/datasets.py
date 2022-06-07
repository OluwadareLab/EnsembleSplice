# -----------------------------------------------------------------------------
# Copyright (c) 2021 Trevor P. Martin. All rights reserved.
# Distributed under the MIT License.
# -----------------------------------------------------------------------------
import numpy as np
import pandas as pd
from random import sample, seed

# this is the default random state across all files
RANDOM_STATE = 123432


# there are idiosyncracies (mostly tags) to each dataset that must be dealt with
def read_seq_data(file):
  with open(file, 'r') as f:
    all_lines = f.readlines()
    seq_data = [elt.replace('\n','').replace(' ','') for elt in list(filter(lambda line: False if '>' in line else True, all_lines))]
    seq_data = list(filter(None, seq_data))
    f.close()
  return seq_data

# read each dataset (call its specific function)
def read_dataset(dataset, kwargs):

  # CHANGE FROM PLAYDATA to Datasets
  # RENAME ALL Subtrain to Train, same for test
  # RENAME Acceptor_Train_Negative to Acceptor_Train_Negative
  # if training, use the training files, otherwise use testing files
  if (kwargs["operation"] == 'train') or (kwargs["operation"] == 'validate') or (kwargs["operation"] == 'tune'):
    atn_x = read_seq_data(f'./ENSDatasets/{dataset}/Train/Acceptor_Train_Negative.txt')
    atp_x = read_seq_data(f'./ENSDatasets/{dataset}/Train/Acceptor_Train_Positive.txt')
    dtn_x = read_seq_data(f'./ENSDatasets/{dataset}/Train/Donor_Train_Negative.txt')
    dtp_x = read_seq_data(f'./ENSDatasets/{dataset}/Train/Donor_Train_Positive.txt')
  if (kwargs["operation"] == 'test'):
    atn_x = read_seq_data(f'./ENSDatasets/{dataset}/Test/Acceptor_Test_Negative.txt')
    atp_x = read_seq_data(f'./ENSDatasets/{dataset}/Test/Acceptor_Test_Positive.txt')
    dtn_x = read_seq_data(f'./ENSDatasets/{dataset}/Test/Donor_Test_Negative.txt')
    dtp_x = read_seq_data(f'./ENSDatasets/{dataset}/Test/Donor_Test_Positive.txt')

  # acceptor and donor labels, positive is 1 negative is 0
  atp_y = [1]*len(atp_x)
  atn_y = [0]*len(atn_x)
  dtp_y = [1]*len(dtp_x)
  dtn_y = [0]*len(dtn_x)

  # acceptor and donor datasets (concatenate pos, neg values and pos, neg labels)
  acc_x, acc_y  = atn_x+atp_x, atn_y+atp_y
  don_x, don_y  = dtn_x+dtp_x, dtn_y+dtp_y
  acc = {'Acceptor Sites':acc_x, 'Acceptor Labels':acc_y}
  don = {'Donor Sites':don_x, 'Donor Labels':don_y}
  acc_df = pd.DataFrame(data=acc)
  don_df = pd.DataFrame(data=don)

  # shuffle datasets
  acc_df, don_df = acc_df.sample(frac=1, random_state=RANDOM_STATE), don_df.sample(frac=1, random_state=RANDOM_STATE)

  # return the acceptor and donor dataframes
  return (acc_df, don_df)


# recall, these are the kwargs

# ensemblesplice.run(
  #     ensemble=args.esplice,
  #     sub_models=sub_models,
  #     datasets=datasets,
  #     splice_sites=splice_sites,
  #     operation=operation[0][1],
  #     folds=args.k,
  #     report=args.report,
  # )
