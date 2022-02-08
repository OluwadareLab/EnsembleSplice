# -----------------------------------------------------------------------------
# Copyright (c) 2021 Trevor P. Martin. All rights reserved.
# Distributed under the MIT License.
# -----------------------------------------------------------------------------
import numpy as np
import pandas as pd
from random import sample, seed


def read_seq_data(
  file,
  dataset,
  number=0
  ):
  """
  Parameters
  ----------
  file: a string defining the path to a particular data file
  dataset: a string {nn269, ce, hs3d} indicating which dataset to use
  number: integer, delimit special cases, such as files with headers

  Returns
  -------
  seq_data: a list of nucleotide sequence strings
  """
  # NN269
  if dataset == 'nn269':
    with open(file, 'r') as f:
      all_lines = f.readlines()
      seq_data = [elt.replace('\n','').replace(' ','') for elt in list(filter(lambda line: False if '>' in line else True, all_lines))]
      seq_data = list(filter(None, seq_data))
      f.close()
    return seq_data
  # HS3D subset
  if dataset == 'hs3d':
    with open(file, 'r') as f:
      all_lines = f.readlines()
      seq_data = [elt.replace('\n','').replace(' ','') for elt in list(filter(lambda line: False if '>' in line else True, all_lines))]
      seq_data = list(filter(None, seq_data))
      f.close()
    return seq_data
  # Splice2Deep data
  if dataset in ['hs2','ce2','dm','oy','ar']:
    with open(file, 'r') as f:
      all_lines = f.readlines()
      seq_data = [elt.replace('\n','').replace(' ','') for elt in list(filter(lambda line: False if '>' in line else True, all_lines))]
      seq_data = list(filter(None, seq_data))
      f.close()
    return seq_data

# ./Datasets/ for local machine
def read_nn269(kwargs):
  dataset = 'nn269'
  # get training data
  if kwargs['train'] or kwargs['validate']:
    atn_x = read_seq_data('./Datasets/NN269/Train/Acceptor_Train_Negative.txt', dataset)
    atp_x = read_seq_data('./Datasets/NN269/Train/Acceptor_Train_Positive.txt', dataset)
    dtn_x = read_seq_data('./Datasets/NN269/Train/Donor_Train_Negative.txt', dataset)
    dtp_x = read_seq_data('./Datasets/NN269/Train/Donor_Train_Positive.txt', dataset)
  # get testing data
  if kwargs['test']:
    atn_x = read_seq_data('./Datasets/NN269/Test/Acceptor_Test_Negative.txt', dataset)
    atp_x = read_seq_data('./Datasets/NN269/Test/Acceptor_Test_Positive.txt', dataset)
    dtn_x = read_seq_data('./Datasets/NN269/Test/Donor_Test_Negative.txt', dataset)
    dtp_x = read_seq_data('./Datasets/NN269/Test/Donor_Test_Positive.txt', dataset)
  # balance the dataset
  if kwargs['balance']:
    if (len(atn_x) >= len(atp_x)):
      seed(kwargs['state'])
      atn_x = sample(atn_x, len(atp_x))
    if (len(atn_x) <= len(atp_x)):
      seed(kwargs['state'])
      atp_x = sample(atp_x, len(atn_x))
    if (len(dtn_x) >= len(dtp_x)):
      seed(kwargs['state'])
      dtn_x = sample(dtn_x, len(dtp_x))
    if (len(dtn_x) <= len(dtp_x)):
      seed(kwargs['state'])
      dtp_x = sample(dtp_x, len(dtn_x))
  # acceptor and donor labels
  atp_y = [1]*len(atp_x)
  atn_y = [0]*len(atn_x)
  dtp_y = [1]*len(dtp_x)
  dtn_y = [0]*len(dtn_x)
  # acceptor and donor dataset
  acc_x, acc_y  = atn_x+atp_x, atn_y+atp_y
  don_x, don_y  = dtn_x+dtp_x, dtn_y+dtp_y
  acc = {'Acceptor Sites':acc_x, 'Acceptor Labels':acc_y}
  don = {'Donor Sites':don_x, 'Donor Labels':don_y}
  acc_df = pd.DataFrame(data=acc)
  don_df = pd.DataFrame(data=don)
  # shuffle datasets
  acc_df, don_df = acc_df.sample(frac=1, random_state=kwargs['state']), don_df.sample(frac=1, random_state=kwargs['state'])
  return (acc_df, don_df)

# currently for subset of hs3d
def read_hs3d(kwargs):
  dataset = 'hs3d'
  # get training data
  if kwargs['train'] or kwargs['validate']:
    atn_x = read_seq_data('./Datasets/HS3D/SubsetTrain/Acceptor_Train_Negative.txt', dataset)
    atp_x = read_seq_data('./Datasets/HS3D/SubsetTrain/Acceptor_Train_Positive.txt', dataset)
    dtn_x = read_seq_data('./Datasets/HS3D/SubsetTrain/Donor_Train_Negative.txt', dataset)
    dtp_x = read_seq_data('./Datasets/HS3D/SubsetTrain/Donor_Train_Positive.txt', dataset)
  # get testing data
  if kwargs['test']:
    atn_x = read_seq_data('./Datasets/HS3D/SubsetTest/Acceptor_Test_Negative.txt', dataset)
    atp_x = read_seq_data('./Datasets/HS3D/SubsetTest/Acceptor_Test_Positive.txt', dataset)
    dtn_x = read_seq_data('./Datasets/HS3D/SubsetTest/Donor_Test_Negative.txt', dataset)
    dtp_x = read_seq_data('./Datasets/HS3D/SubsetTest/Donor_Test_Positive.txt', dataset)
  # balance the dataset
  if kwargs['balance']:
    if (len(atn_x) >= len(atp_x)):
      seed(kwargs['state'])
      atn_x = sample(atn_x, len(atp_x))
    if (len(atn_x) <= len(atp_x)):
      seed(kwargs['state'])
      atp_x = sample(atp_x, len(atn_x))
    if (len(dtn_x) >= len(dtp_x)):
      seed(kwargs['state'])
      dtn_x = sample(dtn_x, len(dtp_x))
    if (len(dtn_x) <= len(dtp_x)):
      seed(kwargs['state'])
      dtp_x = sample(dtp_x, len(dtn_x))

  # acceptor and donor labels
  atp_y = [1]*len(atp_x)
  atn_y = [0]*len(atn_x)
  dtp_y = [1]*len(dtp_x)
  dtn_y = [0]*len(dtn_x)
  # acceptor and donor dataset
  acc_x, acc_y  = atn_x+atp_x, atn_y+atp_y
  don_x, don_y  = dtn_x+dtp_x, dtn_y+dtp_y
  acc = {'Acceptor Sites':acc_x, 'Acceptor Labels':acc_y}
  don = {'Donor Sites':don_x, 'Donor Labels':don_y}
  acc_df = pd.DataFrame(data=acc)
  don_df = pd.DataFrame(data=don)
  # shuffle datasets
  acc_df, don_df = acc_df.sample(frac=1, random_state=kwargs['state']), don_df.sample(frac=1, random_state=kwargs['state'])
  return (acc_df, don_df)


def read_ar(kwargs):
  dataset = 'ar' # multiple will have the same effect
  # get training data
  if kwargs['train'] or kwargs['validate']:
    atn_x = read_seq_data('./Datasets/Arabidopsis/SubsetTrain/Acceptor_Train_Negative.txt', dataset)
    atp_x = read_seq_data('./Datasets/Arabidopsis/SubsetTrain/Acceptor_Train_Positive.txt', dataset)
    dtn_x = read_seq_data('./Datasets/Arabidopsis/SubsetTrain/Donor_Train_Negative.txt', dataset)
    dtp_x = read_seq_data('./Datasets/Arabidopsis/SubsetTrain/Donor_Train_Positive.txt', dataset)
  # get testing data
  if kwargs['test']:
    atn_x = read_seq_data('./Datasets/Arabidopsis/SubsetTest/Acceptor_Test_Negative.txt', dataset)
    atp_x = read_seq_data('./Datasets/Arabidopsis/SubsetTest/Acceptor_Test_Positive.txt', dataset)
    dtn_x = read_seq_data('./Datasets/Arabidopsis/SubsetTest/Donor_Test_Negative.txt', dataset)
    dtp_x = read_seq_data('./Datasets/Arabidopsis/SubsetTest/Donor_Test_Positive.txt', dataset)
  # balance the dataset
  if kwargs['balance']:
    if (len(atn_x) >= len(atp_x)):
      seed(kwargs['state'])
      atn_x = sample(atn_x, len(atp_x))
    if (len(atn_x) <= len(atp_x)):
      seed(kwargs['state'])
      atp_x = sample(atp_x, len(atn_x))
    if (len(dtn_x) >= len(dtp_x)):
      seed(kwargs['state'])
      dtn_x = sample(dtn_x, len(dtp_x))
    if (len(dtn_x) <= len(dtp_x)):
      seed(kwargs['state'])
      dtp_x = sample(dtp_x, len(dtn_x))

  # acceptor and donor labels
  atp_y = [1]*len(atp_x)
  atn_y = [0]*len(atn_x)
  dtp_y = [1]*len(dtp_x)
  dtn_y = [0]*len(dtn_x)
  # acceptor and donor dataset
  acc_x, acc_y  = atn_x+atp_x, atn_y+atp_y
  don_x, don_y  = dtn_x+dtp_x, dtn_y+dtp_y
  acc = {'Acceptor Sites':acc_x, 'Acceptor Labels':acc_y}
  don = {'Donor Sites':don_x, 'Donor Labels':don_y}
  acc_df = pd.DataFrame(data=acc)
  don_df = pd.DataFrame(data=don)
  # shuffle datasets
  acc_df, don_df = acc_df.sample(frac=1, random_state=kwargs['state']), don_df.sample(frac=1, random_state=kwargs['state'])
  return (acc_df, don_df)

def read_hs2(kwargs):
  dataset = 'hs2' # multiple will have the same effect
  # get training data
  if kwargs['train'] or kwargs['validate']:
    atn_x = read_seq_data('./Datasets/Homo_sapiens/SubsetTrain/Acceptor_Train_Negative.txt', dataset)
    atp_x = read_seq_data('./Datasets/Homo_sapiens/SubsetTrain/Acceptor_Train_Positive.txt', dataset)
    dtn_x = read_seq_data('./Datasets/Homo_sapiens/SubsetTrain/Donor_Train_Negative.txt', dataset)
    dtp_x = read_seq_data('./Datasets/Homo_sapiens/SubsetTrain/Donor_Train_Positive.txt', dataset)
  # get testing data
  if kwargs['test']:
    atn_x = read_seq_data('./Datasets/Homo_sapiens/SubsetTest/Acceptor_Test_Negative.txt', dataset)
    atp_x = read_seq_data('./Datasets/Homo_sapiens/SubsetTest/Acceptor_Test_Positive.txt', dataset)
    dtn_x = read_seq_data('./Datasets/Homo_sapiens/SubsetTest/Donor_Test_Negative.txt', dataset)
    dtp_x = read_seq_data('./Datasets/Homo_sapiens/SubsetTest/Donor_Test_Positive.txt', dataset)
  # balance the dataset
  if kwargs['balance']:
    if (len(atn_x) >= len(atp_x)):
      seed(kwargs['state'])
      atn_x = sample(atn_x, len(atp_x))
    if (len(atn_x) <= len(atp_x)):
      seed(kwargs['state'])
      atp_x = sample(atp_x, len(atn_x))
    if (len(dtn_x) >= len(dtp_x)):
      seed(kwargs['state'])
      dtn_x = sample(dtn_x, len(dtp_x))
    if (len(dtn_x) <= len(dtp_x)):
      seed(kwargs['state'])
      dtp_x = sample(dtp_x, len(dtn_x))

  # acceptor and donor labels
  atp_y = [1]*len(atp_x)
  atn_y = [0]*len(atn_x)
  dtp_y = [1]*len(dtp_x)
  dtn_y = [0]*len(dtn_x)
  # acceptor and donor dataset
  acc_x, acc_y  = atn_x+atp_x, atn_y+atp_y
  don_x, don_y  = dtn_x+dtp_x, dtn_y+dtp_y
  acc = {'Acceptor Sites':acc_x, 'Acceptor Labels':acc_y}
  don = {'Donor Sites':don_x, 'Donor Labels':don_y}
  acc_df = pd.DataFrame(data=acc)
  don_df = pd.DataFrame(data=don)
  # shuffle datasets
  acc_df, don_df = acc_df.sample(frac=1, random_state=kwargs['state']), don_df.sample(frac=1, random_state=kwargs['state'])
  return (acc_df, don_df)


def read_dataset(
  dataset,
  kwargs,
  ):
  """
  Parameters
  ----------
  dataset: a string {nn269, ce, hs3d, } indicating which dataset to use
  kwargs: dictionary, arguments found in the exec file

  Returns
  -------
  (acc_df, don_df) : a tuple of pandas dataframes, one per splice site type
  """

  # READ NN269
  if dataset == 'nn269':
    return read_nn269(kwargs)

  # READ NN269
  if dataset == 'hs3d':
    return read_hs3d(kwargs)

  # READ HS2
  if dataset == 'hs2':
    return read_hs2(kwargs)

  # READ AR
  if dataset == 'ar':
    return read_ar(kwargs)

  # COMBINE DATASET
  if dataset == 'combine':
    ar = read_ar(kwargs)
    hs2 = read_hs2(kwargs)
    acc_dfs = [ar[0], hs2[0]]
    don_dfs = [ar[1], hs2[1]]
    acc_comb = pd.concat(acc_dfs, sort=False, ignore_index=True)
    don_comb = pd.concat(don_dfs, sort=False, ignore_index=True)
    # shuffle
    acc_df, don_df = acc_comb.sample(frac=1, random_state=kwargs['state']), don_comb.sample(frac=1, random_state=kwargs['state'])
    return (acc_df, don_df)
