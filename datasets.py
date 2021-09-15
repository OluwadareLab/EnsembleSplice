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
  # combine data

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
  # imbalance the dataset
  # CONSTRUCTION
  # if kwargs['imbalance'] != 0:
  #   if (len(atn_x) >= len(atp_x)):
  #     seed(kwargs['random_state'])
  #     atn_x = sample(atn_x, len(atp_x))
  #   if (len(atn_x) <= len(atp_x)):
  #     seed(kwargs['random_state'])
  #     atp_x = sample(atp_x, len(atn_x))
  #   if (len(dtn_x) >= len(dtp_x)):
  #     seed(kwargs['random_state'])
  #     dtn_x = sample(dtn_x, len(dtp_x))
  #   if (len(dtn_x) <= len(dtp_x)):
  #     seed(kwargs['random_state'])
  #     dtp_x = sample(dtp_x, len(dtn_x))

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
  # imbalance the dataset
  # CONSTRUCTION
  # if kwargs['imbalance'] != 0:
  #   if (len(atn_x) >= len(atp_x)):
  #     seed(kwargs['random_state'])
  #     atn_x = sample(atn_x, len(atp_x))
  #   if (len(atn_x) <= len(atp_x)):
  #     seed(kwargs['random_state'])
  #     atp_x = sample(atp_x, len(atn_x))
  #   if (len(dtn_x) >= len(dtp_x)):
  #     seed(kwargs['random_state'])
  #     dtn_x = sample(dtn_x, len(dtp_x))
  #   if (len(dtn_x) <= len(dtp_x)):
  #     seed(kwargs['random_state'])
  #     dtp_x = sample(dtp_x, len(dtn_x))

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
  # imbalance the dataset
  # CONSTRUCTION
  # if kwargs['imbalance'] != 0:
  #   if (len(atn_x) >= len(atp_x)):
  #     seed(kwargs['random_state'])
  #     atn_x = sample(atn_x, len(atp_x))
  #   if (len(atn_x) <= len(atp_x)):
  #     seed(kwargs['random_state'])
  #     atp_x = sample(atp_x, len(atn_x))
  #   if (len(dtn_x) >= len(dtp_x)):
  #     seed(kwargs['random_state'])
  #     dtn_x = sample(dtn_x, len(dtp_x))
  #   if (len(dtn_x) <= len(dtp_x)):
  #     seed(kwargs['random_state'])
  #     dtp_x = sample(dtp_x, len(dtn_x))

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
  # imbalance the dataset
  # CONSTRUCTION
  # if kwargs['imbalance'] != 0:
  #   if (len(atn_x) >= len(atp_x)):
  #     seed(kwargs['random_state'])
  #     atn_x = sample(atn_x, len(atp_x))
  #   if (len(atn_x) <= len(atp_x)):
  #     seed(kwargs['random_state'])
  #     atp_x = sample(atp_x, len(atn_x))
  #   if (len(dtn_x) >= len(dtp_x)):
  #     seed(kwargs['random_state'])
  #     dtn_x = sample(dtn_x, len(dtp_x))
  #   if (len(dtn_x) <= len(dtp_x)):
  #     seed(kwargs['random_state'])
  #     dtp_x = sample(dtp_x, len(dtn_x))

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
  if kwargs['combine']:
    pass

  # # CONSTRUCTION
  # if dataset == 'ce':
  #   # Acceptor Data
  #   acc_x = read_seq_data('./Datasets/CE/Acceptor_All.txt', dataset)
  #   atn_x = [elt.split()[0] for elt in acc_x if elt.split()[1]=='-1']
  #   atp_x = [elt.split()[0] for elt in acc_x if elt.split()[1]=='1']
  #   if balance:
  #     if len(atn_x) >= len(atp_x):
  #       atn_x = sample(atn_x, len(atp_x))
  #     else:
  #       atp_x = sample(atp_x, len(atn_x))
  #   atn_y = [0]*len(atn_x)
  #   atp_y = [1]*len(atp_x)
  #   acc_x, acc_y = atn_x+atp_x, atn_y+atp_y
  #   acc = {'Acceptor Sites':acc_x, 'Acceptor Labels':acc_y}
  #   acc_df = pd.DataFrame(data=acc)
  #   # Donor Data
  #   don_x = read_seq_data('./Datasets/CE/Donor_All.txt', dataset)
  #
  # if dataset == 'hs3d':
  #   # Note renaming scheme, Intron-Exon --> AG: Acceptor, Exon-Intron --> GT: Donor
  #   # Acceptor Data
  #   extract_seq = lambda seq: [elt.split(':')[1] for elt in seq]
  #   atp_x = read_seq_data('./Datasets/HS3D/Acceptor_Positive.txt', dataset, 1)
  #   atp_x = extract_seq(atp_x)
  #   atp_y = [1]*len(atp_x)
  #   atn_x_1 = read_seq_data('./Datasets/HS3D/Acceptor_Negative01.txt', dataset, 1) # has extract spaces
  #   atn_x_2 = read_seq_data('./Datasets/HS3D/Acceptor_Negative02.txt', dataset, 0)
  #   atn_x_3 = read_seq_data('./Datasets/HS3D/Acceptor_Negative03.txt', dataset, 0)
  #   atn_x_4 = read_seq_data('./Datasets/HS3D/Acceptor_Negative04.txt', dataset, 0)
  #   btwn_1_2 = atn_x_1[-1]+atn_x_2[0] # look at last line of (1) and first line of (2)
  #   btwn_2_3 = atn_x_2[-1]+atn_x_3[0]
  #   btwn_3_4 = atn_x_3[-1]+atn_x_4[0]
  #   atn_x = atn_x_1[:-1]+[btwn_1_2]+atn_x_2[1:-1]+[btwn_2_3]+atn_x_3[1:-1]+[btwn_3_4]+atn_x_4[1:]
  #   if balance:
  #     # len(atp_x) < len(atn_x)
  #     atn_x = extract_seq(atn_x)
  #     atn_x = sample(atn_x, len(atp_x))
  #   else:
  #     atn_x = extract_seq(atn_x)
  #   atn_y = [0]*len(atn_x)
  #   acc_x, acc_y  = atn_x+atp_x, atn_y+atp_y
  #   acc = {'Acceptor Sites':acc_x, 'Acceptor Labels':acc_y}
  #   acc_df = pd.DataFrame(data=acc)
  #   # Donor Data
  #   dtp_x = read_seq_data('./Datasets/HS3D/Donor_Positive.txt', dataset, 1)
  #   dtp_x = extract_seq(dtp_x)
  #   dtp_y = [1]*len(dtp_x)
  #   dtn_x_1 = read_seq_data('./Datasets/HS3D/Donor_Negative01.txt', dataset, 1)
  #   dtn_x_2 = read_seq_data('./Datasets/HS3D/Donor_Negative02.txt', dataset, 0)
  #   dtn_x_3 = read_seq_data('./Datasets/HS3D/Donor_Negative03.txt', dataset, 0)
  #   btwn_1_2 = dtn_x_1[-1]+dtn_x_2[0]
  #   btwn_2_3 = dtn_x_2[-1]+dtn_x_3[0]
  #   dtn_x = dtn_x_1[:-1]+[btwn_1_2]+dtn_x_2[1:-1]+[btwn_2_3]+dtn_x_3[1:]
  #   if balance:
  #     dtn_x = extract_seq(dtn_x)
  #     dtn_x = sample(dtn_x, len(dtp_x)) # balanceanced
  #   else:
  #     dtn_x = extract_seq(dtn_x) # unbalanceanced
  #   dtn_y = [0]*len(dtn_x)
  #   don_x, don_y  = dtn_x+dtp_x, dtn_y+dtp_y
  #   don = {'Donor Sites':don_x, 'Donor Labels':don_y}
  #   don_df = pd.DataFrame(data=don)
  #   return (acc_df, don_df)
  #
  # if dataset == 'ce2':
  #   # Acceptor Data
  #   atp_x = read_seq_data('./Datasets/Caenorhabditis/positive_DNA_seqs_acceptor_c_elegans.fa', dataset)
  #   atn_x =read_seq_data('./Datasets/Caenorhabditis/negative_DNA_seqs_acceptor_c_elegans.fa', dataset)
  #   atp_y = [1]*len(atp_x)
  #   atn_y = [0]*len(atn_x)
  #   acc_x, acc_y  = atn_x+atp_x, atn_y+atp_y
  #   acc = {'Acceptor Sites':acc_x, 'Acceptor Labels':acc_y}
  #   acc_df = pd.DataFrame(data=acc)
  #   # Donor data
  #   dtp_x = read_seq_data('./Datasets/Caenorhabditis/positive_DNA_seqs_donor_c_elegans.fa', dataset)
  #   dtn_x = read_seq_data('./Datasets/Caenorhabditis/negative_DNA_seqs_donor_c_elegans.fa', dataset)
  #   dtn_y = [0]*len(dtn_x)
  #   dtp_y = [1]*len(dtp_x)
  #   don_x, don_y  = dtn_x+dtp_x, dtn_y+dtp_y
  #   don = {'Donor Sites':don_x, 'Donor Labels':don_y}
  #   don_df = pd.DataFrame(data=don)
  #   return (acc_df, don_df)
  #
  # if dataset == 'hs2':
  #   # Acceptor Data
  #   atp_x = read_seq_data('./Datasets/Homo/Train/pos_acceptor_hs_train.fa', dataset)
  #   atn_x = read_seq_data('./Datasets/Homo/Train/neg_acceptor_hs_train.fa', dataset)
  #   atp_y = [1]*len(atp_x)
  #   atn_y = [0]*len(atn_x)
  #   acc_x, acc_y  = atn_x+atp_x, atn_y+atp_y
  #   acc = {'Acceptor Sites':acc_x, 'Acceptor Labels':acc_y}
  #   acc_df = pd.DataFrame(data=acc)
  #   # Donor data
  #   dtp_x = read_seq_data('./Datasets/Homo/Train/pos_donor_hs_train.fa', dataset)
  #   dtn_x = read_seq_data('./Datasets/Homo/Train/neg_donor_hs_train.fa', dataset)
  #   dtn_y = [0]*len(dtn_x)
  #   dtp_y = [1]*len(dtp_x)
  #   don_x, don_y  = dtn_x+dtp_x, dtn_y+dtp_y
  #   don = {'Donor Sites':don_x, 'Donor Labels':don_y}
  #   don_df = pd.DataFrame(data=don)
  #   return (acc_df, don_df)
  #
  # if dataset == 'oy':
  #   # Acceptor Data
  #   atp_x = read_seq_data('./Datasets/Oryza/Train/pos_acceptor_oriza_train.fa', dataset)
  #   atn_x =read_seq_data('./Datasets/Oryza/Train/neg_acceptor_oriza_train.fa', dataset)
  #   atp_y = [1]*len(atp_x)
  #   atn_y = [0]*len(atn_x)
  #   acc_x, acc_y  = atn_x+atp_x, atn_y+atp_y
  #   acc = {'Acceptor Sites':acc_x, 'Acceptor Labels':acc_y}
  #   acc_df = pd.DataFrame(data=acc)
  #   # Donor data
  #   dtp_x = read_seq_data('./Datasets/Oryza/Train/pos_donor_oriza_train.fa', dataset)
  #   dtn_x = read_seq_data('./Datasets/Oryza/Train/neg_donor_oriza_train.fa', dataset)
  #   dtn_y = [0]*len(dtn_x)
  #   dtp_y = [1]*len(dtp_x)
  #   don_x, don_y  = dtn_x+dtp_x, dtn_y+dtp_y
  #   don = {'Donor Sites':don_x, 'Donor Labels':don_y}
  #   don_df = pd.DataFrame(data=don)
  #   return (acc_df, don_df)
  #
  # if dataset == 'ar':
  #   # Acceptor Data
  #   atp_x = read_seq_data('./Datasets/Arabidopsis/positive_DNA_seqs_acceptor_at.fa', dataset)
  #   atn_x =read_seq_data('./Datasets/Arabidopsis/negative_DNA_seqs_acceptor_at.fa', dataset)
  #   atp_y = [1]*len(atp_x)
  #   atn_y = [0]*len(atn_x)
  #   acc_x, acc_y  = atn_x+atp_x, atn_y+atp_y
  #   acc = {'Acceptor Sites':acc_x, 'Acceptor Labels':acc_y}
  #   acc_df = pd.DataFrame(data=acc)
  #   # Donor data
  #   dtp_x = read_seq_data('./Datasets/Arabidopsis/positive_DNA_seqs_donor_at.fa', dataset)
  #   dtn_x = read_seq_data('./Datasets/Arabidopsis/negative_DNA_seqs_donor_at.fa', dataset)
  #   dtn_y = [0]*len(dtn_x)
  #   dtp_y = [1]*len(dtp_x)
  #   don_x, don_y  = dtn_x+dtp_x, dtn_y+dtp_y
  #   don = {'Donor Sites':don_x, 'Donor Labels':don_y}
  #   don_df = pd.DataFrame(data=don)
  #   return (acc_df, don_df)
  #
  # if dataset == 'dm':
  #   # Acceptor Data
  #   atp_x = read_seq_data('./Datasets/Drosophila/positive_DNA_seqs_acceptor_d_mel.fa', dataset)
  #   atn_x =read_seq_data('./Datasets/Drosophila/negative_DNA_seqs_acceptor_d_mel.fa', dataset)
  #   atp_y = [1]*len(atp_x)
  #   atn_y = [0]*len(atn_x)
  #   acc_x, acc_y  = atn_x+atp_x, atn_y+atp_y
  #   acc = {'Acceptor Sites':acc_x, 'Acceptor Labels':acc_y}
  #   acc_df = pd.DataFrame(data=acc)
  #   # Donor data
  #   dtp_x = read_seq_data('./Datasets/Drosophila/positive_DNA_seqs_donor_d_mel.fa', dataset)
  #   dtn_x = read_seq_data('./Datasets/Drosophila/negative_DNA_seqs_donor_d_mel.fa', dataset)
  #   dtn_y = [0]*len(dtn_x)
  #   dtp_y = [1]*len(dtp_x)
  #   don_x, don_y  = dtn_x+dtp_x, dtn_y+dtp_y
  #   don = {'Donor Sites':don_x, 'Donor Labels':don_y}
  #   don_df = pd.DataFrame(data=don)
  #   return (acc_df, don_df)
