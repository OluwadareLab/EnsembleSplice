# -----------------------------------------------------------------------------
# Copyright (c) 2021 Trevor P. Martin. All rights reserved.
# Distributed under the MIT License.
# -----------------------------------------------------------------------------
import numpy as np
from Data import datasets
import tensorflow as tf


def encode(dataset, ss_type, kwargs):

    # acquire shuffled datasets
    acc_df, don_df = datasets.read_dataset(dataset, kwargs)

    # one-hot-encode data by nucleotide, produce matrix L x 4, scheme was used in SpliceRover
    one_hot = {
        'A':np.array([1,0,0,0]).reshape(1,-1),
        'C':np.array([0,1,0,0]).reshape(1,-1),
        'G':np.array([0,0,1,0]).reshape(1,-1),
        'T':np.array([0,0,0,1]).reshape(1,-1),
        '_':np.array([0,0,0,0]).reshape(1,-1), # unknown nucleotides
    }
    make_one_hot = lambda seq: np.vstack([one_hot[nucleo.upper()] if nucleo.upper() in 'ACGT' else one_hot['_'] for nucleo in list(seq)])
    acc_df['Acceptor Sites'] = acc_df['Acceptor Sites'].apply(make_one_hot)
    don_df['Donor Sites'] = don_df['Donor Sites'].apply(make_one_hot)

    # categorically encode the data labels
    acc_y = tf.keras.utils.to_categorical(
        y=np.reshape(acc_df['Acceptor Labels'].to_numpy(), newshape=(-1,1)),
        num_classes=2,
        dtype='float32'
    )
    don_y = tf.keras.utils.to_categorical(
        y=np.reshape(don_df['Donor Labels'].to_numpy(), newshape=(-1,1)),
        num_classes=2,
        dtype='float32'
    )

    # reshapes each entry from (rows, cols) to (1, rows, cols) for use with LTSM, DNN, CONV1D
    expand_1 = lambda seq_mat: seq_mat.reshape(1, seq_mat.shape[0], seq_mat.shape[1])
    acc_df['Acceptor Sites'] = acc_df['Acceptor Sites'].apply(expand_1)
    don_df['Donor Sites'] = don_df['Donor Sites'].apply(expand_1)
    # make dataset of shape (len(data), rows, cols) by vertically stacking data
    acc_x = np.vstack(acc_df['Acceptor Sites'].to_numpy())
    don_x = np.vstack(don_df['Donor Sites'].to_numpy())
    if ss_type == "acceptor":
        return (acc_x, acc_y)
    if ss_type == "donor":
        return (don_x, don_y)
