# -----------------------------------------------------------------------------
# Copyright (c) 2021 Trevor P. Martin. All rights reserved.
# Distributed under the MIT License.
# -----------------------------------------------------------------------------
import numpy as np
from Data import datasets
import tensorflow as tf


def encode(
    dataset,
    model,
    kwargs
    ):
    """
    Parameters
    ----------
    dataset: a string {nn269, ce, hs3d} indicating which dataset to use
    model_names: the names of different models used
    kwargs:

    Returns
    -------
    [acc_x, acc_y, don_x, don_y]: list, of pandas dataframe containing
        true and false acceptor and donor sites and their labels
    """
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

    # Data for CONV2D layers
    if ('cnn1' == model) or ('cnn5' == model):
        # create new dataset if cnn1 or cnn5 is used.
        conv2d_acc_df, conv2d_don_df = acc_df.copy(deep=True), don_df.copy(deep=True)
        # reshape each entry from (rows,cols) to (1, rows, cols, 1) for use with Conv2D
        expand_2 = lambda seq_mat: seq_mat.reshape(1, seq_mat.shape[0], seq_mat.shape[1], -1)
        conv2d_acc_df['Acceptor Sites'] = conv2d_acc_df['Acceptor Sites'].apply(expand_2)
        conv2d_don_df['Donor Sites'] = conv2d_don_df['Donor Sites'].apply(expand_2)
        # make dataset of shape (len(data), rows, cols, 1) by vertically stacking data
        conv2d_acc_x = np.vstack(conv2d_acc_df['Acceptor Sites'].to_numpy())
        conv2d_don_x = np.vstack(conv2d_don_df['Donor Sites'].to_numpy())
        return {'acceptor':(conv2d_acc_x, acc_y), 'donor':(conv2d_don_x, don_y)}
    else:
        # Data for CONV1D, LTSM, Dense
        # reshapes each entry from (rows, cols) to (1, rows, cols) for use with LTSM, DNN, CONV1D
        expand_1 = lambda seq_mat: seq_mat.reshape(1, seq_mat.shape[0], seq_mat.shape[1])
        acc_df['Acceptor Sites'] = acc_df['Acceptor Sites'].apply(expand_1)
        don_df['Donor Sites'] = don_df['Donor Sites'].apply(expand_1)
        # make dataset of shape (len(data), rows, cols) by vertically stacking data
        acc_x = np.vstack(acc_df['Acceptor Sites'].to_numpy())
        don_x = np.vstack(don_df['Donor Sites'].to_numpy())
        return {'acceptor':(acc_x, acc_y), 'donor':(don_x, don_y)}
