# -----------------------------------------------------------------------------
# Copyright (c) 2021 Trevor P. Martin. All rights reserved.
# Distributed under the MIT License.
# -----------------------------------------------------------------------------
from keras.utils import plot_model
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Perceptron
from sklearn.svm import LinearSVC
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
import numpy as np
import pandas as pd
import tensorflow as tf
import copy

class ENSEMBLE(tf.keras.Model):
    @staticmethod
    def build(rows, X_trn, y_trn, X_val, y_val, kwargs):
        model = tf.keras.Sequential()
        input_shape = (rows,)
        model.add(tf.keras.layers.InputLayer(input_shape=input_shape))
        model.add(tf.keras.layers.Dense(2, activation="sigmoid"))
        model.compile(
            loss=tf.keras.losses.BinaryCrossentropy(),
            optimizer=tf.keras.optimizers.Adam(),
            metrics=[
                tf.keras.metrics.BinaryAccuracy(name='bin_acc'),
                tf.keras.metrics.BinaryCrossentropy(name='bin_cross'),
                tf.keras.metrics.AUC(name='auc'),
                tf.keras.metrics.Precision(name='pre'),
                tf.keras.metrics.Recall(name='rec'),
                tf.keras.metrics.TruePositives(name='tru_pos'),
                tf.keras.metrics.TrueNegatives(name='tru_neg'),
                tf.keras.metrics.FalsePositives(name='fal_pos'),
                tf.keras.metrics.FalseNegatives(name='fal_neg'),
            ]
        )
        if kwargs["operation"] == "validate":
            history = model.fit(
                x=X_trn,
                y=y_trn,
                batch_size=32,
                epochs=30,
                validation_data=(X_val, y_val),
                callbacks=[
                    tf.keras.callbacks.EarlyStopping(monitor='loss', patience=9, mode='min', min_delta=0.001),
                ],
                verbose=1,
            )
        if kwargs["operation"] == "train":
            history = model.fit(
                x=X_trn,
                y=y_trn,
                batch_size=32,
                epochs=30,
                callbacks=[
                    tf.keras.callbacks.EarlyStopping(monitor='loss', patience=9, mode='min', min_delta=0.001),
                ],
                verbose=1,
            )
        if kwargs["operation"] == "test":
            return model
        return (history, model)

class CNN01(tf.keras.Model):
    @staticmethod
    def build(rows, X_trn, y_trn, X_val, y_val, kwargs):
        print(X_trn.shape)
        model = tf.keras.Sequential()
        input_shape = (rows, 4)
        model.add(tf.keras.layers.InputLayer(input_shape=input_shape))
        model.add(tf.keras.layers.Conv1D(
            filters=72,
            kernel_size=5,
            activation="relu",
            padding="same"
            )
        )
        model.add(tf.keras.layers.Conv1D(
            filters=144,
            kernel_size=7,
            activation="relu",
            padding="same"
            )
        )
        model.add(tf.keras.layers.Conv1D(
            filters=168,
            kernel_size=7,
            activation="relu",
            padding="same"
            )
        )
        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dropout(rate=0.20))
        model.add(tf.keras.layers.Dense(2, activation="sigmoid"))
        lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(
            initial_learning_rate=0.001,
            decay_steps=120,
            decay_rate=0.9,
            staircase=False
        )
        model.compile(
            loss='binary_crossentropy',
            optimizer=tf.keras.optimizers.Adam(lr_schedule),
            metrics=[
                tf.keras.metrics.BinaryAccuracy(name='bin_acc'),
                tf.keras.metrics.BinaryCrossentropy(name='bin_cross'),
                tf.keras.metrics.AUC(name='auc'),
                tf.keras.metrics.Precision(name='pre'),
                tf.keras.metrics.Recall(name='rec'),
                tf.keras.metrics.TruePositives(name='tru_pos'),
                tf.keras.metrics.TrueNegatives(name='tru_neg'),
                tf.keras.metrics.FalsePositives(name='fal_pos'),
                tf.keras.metrics.FalseNegatives(name='fal_neg'),
                #tf.keras.metrics.PrecisionAtRecall(name='preATrec'),
                # tf.keras.metrics.SensitivityAtSpecificity(name='snATsp'),
                # tf.keras.metrics.SpecificityAtSensitivity(name='spATsn'),
            ]
        )
        plot_model(model, to_file='cnn01.png')
        if kwargs["operation"] == "train":
            history = model.fit(
                x=X_trn,
                y=y_trn,
                batch_size=32,
                epochs=30,
                callbacks=[
                    tf.keras.callbacks.EarlyStopping(monitor='loss', patience=9, mode='min', min_delta=0.001),
                ],
                verbose=1,
            )
        # only the callback and validation_data differ here
        if kwargs["operation"] == "validate":
            history = model.fit(
                x=X_trn,
                y=y_trn,
                batch_size=32,
                epochs=30,
                validation_data=(X_val, y_val),
                callbacks=[
                    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=9,  mode='min', min_delta=0.001),
                ],
                verbose=1,
            )
           
        if kwargs["operation"] == "test":
            return model
        return (history, model)

class CNN02(tf.keras.Model):
    @staticmethod
    def build(rows, X_trn, y_trn, X_val, y_val, kwargs):
        model = tf.keras.Sequential()
        input_shape = (rows, 4)
        model.add(tf.keras.layers.InputLayer(input_shape=input_shape))
        model.add(tf.keras.layers.Conv1D(
            filters=136,
            kernel_size=3,
            activation="relu",
            padding="same"
            )
        )
        model.add(tf.keras.layers.Conv1D(
            filters=72,
            kernel_size=4,
            activation="relu",
            padding="same"
            )
        )
        model.add(tf.keras.layers.MaxPooling1D(pool_size=7, padding="same"))
        model.add(tf.keras.layers.Conv1D(
            filters=272,
            kernel_size=7,
            activation="relu",
            padding="same"
            )
        )
        model.add(tf.keras.layers.MaxPooling1D(pool_size=3, padding="same"))
        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dropout(rate=0.35))
        model.add(tf.keras.layers.Dense(2, activation="sigmoid"))
        lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(
            initial_learning_rate=0.001,
            decay_steps=140,
            decay_rate=0.1,
            staircase=False
        )
        model.compile(
            loss='binary_crossentropy',
            optimizer=tf.keras.optimizers.Adam(lr_schedule),
            metrics=[
                tf.keras.metrics.BinaryAccuracy(name='bin_acc'),
                tf.keras.metrics.BinaryCrossentropy(name='bin_cross'),
                tf.keras.metrics.AUC(name='auc'),
                tf.keras.metrics.Precision(name='pre'),
                tf.keras.metrics.Recall(name='rec'),
                tf.keras.metrics.TruePositives(name='tru_pos'),
                tf.keras.metrics.TrueNegatives(name='tru_neg'),
                tf.keras.metrics.FalsePositives(name='fal_pos'),
                tf.keras.metrics.FalseNegatives(name='fal_neg'),
                #tf.keras.metrics.PrecisionAtRecall(name='preATrec'),
                # tf.keras.metrics.SensitivityAtSpecificity(name='snATsp'),
                # tf.keras.metrics.SpecificityAtSensitivity(name='spATsn'),
            ]
        )
        plot_model(model, to_file='cnn02.png')
        if kwargs["operation"] == "train":
            history = model.fit(
                x=X_trn,
                y=y_trn,
                batch_size=32,
                epochs=30,
                callbacks=[
                    tf.keras.callbacks.EarlyStopping(monitor='loss', patience=9,  mode='min', min_delta=0.001),
                ]
            )
        if kwargs["operation"] == "validate":
            history = model.fit(
                x=X_trn,
                y=y_trn,
                batch_size=32,
                epochs=30,
                validation_data=(X_val, y_val),
                callbacks=[
                    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=9,  mode='min', min_delta=0.001),
                ]
            )
        if kwargs["operation"] == "test":
            return model
        return (history, model)

class CNN03(tf.keras.Model):
    @staticmethod
    def build(rows, X_trn, y_trn, X_val, y_val, kwargs):
        model = tf.keras.Sequential()
        input_shape = (rows, 4)
        model.add(tf.keras.layers.InputLayer(input_shape=input_shape))
        model.add(tf.keras.layers.Conv1D(
            filters=208,
            kernel_size=9,
            activation="relu",
            padding="same"
            )
        )
        model.add(tf.keras.layers.MaxPooling1D(pool_size=6, padding="same"))
        model.add(tf.keras.layers.Conv1D(
            filters=120,
            kernel_size=5,
            activation="relu",
            padding="same"
            )
        )
        model.add(tf.keras.layers.MaxPooling1D(pool_size=3, padding="same"))
        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dropout(rate=0.20))
        model.add(tf.keras.layers.Dense(2, activation="sigmoid"))
        lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(
            initial_learning_rate=0.0065,
            decay_steps=300,
            decay_rate=0.4,
            staircase=False
        )
        model.compile(
            loss='binary_crossentropy',
            optimizer=tf.keras.optimizers.Adam(lr_schedule),
            metrics=[
                tf.keras.metrics.BinaryAccuracy(name='bin_acc'),
                tf.keras.metrics.BinaryCrossentropy(name='bin_cross'),
                tf.keras.metrics.AUC(name='auc'),
                tf.keras.metrics.Precision(name='pre'),
                tf.keras.metrics.Recall(name='rec'),
                tf.keras.metrics.TruePositives(name='tru_pos'),
                tf.keras.metrics.TrueNegatives(name='tru_neg'),
                tf.keras.metrics.FalsePositives(name='fal_pos'),
                tf.keras.metrics.FalseNegatives(name='fal_neg'),
                #tf.keras.metrics.PrecisionAtRecall(name='preATrec'),
                # tf.keras.metrics.SensitivityAtSpecificity(name='snATsp'),
                # tf.keras.metrics.SpecificityAtSensitivity(name='spATsn'),
            ]
        )
        plot_model(model, to_file='cnn03.png')
        if kwargs["operation"] == "train":
            history = model.fit(
                x=X_trn,
                y=y_trn,
                batch_size=32,
                epochs=30,
                callbacks=[
                    tf.keras.callbacks.EarlyStopping(monitor='loss', patience=9,  mode='min', min_delta=0.001),
                ]
            )
        if kwargs["operation"] == "validate":
            history = model.fit(
                x=X_trn,
                y=y_trn,
                batch_size=32,
                epochs=30,
                validation_data=(X_val, y_val),
                callbacks=[
                    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=9,  mode='min', min_delta=0.001),
                ]
            )
        if kwargs["operation"] == "test":
            return model

        return (history, model)

class CNN04(tf.keras.Model):
    @staticmethod
    def build(rows, X_trn, y_trn, X_val, y_val, kwargs):
        model = tf.keras.Sequential()
        input_shape = (rows, 4)
        model.add(tf.keras.layers.InputLayer(input_shape=input_shape))
        model.add(tf.keras.layers.Conv1D(
            filters=250,
            kernel_size=5,
            activation="relu",
            padding="same",
            kernel_initializer="random_normal",
            bias_initializer="zeros",
            )
        )
        model.add(tf.keras.layers.Conv1D(
            filters=250,
            kernel_size=5,
            activation="relu",
            padding="same",
            kernel_initializer="random_normal",
            bias_initializer="zeros",
            )
        )
        model.add(tf.keras.layers.Conv1D(
            filters=250,
            kernel_size=5,
            activation="relu",
            padding="same",
            kernel_initializer="random_normal",
            bias_initializer="zeros",
            )
        )
        model.add(tf.keras.layers.MaxPooling1D(pool_size=3, padding="same"))
        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dropout(rate=0.20))
        model.add(tf.keras.layers.Dense(2, activation="sigmoid"))
        lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(
            initial_learning_rate=0.005,
            decay_steps=350,
            decay_rate=0.4,
            staircase=False
        )
        model.compile(
            loss='binary_crossentropy',
            optimizer=tf.keras.optimizers.Adam(lr_schedule),
            metrics=[
                tf.keras.metrics.BinaryAccuracy(name='bin_acc'),
                tf.keras.metrics.BinaryCrossentropy(name='bin_cross'),
                tf.keras.metrics.AUC(name='auc'),
                tf.keras.metrics.Precision(name='pre'),
                tf.keras.metrics.Recall(name='rec'),
                tf.keras.metrics.TruePositives(name='tru_pos'),
                tf.keras.metrics.TrueNegatives(name='tru_neg'),
                tf.keras.metrics.FalsePositives(name='fal_pos'),
                tf.keras.metrics.FalseNegatives(name='fal_neg'),
                #tf.keras.metrics.PrecisionAtRecall(name='preATrec'),
                # tf.keras.metrics.SensitivityAtSpecificity(name='snATsp'),
                # tf.keras.metrics.SpecificityAtSensitivity(name='spATsn'),
            ]
        )
        plot_model(model, to_file='cnn04.png')
        if kwargs["operation"] == "train":
            history = model.fit(
                x=X_trn,
                y=y_trn,
                batch_size=32,
                epochs=30,
                callbacks=[
                    tf.keras.callbacks.EarlyStopping(monitor='loss', patience=9,  mode='min', min_delta=0.001),
                ]
            )
        if kwargs["operation"] == "validate":
            history = model.fit(
                x=X_trn,
                y=y_trn,
                batch_size=32,
                epochs=30,
                validation_data=(X_val, y_val),
                callbacks=[
                    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=9,  mode='min', min_delta=0.001),
                ]
            )
        if kwargs["operation"] == "test":
            return model

        return (history, model)

class DNN01(tf.keras.Model):
    @staticmethod
    def build(rows, X_trn, y_trn, X_val, y_val, kwargs):
        model = tf.keras.Sequential()
        input_shape = (rows, 4)
        model.add(tf.keras.layers.InputLayer(input_shape=input_shape))
        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(
            units=704,
            activation='relu',
            kernel_regularizer=tf.keras.regularizers.l2(0.025),
            )
        )
        model.add(tf.keras.layers.Dense(
            units=224,
            activation='relu',
            kernel_regularizer=tf.keras.regularizers.l2(0.025),
            )
        )
        model.add(tf.keras.layers.Dropout(rate=0.1))
        model.add(tf.keras.layers.Dense(
            units=512,
            activation='relu',
            )
        )
        model.add(tf.keras.layers.Dropout(rate=0.15))
        model.add(tf.keras.layers.Dense(2, activation="sigmoid"))
        lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(
            initial_learning_rate=0.002,
            decay_steps=80,
            decay_rate=1.4,
            staircase=False
        )
        model.compile(
            loss='binary_crossentropy',
            optimizer=tf.keras.optimizers.Adam(lr_schedule),
            metrics=[
                tf.keras.metrics.BinaryAccuracy(name='bin_acc'),
                tf.keras.metrics.BinaryCrossentropy(name='bin_cross'),
                tf.keras.metrics.AUC(name='auc'),
                tf.keras.metrics.Precision(name='pre'),
                tf.keras.metrics.Recall(name='rec'),
                tf.keras.metrics.TruePositives(name='tru_pos'),
                tf.keras.metrics.TrueNegatives(name='tru_neg'),
                tf.keras.metrics.FalsePositives(name='fal_pos'),
                tf.keras.metrics.FalseNegatives(name='fal_neg'),
                #tf.keras.metrics.PrecisionAtRecall(name='preATrec'),
                # tf.keras.metrics.SensitivityAtSpecificity(name='snATsp'),
                # tf.keras.metrics.SpecificityAtSensitivity(name='spATsn'),
            ]
        )
        if kwargs["operation"] == "train":
            history = model.fit(
                x=X_trn, y=y_trn,
                batch_size=32, epochs=30, # CHANGE to 30, 10 for testing
                callbacks=[
                    tf.keras.callbacks.EarlyStopping(monitor='loss', patience=6, min_delta=0.001, restore_best_weights=True)
                ],
                verbose=1,
            )
        if kwargs["operation"] == "validate":
            history = model.fit(
                x=X_trn, y=y_trn,
                batch_size=32, epochs=30,# CHANGE TO 30, 10 for testing
                validation_data=(X_val, y_val),
                callbacks=[
                    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=6, min_delta=0.001, restore_best_weights=True)
                ],
                verbose=1,
            )
        if kwargs["operation"] == "test":
            return model

        return (history, model)

class DNN02(tf.keras.Model):
    @staticmethod
    def build(rows, X_trn, y_trn, X_val, y_val, kwargs):
        model = tf.keras.Sequential()
        input_shape = (rows, 4)
        model.add(tf.keras.layers.InputLayer(input_shape=input_shape))
        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(
            units=704,
            activation='relu',
            kernel_regularizer=tf.keras.regularizers.l2(0.025),
            )
        )
        model.add(tf.keras.layers.Dense(
            units=224,
            activation='relu',
            kernel_regularizer=tf.keras.regularizers.l2(0.025),
            )
        )
        model.add(tf.keras.layers.Dense(
            units=128,
            activation='relu',
            kernel_regularizer=tf.keras.regularizers.l2(0.036),
            )
        )
        model.add(tf.keras.layers.Dropout(rate=0.15))
        model.add(tf.keras.layers.Dense(2, activation="sigmoid"))
        lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(
            initial_learning_rate=0.002,
            decay_steps=80,
            decay_rate=1.4,
            staircase=False
        )
        model.compile(
            loss='binary_crossentropy',
            optimizer=tf.keras.optimizers.Adam(lr_schedule),
            metrics=[
                tf.keras.metrics.BinaryAccuracy(name='bin_acc'),
                tf.keras.metrics.BinaryCrossentropy(name='bin_cross'),
                tf.keras.metrics.AUC(name='auc'),
                tf.keras.metrics.Precision(name='pre'),
                tf.keras.metrics.Recall(name='rec'),
                tf.keras.metrics.TruePositives(name='tru_pos'),
                tf.keras.metrics.TrueNegatives(name='tru_neg'),
                tf.keras.metrics.FalsePositives(name='fal_pos'),
                tf.keras.metrics.FalseNegatives(name='fal_neg'),
                #tf.keras.metrics.PrecisionAtRecall(name='preATrec'),
                # tf.keras.metrics.SensitivityAtSpecificity(name='snATsp'),
                # tf.keras.metrics.SpecificityAtSensitivity(name='spATsn'),
            ]
        )
        if kwargs["operation"] == "train":
            history = model.fit(
                x=X_trn,
                y=y_trn,
                batch_size=32,
                epochs=30,
                callbacks=[tf.keras.callbacks.EarlyStopping(monitor='loss', patience=6, min_delta=0.001, restore_best_weights=True)]
            )

        if kwargs["operation"] == "validate":
            history = model.fit(
                x=X_trn,
                y=y_trn,
                batch_size=32,
                epochs=30,
                validation_data=(X_val, y_val),
                callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=6, min_delta=0.001, restore_best_weights=True)]
            )
        if kwargs["operation"] == "test":
            return model
        return (history, model)

class DNN03(tf.keras.Model):
    @staticmethod
    def build(rows, X_trn, y_trn, X_val, y_val, kwargs):
        model = tf.keras.Sequential()
        input_shape = (rows, 4)
        model.add(tf.keras.layers.InputLayer(input_shape=input_shape))
        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(
            units=256,
            activation='relu',
            kernel_regularizer=tf.keras.regularizers.l2(0.0025),
            )
        )
        model.add(tf.keras.layers.Dense(
            units=352,
            activation='relu',
            kernel_regularizer=tf.keras.regularizers.l2(0.0025),
            )
        )
        model.add(tf.keras.layers.Dense(
            units=32,
            activation='relu',
            kernel_regularizer=tf.keras.regularizers.l2(0.036),
            )
        )
        model.add(tf.keras.layers.Dense(
            units=352,
            activation='relu',
            )
        )
        model.add(tf.keras.layers.Dropout(rate=0.15))
        model.add(tf.keras.layers.Dense(2, activation="sigmoid"))
        lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(
            initial_learning_rate=0.002,
            decay_steps=240,
            decay_rate=1.5,
            staircase=False
        )
        model.compile(
            loss='binary_crossentropy',
            optimizer=tf.keras.optimizers.Adam(lr_schedule),
            metrics=[
                tf.keras.metrics.BinaryAccuracy(name='bin_acc'),
                tf.keras.metrics.BinaryCrossentropy(name='bin_cross'),
                tf.keras.metrics.AUC(name='auc'),
                tf.keras.metrics.Precision(name='pre'),
                tf.keras.metrics.Recall(name='rec'),
                tf.keras.metrics.TruePositives(name='tru_pos'),
                tf.keras.metrics.TrueNegatives(name='tru_neg'),
                tf.keras.metrics.FalsePositives(name='fal_pos'),
                tf.keras.metrics.FalseNegatives(name='fal_neg'),
                #tf.keras.metrics.PrecisionAtRecall(name='preATrec'),
                # tf.keras.metrics.SensitivityAtSpecificity(name='snATsp'),
                # tf.keras.metrics.SpecificityAtSensitivity(name='spATsn'),
            ]
        )
        if kwargs["operation"] == "train":
            history = model.fit(
                x=X_trn,
                y=y_trn,
                batch_size=32,
                epochs=30,
                callbacks=[tf.keras.callbacks.EarlyStopping(monitor='loss', patience=6, min_delta=0.001, restore_best_weights=True)]
            )
        if kwargs["operation"] == "validate":
            history = model.fit(
                x=X_trn,
                y=y_trn,
                batch_size=32,
                epochs=30,
                validation_data=(X_val, y_val),
                callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=6, min_delta=0.001, restore_best_weights=True)]
            )
        if kwargs["operation"] == "test":
            return model
        return (history, model)

class DNN04(tf.keras.Model):
    @staticmethod
    def build(rows, X_trn, y_trn, X_val, y_val, kwargs):
        model = tf.keras.Sequential()
        input_shape = (rows, 4)
        model.add(tf.keras.layers.InputLayer(input_shape=input_shape))
        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(
            units=250,
            activation='relu',
            kernel_initializer='random_normal',
            bias_initializer="zeros",
            )
        )
        model.add(tf.keras.layers.Dense(
            units=250,
            activation='relu',
            kernel_initializer='random_normal',
            bias_initializer="zeros",
            )
        )
        model.add(tf.keras.layers.Dense(
            units=250,
            activation='relu',
            kernel_initializer='random_normal',
            bias_initializer="zeros",
            )
        )
        model.add(tf.keras.layers.Dropout(rate=0.25))
        model.add(tf.keras.layers.Dense(2, activation="sigmoid"))
        lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(
            initial_learning_rate=0.001,
            decay_steps=50,
            decay_rate=1.5,
            staircase=False
        )
        model.compile(
            loss='binary_crossentropy',
            optimizer=tf.keras.optimizers.Adam(lr_schedule),
            metrics=[
                tf.keras.metrics.BinaryAccuracy(name='bin_acc'),
                tf.keras.metrics.BinaryCrossentropy(name='bin_cross'),
                tf.keras.metrics.AUC(name='auc'),
                tf.keras.metrics.Precision(name='pre'),
                tf.keras.metrics.Recall(name='rec'),
                tf.keras.metrics.TruePositives(name='tru_pos'),
                tf.keras.metrics.TrueNegatives(name='tru_neg'),
                tf.keras.metrics.FalsePositives(name='fal_pos'),
                tf.keras.metrics.FalseNegatives(name='fal_neg'),
                #tf.keras.metrics.PrecisionAtRecall(name='preATrec'),
                # tf.keras.metrics.SensitivityAtSpecificity(name='snATsp'),
                # tf.keras.metrics.SpecificityAtSensitivity(name='spATsn'),
            ]
        )
        if kwargs["operation"] == "train":
            history = model.fit(
                x=X_trn,
                y=y_trn,
                batch_size=32,
                epochs=30,
                callbacks=[tf.keras.callbacks.EarlyStopping(monitor='loss', patience=6, min_delta=0.001, restore_best_weights=True)]
            )

        if kwargs["operation"] == "validate":
            history = model.fit(
                x=X_trn,
                y=y_trn,
                batch_size=32,
                epochs=30,
                validation_data=(X_val, y_val),
            )
        if kwargs["operation"] == "test":
            return model
        return (history, model)

class SpliceFinder(tf.keras.Model):
    @staticmethod
    def build(rows, X_trn, y_trn, X_val, y_val, kwargs):
        print(X_trn.shape)
        model = tf.keras.Sequential()
        input_shape = (rows, 4)
        model.add(tf.keras.layers.InputLayer(input_shape=input_shape))
        model.add(tf.keras.layers.Conv1D(
            filters=50,
            kernel_size=9,
            strides=1,
            activation="relu",
            padding="same"
            )
        )
        model.add(tf.keras.layers.Flatten())
        
        model.add(tf.keras.layers.Dense(100,activation='relu'))
    
        model.add(tf.keras.layers.Dropout(0.3))
        model.add(tf.keras.layers.Dense(2, activation="sigmoid"))

        adam = tf.keras.optimizers.Adam(lr=1e-4)
        model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=[tf.keras.metrics.BinaryAccuracy(name='bin_acc'),])

        if kwargs["operation"] == "train":
            history = model.fit(
                x=X_trn,
                y=y_trn,
                batch_size=40,
                epochs=50,
                callbacks=[
                    tf.keras.callbacks.EarlyStopping(monitor='loss', patience=9, mode='min', min_delta=0.001),
                ],
                verbose=1,
            )
        # only the callback and validation_data differ here
        if kwargs["operation"] == "validate":
            history = model.fit(
                x=X_trn,
                y=y_trn,
                batch_size=40,
                epochs=50,
                validation_data=(X_val, y_val),
                callbacks=[
                    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=9,  mode='min', min_delta=0.001),
                ],
                verbose=1,
            )
        if kwargs["operation"] == "test":
            return model
        return (history, model)

class DeepSplicer(tf.keras.Model):
    @staticmethod
    def build(rows, X_trn, y_trn, X_val, y_val, kwargs):
        print(X_trn.shape)
        model = tf.keras.Sequential()
        input_shape = (rows, 4)
        model.add(tf.keras.layers.InputLayer(input_shape=input_shape))
        model.add(tf.keras.layers.Conv1D(
            filters=50,
            kernel_size=9,
            strides=1,
            activation="relu",
            padding="same"
            )
        )
        model.add(tf.keras.layers.Conv1D(
            filters=50,
            kernel_size=9,
            strides=1,
            activation="relu",
            padding="same"
            )
        )
        model.add(tf.keras.layers.Conv1D(
            filters=50,
            kernel_size=9,
            strides=1,
            activation="relu",
            padding="same"
            )
        )
        model.add(tf.keras.layers.Flatten())
        
        model.add(tf.keras.layers.Dense(100,activation='relu'))
    
        model.add(tf.keras.layers.Dropout(0.3))
        model.add(tf.keras.layers.Dense(2, activation="sigmoid"))

        adam = tf.keras.optimizers.Adam(lr=1e-4)
        model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=[tf.keras.metrics.BinaryAccuracy(name='bin_acc'),])

        if kwargs["operation"] == "train":
            history = model.fit(
                x=X_trn,
                y=y_trn,
                batch_size=40,
                epochs=50,
                callbacks=[
                    tf.keras.callbacks.EarlyStopping(monitor='loss', patience=9, mode='min', min_delta=0.001),
                ],
                verbose=1,
            )
        # only the callback and validation_data differ here
        if kwargs["operation"] == "validate":
            history = model.fit(
                x=X_trn,
                y=y_trn,
                batch_size=40,
                epochs=50,
                validation_data=(X_val, y_val),
                callbacks=[
                    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=9,  mode='min', min_delta=0.001),
                ],
                verbose=1,
            )
        if kwargs["operation"] == "test":
            return model
        return (history, model)

# class CNN04(tf.keras.Model):
#     @staticmethod
#     def build(rows, X_trn, y_trn, X_val, y_val, kwargs):
#         model = tf.keras.Sequential()
#         input_shape = (rows, 4)
#         model.add(tf.keras.layers.InputLayer(input_shape=input_shape))
#         model.add(tf.keras.layers.Conv1D(
#             filters=144,
#             kernel_size=10,
#             activation="relu",
#             padding="same"
#             )
#         )
#         model.add(tf.keras.layers.MaxPooling1D(pool_size=5, padding="same"))
#         model.add(tf.keras.layers.Flatten())
#         model.add(tf.keras.layers.Dropout(rate=0.1))
#         model.add(tf.keras.layers.Dense(2, activation="sigmoid"))
#         lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(
#             initial_learning_rate=0.005,
#             decay_steps=460,
#             decay_rate=2.0,
#             staircase=False
#         )
#         model.compile(
#             loss='binary_crossentropy',
#             optimizer=tf.keras.optimizers.Adam(lr_schedule),
#             metrics=[
#                 tf.keras.metrics.BinaryAccuracy(name='bin_acc'),
#                 tf.keras.metrics.BinaryCrossentropy(name='bin_cross'),
#                 tf.keras.metrics.AUC(name='auc'),
#                 tf.keras.metrics.Precision(name='pre'),
#                 tf.keras.metrics.Recall(name='rec'),
#                 tf.keras.metrics.TruePositives(name='tru_pos'),
#                 tf.keras.metrics.TrueNegatives(name='tru_neg'),
#                 tf.keras.metrics.FalsePositives(name='fal_pos'),
#                 tf.keras.metrics.FalseNegatives(name='fal_neg'),
#                 #tf.keras.metrics.PrecisionAtRecall(name='preATrec'),
#                 # tf.keras.metrics.SensitivityAtSpecificity(name='snATsp'),
#                 # tf.keras.metrics.SpecificityAtSensitivity(name='spATsn'),
#             ]
#         )
#         if kwargs["operation"] == "train":
#             history = model.fit(
#                 x=X_trn,
#                 y=y_trn,
#                 batch_size=32,
#                 epochs=30,
#                 callbacks=[tf.keras.callbacks.EarlyStopping(monitor='loss', patience=6, min_delta=0.001, restore_best_weights=True)]
#             )
#         if kwargs["operation"] == "validate":
#             history = model.fit(
#                 x=X_trn,
#                 y=y_trn,
#                 batch_size=32,
#                 epochs=30,
#                 validation_data=(X_val, y_val),
#                 callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=6, min_delta=0.001, restore_best_weights=True)]
#             )
#         if kwargs["operation"] == "test":
#             return model
#         return (history, model)
#
# class CNN05(tf.keras.Model):
#     @staticmethod
#     def build(rows, X_trn, y_trn, X_val, y_val, kwargs):
#         model = tf.keras.Sequential()
#         input_shape = (rows, 4, 1)
#         model.add(tf.keras.layers.InputLayer(input_shape=input_shape))
#
#         model.add(tf.keras.layers.Conv2D(
#             filters=56,
#             kernel_size=(3, 3),
#             activation="relu",
#             padding="same"
#             )
#         )
#         model.add(tf.keras.layers.Conv2D(
#             filters=48,
#             kernel_size=(1, 1),
#             activation="relu",
#             padding="same"
#             )
#         )
#         model.add(tf.keras.layers.Conv2D(
#             filters=400,
#             kernel_size=(3, 3),
#             activation="relu",
#             padding="same"
#             )
#         )
#         model.add(tf.keras.layers.MaxPooling2D(
#             pool_size=(5, 5),
#             padding='same'
#             )
#         )
#         model.add(tf.keras.layers.Conv2D(
#             filters=80,
#             kernel_size=(4,4),
#             activation="relu",
#             padding="same"
#             )
#         )
#
#         model.add(tf.keras.layers.Flatten())
#         model.add(tf.keras.layers.Dropout(rate=0.2))
#         model.add(tf.keras.layers.Dense(2, activation="sigmoid"))
#         lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(
#             initial_learning_rate=0.0025,
#             decay_steps=540,
#             decay_rate=0.5,
#             staircase=False
#         )
#         model.compile(
#             loss='binary_crossentropy',
#             optimizer=tf.keras.optimizers.Adam(lr_schedule),
#             metrics=[
#                 tf.keras.metrics.BinaryAccuracy(name='bin_acc'),
#                 tf.keras.metrics.BinaryCrossentropy(name='bin_cross'),
#                 tf.keras.metrics.AUC(name='auc'),
#                 tf.keras.metrics.Precision(name='pre'),
#                 tf.keras.metrics.Recall(name='rec'),
#                 tf.keras.metrics.TruePositives(name='tru_pos'),
#                 tf.keras.metrics.TrueNegatives(name='tru_neg'),
#                 tf.keras.metrics.FalsePositives(name='fal_pos'),
#                 tf.keras.metrics.FalseNegatives(name='fal_neg'),
#                 #tf.keras.metrics.PrecisionAtRecall(name='preATrec'),
#                 # tf.keras.metrics.SensitivityAtSpecificity(name='snATsp'),
#                 # tf.keras.metrics.SpecificityAtSensitivity(name='spATsn'),
#             ]
#         )
#         if kwargs["operation"] == "train":
#             history = model.fit(
#                 x=X_trn,
#                 y=y_trn,
#                 batch_size=32,
#                 epochs=30,
#                 callbacks=[tf.keras.callbacks.EarlyStopping(monitor='loss', patience=6, min_delta=0.001, restore_best_weights=True)]
#             )
#         if kwargs["operation"] == "validate":
#             history = model.fit(
#                 x=X_trn,
#                 y=y_trn,
#                 batch_size=32,
#                 epochs=30,
#                 validation_data=(X_val, y_val),
#                 callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=6, min_delta=0.001, restore_best_weights=True)]
#             )
#         if kwargs["operation"] == "test":
#             return model
#         return (history, model)
