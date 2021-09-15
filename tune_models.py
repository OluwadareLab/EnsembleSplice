from Models import sub_models
from Data import encode_data
from Visuals import make_report

import kerastuner as kt
import tensorflow as tf
from tensorflow import keras

def tune_cnn1(
    tuner,
    row=602,
    columns=4,
    channels=1,
    classes=2,
    ):
    model = tf.keras.Sequential()
    input_shape = (row, columns, channels)
    model.add(tf.keras.layers.InputLayer(input_shape=input_shape))

    # tuner_cnn_act = tuner.Choice(name='cnn act', values=['relu','tanh','elu'])

    tuner_filters_1 = tuner.Int(name='filters 1', min_value=8, max_value=400, step=8)
    tuner_kernal_1 = tuner.Int(name='kernal 1', min_value=1, max_value=10, step=1)
    #tuner_stride_1 = tuner.Int(name='stride 1', min_value=1, max_value=10, step=1)
    #tuner_cnn_act_1 = tuner.Choice(name='act 1', values=['relu','tanh','elu'])
    # tuner_kern_init_1 = tuner.Choice(
    #     name='kern init 1',
    #     values=[
    #         'random_normal','random_uniform',
    #         'zeros','ones',
    #         'glorot_normal','glorot_uniform',
    #     ]
    # )
    # tuner_bias_init_1 = tuner.Choice(
    #     name='bias init 1',
    #     values=[
    #         'random_normal', 'random_uniform',
    #         'zeros','ones',
    #         'glorot_normal','glorot_uniform',
    #     ]
    # )
    # tuner_act_reg_1 = tuner.Float(name='act reg 1', min_value=0.001, max_value=0.025, step=0.0005)
    # tuner_kern_reg_1 = tuner.Float(name='kern reg 1', min_value=0.001, max_value=0.025, step=0.0005)
    # tuner_bias_reg_1 = tuner.Float(name='bias reg 1', min_value=0.001, max_value=0.025, step=0.0005)
    model.add(tf.keras.layers.Conv2D(
        filters=tuner_filters_1,
        kernel_size=(tuner_kernal_1, tuner_kernal_1),
        # strides=(tuner_stride_1, tuner_stride_1),
        activation='relu',
        # activation=tuner_cnn_act,
        # kernel_initializer=tuner_kern_init_1,
        # bias_initializer=tuner_bias_init_1,
        # kernel_regularizer=tf.keras.regularizers.l1(tuner_kern_reg_1),
        # bias_regularizer=tf.keras.regularizers.l1(tuner_bias_reg_1),
        # activity_regularizer=tf.keras.regularizers.l2(tuner_act_reg_1),
        padding="same"
        )
    )
    tuner_filters_2 = tuner.Int(name='filters 2', min_value=8, max_value=400, step=8)
    tuner_kernal_2 = tuner.Int(name='kernal 2', min_value=1, max_value=10, step=1)
    # tuner_stride_2 = tuner.Int(name='stride 2', min_value=1, max_value=10, step=1)
    #tuner_cnn_act_2 = tuner.Choice(name='act 2', values=['relu','tanh','elu'])
    # tuner_kern_init_2 = tuner.Choice(
    #     name='kern init 2',
    #     values=[
    #         'random_normal','random_uniform',
    #         'zeros','ones',
    #         'glorot_normal','glorot_uniform',
    #     ]
    # )
    # tuner_bias_init_2 = tuner.Choice(
    #     name='bias init 2',
    #     values=[
    #         'random_normal', 'random_uniform',
    #         'zeros','ones',
    #         'glorot_normal','glorot_uniform',
    #     ]
    # )
    # tuner_act_reg_2 = tuner.Float(name='act reg 2', min_value=0.001, max_value=0.025, step=0.0005)
    # tuner_kern_reg_2 = tuner.Float(name='kern reg 2', min_value=0.001, max_value=0.025, step=0.0005)
    # tuner_bias_reg_2 = tuner.Float(name='bias reg 2', min_value=0.001, max_value=0.025, step=0.0005)
    model.add(tf.keras.layers.Conv2D(
        filters=tuner_filters_2,
        kernel_size=(tuner_kernal_2, tuner_kernal_2),
        # strides=(tuner_stride_2, tuner_stride_2),
        activation='relu',
        # kernel_initializer=tuner_kern_init_2,
        # bias_initializer=tuner_bias_init_2,
        # kernel_regularizer=tf.keras.regularizers.l1(tuner_kern_reg_2),
        # bias_regularizer=tf.keras.regularizers.l1(tuner_bias_reg_2),
        # activity_regularizer=tf.keras.regularizers.l2(tuner_act_reg_2),
        padding="same"
        )
    )

    #tuner_pooling_1 = tuner.Int(name='pooling 1', min_value=1, max_value=10, step=1)
    #tuner_pool_strides_1 = tuner.Int(name='pool stride 1', min_value=1, max_value=10, step=1)
    #model.add(tf.keras.layers.MaxPooling2D(
        #pool_size=(tuner_pooling_1, tuner_pooling_1),
        # strides=(tuner_pool_strides_1, tuner_pool_strides_1),
        #padding='same'
        #)
    #)

    tuner_filters_3 = tuner.Int(name='filters 3', min_value=8, max_value=400, step=8)
    tuner_kernal_3 = tuner.Int(name='kernal 3', min_value=1, max_value=10, step=1)
    #tuner_stride_3 = tuner.Int(name='stride 3', min_value=1, max_value=10, step=1)
    #tuner_cnn_act_3 = tuner.Choice(name='act 3', values=['relu','tanh','elu'])
    # tuner_kern_init_3 = tuner.Choice(
    #     name='kern init 3',
    #     values=[
    #         'random_normal','random_uniform',
    #         'zeros','ones',
    #         'glorot_normal','glorot_uniform',
    #     ]
    # )
    # tuner_bias_init_3 = tuner.Choice(
    #     name='bias init 3',
    #     values=[
    #         'random_normal', 'random_uniform',
    #         'zeros','ones',
    #         'glorot_normal','glorot_uniform',
    #     ]
    # )
    # tuner_act_reg_3 = tuner.Float(name='act reg 3', min_value=0.001, max_value=0.025, step=0.0005)
    # tuner_kern_reg_3 = tuner.Float(name='kern reg 3', min_value=0.001, max_value=0.025, step=0.0005)
    # tuner_bias_reg_3 = tuner.Float(name='bias reg 3', min_value=0.001, max_value=0.025, step=0.0005)
    model.add(tf.keras.layers.Conv2D(
        filters=tuner_filters_3,
        kernel_size=(tuner_kernal_3, tuner_kernal_3),
        # strides=(tuner_stride_3, tuner_stride_3),
        activation='relu',
        # kernel_initializer=tuner_kern_init_3,
        # bias_initializer=tuner_bias_init_3,
        # kernel_regularizer=tf.keras.regularizers.l1(tuner_kern_reg_3),
        # bias_regularizer=tf.keras.regularizers.l1(tuner_bias_reg_3),
        # activity_regularizer=tf.keras.regularizers.l2(tuner_act_reg_3),
        padding="same"
        )
    )

    # tuner_pooling_2 = tuner.Int(name='pooling 2', min_value=1, max_value=10, step=1)
    # #tuner_pool_strides_2 = tuner.Int(name='pool stride 2', min_value=1, max_value=10, step=1)
    # model.add(tf.keras.layers.MaxPooling2D(
    #     pool_size=(tuner_pooling_2, tuner_pooling_2),
    #     # strides=(tuner_pool_strides_2, tuner_pool_strides_2),
    #     padding='same'
    #     )
    # )

    model.add(tf.keras.layers.Flatten())

    tuner_dropout_1 = tuner.Float(name='dropout 1', min_value=0.05, max_value=0.5, step=0.05)
    model.add(tf.keras.layers.Dropout(tuner_dropout_1))

    model.add(tf.keras.layers.Dense(classes, activation="softmax"))

    tuner_initial_lr = tuner.Float(name='learning_rate', min_value=0.001, max_value=0.025, step=0.001)
    tuner_decay_steps = tuner.Int(name='decay_steps', min_value=20, max_value=600, step=20)
    tuner_decay_rate = tuner.Float(name='decay_rate', min_value=0.1, max_value=2.0, step=0.1)
    lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(
        initial_learning_rate=tuner_initial_lr,
        decay_steps=tuner_decay_steps,
        decay_rate=tuner_decay_rate,
        staircase=False
    )
    model.compile(
        loss='binary_crossentropy',
        optimizer=tf.keras.optimizers.Adam(lr_schedule),
        metrics=['accuracy']
    )

    # tuner_optimizers = tuner.Choice(name='opimizer', values=['adam', 'sgd', 'RMSprop'])
    # if tuner_optimizers == 'adam':
    #     model.compile(
    #         loss='binary_crossentropy',
    #         optimizer=tf.keras.optimizers.Adam(lr_schedule),
    #         metrics=['accuracy']
    #     )
    # if tuner_optimizers == 'sgd':
    #     model.compile(
    #         loss='binary_crossentropy',
    #         optimizer=tf.keras.optimizers.SGD(lr_schedule),
    #         metrics=['accuracy']
    #     )
    # if tuner_optimizers == 'RMSprop':
    #     model.compile(
    #         loss='binary_crossentropy',
    #         optimizer=tf.keras.optimizers.RMSprop(lr_schedule),
    #         metrics=['accuracy']
    #     )

    return model

def tune_cnn2(
    tuner,
    row=602,
    columns=4,
    channels=1,
    classes=2,
    ):
    model = tf.keras.Sequential()
    input_shape = (row, columns)
    model.add(tf.keras.layers.InputLayer(input_shape=input_shape))


    tuner_filters_1 = tuner.Int(name='filters 1', min_value=8, max_value=400, step=8)
    tuner_kernal_1 = tuner.Int(name='kernal 1', min_value=1, max_value=10, step=1)
    model.add(tf.keras.layers.Conv1D(
        filters=tuner_filters_1,
        kernel_size=tuner_kernal_1,
        activation='relu',
        padding="same"
        )
    )
    tuner_filters_2 = tuner.Int(name='filters 2', min_value=8, max_value=400, step=8)
    tuner_kernal_2 = tuner.Int(name='kernal 2', min_value=1, max_value=10, step=1)
    model.add(tf.keras.layers.Conv1D(
        filters=tuner_filters_2,
        kernel_size=tuner_kernal_2,
        activation='relu',
        padding="same"
        )
    )

    tuner_pooling_1 = tuner.Int(name='pooling 1', min_value=1, max_value=10, step=1)
    model.add(tf.keras.layers.MaxPooling1D(
        pool_size=tuner_pooling_1,
        padding='same'
        )
    )

    tuner_filters_3 = tuner.Int(name='filters 3', min_value=8, max_value=400, step=8)
    tuner_kernal_3 = tuner.Int(name='kernal 3', min_value=1, max_value=10, step=1)
    model.add(tf.keras.layers.Conv1D(
        filters=tuner_filters_3,
        kernel_size=tuner_kernal_3,
        activation='relu',
        padding="same"
        )
    )

    # tuner_pooling_2 = tuner.Int(name='pooling 2', min_value=1, max_value=10, step=1)
    # model.add(tf.keras.layers.MaxPooling1D(
    #     pool_size=tuner_pooling_2,
    #     padding='same'
    #     )
    # )

    model.add(tf.keras.layers.Flatten())

    tuner_dropout_1 = tuner.Float(name='dropout 1', min_value=0.05, max_value=0.5, step=0.05)
    model.add(tf.keras.layers.Dropout(tuner_dropout_1))

    model.add(tf.keras.layers.Dense(classes, activation="softmax"))

    tuner_initial_lr = tuner.Float(name='learning_rate', min_value=0.001, max_value=0.025, step=0.001)
    tuner_decay_steps = tuner.Int(name='decay_steps', min_value=20, max_value=600, step=20)
    tuner_decay_rate = tuner.Float(name='decay_rate', min_value=0.1, max_value=2.0, step=0.1)
    lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(
        initial_learning_rate=tuner_initial_lr,
        decay_steps=tuner_decay_steps,
        decay_rate=tuner_decay_rate,
        staircase=False
    )
    model.compile(
        loss='binary_crossentropy',
        optimizer=tf.keras.optimizers.Adam(lr_schedule),
        metrics=['accuracy']
    )

    return model

def tune_cnn3(
    tuner,
    row=602,
    columns=4,
    channels=1,
    classes=2,
    ):
    model = tf.keras.Sequential()
    input_shape = (row, columns)
    model.add(tf.keras.layers.InputLayer(input_shape=input_shape))


    tuner_filters_1 = tuner.Int(name='filters 1', min_value=8, max_value=400, step=8)
    tuner_kernal_1 = tuner.Int(name='kernal 1', min_value=1, max_value=10, step=1)
    model.add(tf.keras.layers.Conv1D(
        filters=tuner_filters_1,
        kernel_size=tuner_kernal_1,
        activation='relu',
        padding="same"
        )
    )

    # tuner_pooling_1 = tuner.Int(name='pooling 1', min_value=1, max_value=10, step=1)
    # model.add(tf.keras.layers.MaxPooling1D(
    #     pool_size=tuner_pooling_1,
    #     padding='same'
    #     )
    # )


    tuner_filters_2 = tuner.Int(name='filters 2', min_value=8, max_value=400, step=8)
    tuner_kernal_2 = tuner.Int(name='kernal 2', min_value=1, max_value=10, step=1)
    model.add(tf.keras.layers.Conv1D(
        filters=tuner_filters_2,
        kernel_size=tuner_kernal_2,
        activation='relu',
        padding="same"
        )
    )

    # tuner_pooling_2 = tuner.Int(name='pooling 2', min_value=1, max_value=10, step=1)
    # model.add(tf.keras.layers.MaxPooling1D(
    #     pool_size=tuner_pooling_2,
    #     padding='same'
    #     )
    # )

    model.add(tf.keras.layers.Flatten())

    tuner_dropout_1 = tuner.Float(name='dropout 1', min_value=0.05, max_value=0.5, step=0.05)
    model.add(tf.keras.layers.Dropout(tuner_dropout_1))

    model.add(tf.keras.layers.Dense(classes, activation="softmax"))

    tuner_initial_lr = tuner.Float(name='learning_rate', min_value=0.001, max_value=0.025, step=0.001)
    tuner_decay_steps = tuner.Int(name='decay_steps', min_value=20, max_value=600, step=20)
    tuner_decay_rate = tuner.Float(name='decay_rate', min_value=0.1, max_value=2.0, step=0.1)
    lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(
        initial_learning_rate=tuner_initial_lr,
        decay_steps=tuner_decay_steps,
        decay_rate=tuner_decay_rate,
        staircase=False
    )
    model.compile(
        loss='binary_crossentropy',
        optimizer=tf.keras.optimizers.Adam(lr_schedule),
        metrics=['accuracy']
    )

    return model

def tune_cnn4(
    tuner,
    row=602,
    columns=4,
    classes=2,
    ):
    model = tf.keras.Sequential()
    input_shape = (row, columns)
    model.add(tf.keras.layers.InputLayer(input_shape=input_shape))

    tuner_filters_1 = tuner.Int(name='filters 1', min_value=8, max_value=400, step=8)
    tuner_kernal_1 = tuner.Int(name='kernal 1', min_value=1, max_value=10, step=1)
    model.add(tf.keras.layers.Conv1D(
        filters=tuner_filters_1,
        kernel_size=tuner_kernal_1,
        activation="relu",
        padding="same"
        )
    )
    tuner_pooling_1 = tuner.Int(name='pooling 1', min_value=1, max_value=10, step=1)
    model.add(tf.keras.layers.MaxPooling1D(pool_size=tuner_pooling_1, padding="same"))

    model.add(tf.keras.layers.Flatten())

    tuner_dropout_1 = tuner.Float(name='dropout 1', min_value=0.05, max_value=0.5, step=0.05)
    model.add(tf.keras.layers.Dropout(tuner_dropout_1))


    model.add(tf.keras.layers.Dense(classes, activation="softmax"))

    tuner_initial_lr = tuner.Float(name='learning_rate', min_value=0.001, max_value=0.025, step=0.0005)
    tuner_decay_steps = tuner.Int(name='decay_steps', min_value=20, max_value=600, step=20)
    tuner_decay_rate = tuner.Float(name='decay_rate', min_value=0.05, max_value=2.0, step=0.05)
    lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(
        initial_learning_rate=tuner_initial_lr,
        decay_steps=tuner_decay_steps,
        decay_rate=tuner_decay_rate,
        staircase=False
    )
    model.compile(
        loss='binary_crossentropy',
        optimizer=tf.keras.optimizers.Adam(lr_schedule),
        metrics=['accuracy']
    )
    return model

def tune_cnn5(
    tuner,
    row=602,
    columns=4,
    channels=1,
    classes=2,
    ):
    model = tf.keras.Sequential()
    input_shape = (row, columns, channels)
    model.add(tf.keras.layers.InputLayer(input_shape=input_shape))

    tuner_filters_1 = tuner.Int(name='filters 1', min_value=8, max_value=400, step=8)
    tuner_kernal_1 = tuner.Choice(name='kernal 1', values=[1, 2, 3, 4, 5, 6])
    model.add(tf.keras.layers.Conv2D(
        filters=tuner_filters_1,
        kernel_size=(tuner_kernal_1, tuner_kernal_1),
        activation="relu",
        padding="same"
        )
    )
    tuner_filters_2 = tuner.Int(name='filters 2', min_value=8, max_value=400, step=8)
    tuner_kernal_2 = tuner.Choice(name='kernal 2', values=[1, 2, 3, 4, 5, 6])
    model.add(tf.keras.layers.Conv2D(
        filters=tuner_filters_2,
        kernel_size=(tuner_kernal_2, tuner_kernal_2),
        activation="relu",
        padding="same"
        )
    )
    tuner_filters_3 = tuner.Int(name='filters 3', min_value=8, max_value=400, step=8)
    tuner_kernal_3 = tuner.Choice(name='kernal 3', values=[1, 2, 3, 4, 5, 6])
    model.add(tf.keras.layers.Conv2D(
        filters=tuner_filters_3,
        kernel_size=(tuner_kernal_3, tuner_kernal_3),
        activation="relu",
        padding="same"
        )
    )

    tuner_pooling_1 = tuner.Choice(name='pooling 1', values=[1, 2, 3, 4, 5, 6])
    model.add(tf.keras.layers.MaxPooling2D(
        pool_size=(tuner_pooling_1, tuner_pooling_1),
        padding='same'
        )
    )

    tuner_filters_4 = tuner.Int(name='filters 4', min_value=8, max_value=400, step=8)
    tuner_kernal_4 = tuner.Choice(name='kernal 4', values=[1, 2, 3, 4, 5, 6])
    model.add(tf.keras.layers.Conv2D(
        filters=tuner_filters_4,
        kernel_size=(tuner_kernal_4, tuner_kernal_4),
        activation="relu",
        padding="same"
        )
    )
    model.add(tf.keras.layers.Flatten())

    tuner_dropout_1 = tuner.Float(name='dropout 1', min_value=0.05, max_value=0.5, step=0.05)
    model.add(tf.keras.layers.Dropout(tuner_dropout_1))

    model.add(tf.keras.layers.Dense(classes, activation="softmax"))

    tuner_initial_lr = tuner.Float(name='learning_rate', min_value=0.001, max_value=0.025, step=0.0005)
    tuner_decay_steps = tuner.Int(name='decay_steps', min_value=20, max_value=600, step=20)
    tuner_decay_rate = tuner.Float(name='decay_rate', min_value=0.05, max_value=2.0, step=0.05)

    lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(
        initial_learning_rate=tuner_initial_lr,
        decay_steps=tuner_decay_steps,
        decay_rate=tuner_decay_rate,
        staircase=False
    )
    model.compile(
        loss='binary_crossentropy',
        optimizer=tf.keras.optimizers.Adam(lr_schedule),
        metrics=['accuracy']
    )
    return model

def tune_dnn1(
    tuner,
    row=602,
    columns=4,
    classes=2,
    ):
    model = tf.keras.Sequential()
    input_shape = (row, columns)
    model.add(tf.keras.layers.InputLayer(input_shape=input_shape))
    model.add(tf.keras.layers.Flatten())

    tuner_units_1 = tuner.Int(name='units 1', min_value=32, max_value=704, step=32)
    tuner_regularizer_1 =  tuner.Choice(name='regularizer 1', values=[0.001, 0.0025, 0.005, 0.01, 0.025, 0.05])
    model.add(tf.keras.layers.Dense(
        units=tuner_units_1,
        activation='relu',
        kernel_regularizer=tf.keras.regularizers.l2(tuner_regularizer_1),
        )
    )

    tuner_units_2 = tuner.Int(name='units 2', min_value=32, max_value=704, step=32)
    tuner_regularizer_2 =  tuner.Choice(name='regularizer 2', values=[0.001, 0.0025, 0.005, 0.01, 0.025, 0.05])
    model.add(tf.keras.layers.Dense(
        units=tuner_units_2,
        activation='relu',
        kernel_regularizer=tf.keras.regularizers.l2(tuner_regularizer_2),
        )
    )

    tuner_dropout_1 = tuner.Float(name='dropout 1', min_value=0.05, max_value=0.5, step=0.05)
    model.add(tf.keras.layers.Dropout(tuner_dropout_1))


    tuner_units_3 = tuner.Int(name='units 3', min_value=32, max_value=704, step=32)
    #tuner_regularizer_3 = tuner.Float(name='regularizer 3', min_value=0.001, max_value=0.05, step=0.001)
    model.add(tf.keras.layers.Dense(
        units=tuner_units_3,
        activation='relu',
        #kernel_regularizer=tf.keras.regularizers.l2(tuner_regularizer_3),
        )
    )


    # tuner_units_4 = tuner.Int(name='units 4', min_value=32, max_value=704, step=32)
    # tuner_regularizer_4 = tuner.Float(name='regularizer 4', min_value=0.001, max_value=0.05, step=0.001)
    # model.add(tf.keras.layers.Dense(
    #     units=tuner_units_4,
    #     activation='relu',
    #     kernel_regularizer=tf.keras.regularizers.l2(tuner_regularizer_4),
    #     )
    # )
    #
    # tuner_units_5 = tuner.Int(name='units 5', min_value=32, max_value=704, step=32)
    # tuner_regularizer_5 = tuner.Float(name='regularizer 5', min_value=0.001, max_value=0.05, step=0.001)
    # model.add(tf.keras.layers.Dense(
    #     units=tuner_units_5,
    #     activation='relu',
    #     kernel_regularizer=tf.keras.regularizers.l2(tuner_regularizer_5),
    #     )
    # )
    #
    # tuner_units_6 = tuner.Int(name='units 6', min_value=32, max_value=704, step=32)
    # tuner_regularizer_6 = tuner.Float(name='regularizer 6', min_value=0.001, max_value=0.05, step=0.001)
    # model.add(tf.keras.layers.Dense(
    #     units=tuner_units_6,
    #     activation='relu',
    #     kernel_regularizer=tf.keras.regularizers.l2(tuner_regularizer_6),
    #     )
    # )



    tuner_dropout_2 = tuner.Float(name='dropout 2', min_value=0.05, max_value=0.5, step=0.05)
    model.add(tf.keras.layers.Dropout(tuner_dropout_2))

    model.add(tf.keras.layers.Dense(classes, activation="softmax"))

    tuner_initial_lr = tuner.Float(name='learning_rate', min_value=0.001, max_value=0.025, step=0.001)
    tuner_decay_steps = tuner.Int(name='decay_steps', min_value=20, max_value=600, step=20)
    tuner_decay_rate = tuner.Float(name='decay_rate', min_value=0.1, max_value=2.0, step=0.1)
    lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(
        initial_learning_rate=tuner_initial_lr,
        decay_steps=tuner_decay_steps,
        decay_rate=tuner_decay_rate,
        staircase=False
    )
    model.compile(
        loss='binary_crossentropy',
        optimizer=tf.keras.optimizers.Adam(lr_schedule),
        metrics=['accuracy']
    )
    return model

def tune_dnn2(
    tuner,
    row=602,
    columns=4,
    classes=2,
    ):
    model = tf.keras.Sequential()
    input_shape = (row, columns)
    model.add(tf.keras.layers.InputLayer(input_shape=input_shape))
    model.add(tf.keras.layers.Flatten())

    tuner_units_1 = tuner.Int(name='units 1', min_value=32, max_value=704, step=32)
    tuner_regularizer_1 =  tuner.Choice(name='regularizer 1', values=[0.001, 0.0025, 0.005, 0.01, 0.025, 0.05])
    model.add(tf.keras.layers.Dense(
        units=tuner_units_1,
        activation='relu',
        kernel_regularizer=tf.keras.regularizers.l2(tuner_regularizer_1),
        )
    )

    tuner_units_2 = tuner.Int(name='units 2', min_value=32, max_value=704, step=32)
    tuner_regularizer_2 =  tuner.Choice(name='regularizer 2', values=[0.001, 0.0025, 0.005, 0.01, 0.025, 0.05])
    model.add(tf.keras.layers.Dense(
        units=tuner_units_2,
        activation='relu',
        kernel_regularizer=tf.keras.regularizers.l2(tuner_regularizer_2),
        )
    )

    tuner_units_3 = tuner.Int(name='units 3', min_value=32, max_value=704, step=32)
    tuner_regularizer_3 = tuner.Float(name='regularizer 3', min_value=0.001, max_value=0.05, step=0.001)
    model.add(tf.keras.layers.Dense(
        units=tuner_units_3,
        activation='relu',
        kernel_regularizer=tf.keras.regularizers.l2(tuner_regularizer_3),
        )
    )

    tuner_dropout_1 = tuner.Float(name='dropout 1', min_value=0.05, max_value=0.5, step=0.05)
    model.add(tf.keras.layers.Dropout(tuner_dropout_1))

    model.add(tf.keras.layers.Dense(classes, activation="softmax"))

    tuner_initial_lr = tuner.Float(name='learning_rate', min_value=0.001, max_value=0.025, step=0.001)
    tuner_decay_steps = tuner.Int(name='decay_steps', min_value=20, max_value=600, step=20)
    tuner_decay_rate = tuner.Float(name='decay_rate', min_value=0.1, max_value=2.0, step=0.1)
    lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(
        initial_learning_rate=tuner_initial_lr,
        decay_steps=tuner_decay_steps,
        decay_rate=tuner_decay_rate,
        staircase=False
    )
    model.compile(
        loss='binary_crossentropy',
        optimizer=tf.keras.optimizers.RMSprop(lr_schedule),
        metrics=['accuracy']
    )
    return model

def tune_dnn3(
    tuner,
    row=602,
    columns=4,
    classes=2,
    ):
    model = tf.keras.Sequential()
    input_shape = (row, columns)
    model.add(tf.keras.layers.InputLayer(input_shape=input_shape))
    model.add(tf.keras.layers.Flatten())

    tuner_units_1 = tuner.Int(name='units 1', min_value=32, max_value=704, step=32)
    tuner_regularizer_1 =  tuner.Choice(name='regularizer 1', values=[0.001, 0.0025, 0.005, 0.01, 0.025, 0.05])
    model.add(tf.keras.layers.Dense(
        units=tuner_units_1,
        activation='relu',
        kernel_regularizer=tf.keras.regularizers.l2(tuner_regularizer_1),
        )
    )

    tuner_units_2 = tuner.Int(name='units 2', min_value=32, max_value=704, step=32)
    tuner_regularizer_2 =  tuner.Choice(name='regularizer 2', values=[0.001, 0.0025, 0.005, 0.01, 0.025, 0.05])
    model.add(tf.keras.layers.Dense(
        units=tuner_units_2,
        activation='relu',
        kernel_regularizer=tf.keras.regularizers.l2(tuner_regularizer_2),
        )
    )

    tuner_units_3 = tuner.Int(name='units 3', min_value=32, max_value=704, step=32)
    tuner_regularizer_3 = tuner.Float(name='regularizer 3', min_value=0.001, max_value=0.05, step=0.001)
    model.add(tf.keras.layers.Dense(
        units=tuner_units_3,
        activation='relu',
        kernel_regularizer=tf.keras.regularizers.l2(tuner_regularizer_3),
        )
    )

    tuner_units_4 = tuner.Int(name='units 4', min_value=32, max_value=704, step=32)
    #tuner_regularizer_4 = tuner.Float(name='regularizer 4', min_value=0.001, max_value=0.05, step=0.001)
    model.add(tf.keras.layers.Dense(
        units=tuner_units_4,
        activation='relu',
        #kernel_regularizer=tf.keras.regularizers.l2(tuner_regularizer_3),
        )
    )

    tuner_dropout_1 = tuner.Float(name='dropout 1', min_value=0.05, max_value=0.5, step=0.05)
    model.add(tf.keras.layers.Dropout(tuner_dropout_1))

    model.add(tf.keras.layers.Dense(classes, activation="softmax"))

    tuner_initial_lr = tuner.Float(name='learning_rate', min_value=0.001, max_value=0.025, step=0.001)
    tuner_decay_steps = tuner.Int(name='decay_steps', min_value=20, max_value=600, step=20)
    tuner_decay_rate = tuner.Float(name='decay_rate', min_value=0.1, max_value=2.0, step=0.1)
    lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(
        initial_learning_rate=tuner_initial_lr,
        decay_steps=tuner_decay_steps,
        decay_rate=tuner_decay_rate,
        staircase=False
    )
    model.compile(
        loss='binary_crossentropy',
        optimizer=tf.keras.optimizers.RMSprop(lr_schedule),
        metrics=['accuracy']
    )
    return model

def tune_rnn1(
    tuner,
    row=602,
    columns=4,
    classes=2,
    ):
    model = tf.keras.Sequential()
    input_shape = (row, columns)
    model.add(tf.keras.layers.InputLayer(input_shape=input_shape))

    tuner_units_1 = tuner.Int(name='units 1', min_value=32, max_value=704, step=32)
    model.add(tf.keras.layers.LSTM(
        units=tuner_units_1,
        activation='tanh',
        return_sequences=True,
        )
    )



    tuner_units_2 = tuner.Int(name='units 2', min_value=32, max_value=704, step=32)
    model.add(tf.keras.layers.LSTM(
        units=tuner_units_2,
        activation='tanh',
        )
    )

    tuner_dropout_1 = tuner.Float(name='dropout 1', min_value=0.05, max_value=0.5, step=0.05)
    model.add(tf.keras.layers.Dropout(rate=tuner_dropout_1))
    # tuner_dropout_2 = tuner.Choice(name='dropout 2', values=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
    # model.add(tf.keras.layers.Dropout(rate=tuner_dropout_2))

    tuner_units_3 = tuner.Int(name='units 3', min_value=32, max_value=704, step=32)
    #tuner_regularizer_1 = tuner.Choice(name='regularizer 1', values=[0.001, 0.0025, 0.005, 0.01, 0.025, 0.05])
    model.add(tf.keras.layers.Dense(
        tuner_units_3,
        activation="relu",
        #kernel_regularizer=tf.keras.regularizers.l2(tuner_regularizer_1)
        )
    )
    model.add(tf.keras.layers.Dense(classes, activation="softmax"))

    tuner_initial_lr = tuner.Float(name='learning_rate', min_value=0.001, max_value=0.025, step=0.001)
    tuner_decay_steps = tuner.Int(name='decay_steps', min_value=20, max_value=600, step=20)
    tuner_decay_rate = tuner.Float(name='decay_rate', min_value=0.1, max_value=2.0, step=0.1)
    lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(
        initial_learning_rate=tuner_initial_lr,
        decay_steps=tuner_decay_steps,
        decay_rate=tuner_decay_rate,
        staircase=False
    )
    model.compile(
        loss='binary_crossentropy',
        optimizer=tf.keras.optimizers.Adam(lr_schedule),
        metrics=['accuracy']
    )
    return model

def tune_rnn2(
    tuner,
    row=602,
    columns=4,
    classes=2,
    ):
    model = tf.keras.Sequential()
    input_shape = (row, columns)
    model.add(tf.keras.layers.InputLayer(input_shape=input_shape))

    tuner_units_1 = tuner.Int(name='units 1', min_value=32, max_value=704, step=32)
    model.add(tf.keras.layers.LSTM(
        units=tuner_units_1,
        activation='tanh',
        return_sequences=True,
        )
    )

    tuner_units_2 = tuner.Int(name='units 2', min_value=32, max_value=704, step=32)
    model.add(tf.keras.layers.LSTM(
        units=tuner_units_2,
        activation='tanh',
        return_sequences=True,
        )
    )

    tuner_units_3 = tuner.Int(name='units 3', min_value=32, max_value=704, step=32)
    model.add(tf.keras.layers.LSTM(
        units=tuner_units_3,
        activation='tanh',
        )
    )
    tuner_dropout_1 = tuner.Float(name='dropout 1', min_value=0.05, max_value=0.5, step=0.05)
    model.add(tf.keras.layers.Dropout(rate=tuner_dropout_1))

    model.add(tf.keras.layers.Dense(classes, activation="softmax"))

    tuner_initial_lr = tuner.Float(name='learning_rate', min_value=0.001, max_value=0.025, step=0.0005)
    tuner_decay_steps = tuner.Int(name='decay_steps', min_value=20, max_value=600, step=20)
    tuner_decay_rate = tuner.Float(name='decay_rate', min_value=0.05, max_value=2.0, step=0.05)
    lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(
        initial_learning_rate=tuner_initial_lr,
        decay_steps=tuner_decay_steps,
        decay_rate=tuner_decay_rate,
        staircase=False
    )
    model.compile(
        loss='binary_crossentropy',
        optimizer=tf.keras.optimizers.Adam(lr_schedule),
        metrics=['accuracy']
    )
    return model

def tune_rnn3(
    tuner,
    row=602,
    columns=4,
    classes=2,
    ):
    model = tf.keras.Sequential()
    input_shape = (row, columns)
    model.add(tf.keras.layers.InputLayer(input_shape=input_shape))

    tuner_units_1 = tuner.Int(name='units 1', min_value=32, max_value=704, step=32)
    model.add(tf.keras.layers.LSTM(
        units=tuner_units_1,
        activation='tanh',
        return_sequences=True,
        )
    )

    tuner_units_2 = tuner.Int(name='units 2', min_value=32, max_value=704, step=32)
    tuner_regularizer_1 = tuner.Choice(name='regularizer 1', values=[0.001, 0.0025, 0.005, 0.01, 0.025, 0.05])
    model.add(tf.keras.layers.Dense(
        tuner_units_2,
        activation="relu",
        kernel_regularizer=tf.keras.regularizers.l2(tuner_regularizer_1)
        )
    )

    tuner_units_3 = tuner.Int(name='units 3', min_value=32, max_value=704, step=32)
    model.add(tf.keras.layers.LSTM(
        units=tuner_units_3,
        activation='tanh',
        )
    )

    tuner_units_4 = tuner.Int(name='units 4', min_value=32, max_value=704, step=32)
    tuner_regularizer_2 = tuner.Choice(name='regularizer 2', values=[0.001, 0.0025, 0.005, 0.01, 0.025, 0.05])
    model.add(tf.keras.layers.Dense(
        tuner_units_4,
        activation="relu",
        kernel_regularizer=tf.keras.regularizers.l2(tuner_regularizer_2)
        )
    )

    tuner_dropout_1 = tuner.Float(name='dropout 1', min_value=0.05, max_value=0.5, step=0.05)
    model.add(tf.keras.layers.Dropout(rate=tuner_dropout_1))

    model.add(tf.keras.layers.Dense(classes, activation="softmax"))

    tuner_initial_lr = tuner.Float(name='learning_rate', min_value=0.001, max_value=0.025, step=0.0005)
    tuner_decay_steps = tuner.Int(name='decay_steps', min_value=20, max_value=600, step=20)
    tuner_decay_rate = tuner.Float(name='decay_rate', min_value=0.05, max_value=2.0, step=0.05)
    lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(
        initial_learning_rate=tuner_initial_lr,
        decay_steps=tuner_decay_steps,
        decay_rate=tuner_decay_rate,
        staircase=False
    )
    model.compile(
        loss='binary_crossentropy',
        optimizer=tf.keras.optimizers.Adam(lr_schedule),
        metrics=['accuracy']
    )
    return model

def run(**kwargs):


    conv2d_dict = encode_data.encode(kwargs['datasets'][0], 'cnn1', kwargs)
    other_dict = encode_data.encode(kwargs['datasets'][0], 'cnn2', kwargs)
    models_tuned = [model[0] for model in kwargs['models']]
    report_path = '../Visuals/report.md'

    for ss_type in kwargs['splice_sites']:

        conv2d_X_trn, y_trn = conv2d_dict[ss_type]
        X_trn, y_trn = other_dict[ss_type]

        if 'cnn1' in models_tuned:
            # cnn1 tuning
            tuner01 = kt.Hyperband(
                tune_cnn1,
                objective='val_accuracy',
                max_epochs=15,
                factor=3,
                seed=kwargs['state'],
                directory='TuningLogs',
                project_name='cnn1_tuning'
            )
            tuner01.search(conv2d_X_trn, y_trn, epochs=125, validation_split=0.2, callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, min_delta=0.001)])
            tuner01_best_hps = tuner01.get_best_hyperparameters(num_trials=1)[0]
            model01 = tuner01.hypermodel.build(tuner01_best_hps)
            history01 = model01.fit(conv2d_X_trn, y_trn, epochs=50, validation_split=0.2, callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, min_delta=0.001)])
            val_acc_per_epoch_01 = history01.history['val_accuracy']
            tuner01_best_epoch = val_acc_per_epoch_01.index(max(val_acc_per_epoch_01)) + 1
            tuner01_best_val_acc = max(val_acc_per_epoch_01)

            #REMOVED
            # 1st layer kern init: {tuner01_best_hps.get('kern init 1')}
            # 1st layer bias init: {tuner01_best_hps.get('bias init 1')}
            # 1st layer act reg: {tuner01_best_hps.get('act reg 1')}
            # 1st layer kern reg: {tuner01_best_hps.get('kern reg 1')}
            # 1st layer bias reg: {tuner01_best_hps.get('bias reg 1')}
            # 2nd layer kern init: {tuner01_best_hps.get('kern init 2')}
            # 2nd layer bias init: {tuner01_best_hps.get('bias init 2')}
            # 2nd layer act reg: {tuner01_best_hps.get('act reg 2')}
            # 2nd layer kern reg: {tuner01_best_hps.get('kern reg 2')}
            # 2nd layer bias reg: {tuner01_best_hps.get('bias reg 2')}
            # 3rd layer kern init: {tuner01_best_hps.get('kern init 3')}
            # 3rd layer bias init: {tuner01_best_hps.get('bias init 3')}
            # 3rd layer act reg: {tuner01_best_hps.get('act reg 3')}
            # 3rd layer kern reg: {tuner01_best_hps.get('kern reg 3')}
            # 3rd layer bias reg: {tuner01_best_hps.get('bias reg 3')}
            #1st layer activation: {tuner01_best_hps.get('act 1')}
            #2nd layer activation: {tuner01_best_hps.get('act 2')}
            #3rd layer activation: {tuner01_best_hps.get('act 3')}
            #optimizer: {tuner01_best_hps.get('optimizer')}
            #CNN activation: {tuner01_best_hps.get('act')}
            #1st layer stride: ({tuner01_best_hps.get('stride 1')},{tuner01_best_hps.get('stride 1')})
            #2nd layer stride: ({tuner01_best_hps.get('stride 2')},{tuner01_best_hps.get('stride 2')})
            #gloMax pool 1: ({tuner01_best_hps.get('pooling 1')}, {tuner01_best_hps.get('pooling 1')})
            #gloMax pool stride 1: {tuner01_best_hps.get('pool stride 1')}
            #gloMax pool 2: ({tuner01_best_hps.get('pooling 2')}, {tuner01_best_hps.get('pooling 2')})
            #gloMax pool stride 2: {tuner01_best_hps.get('pool stride 2')}
            #3rd layer stride: ({tuner01_best_hps.get('stride 3')},{tuner01_best_hps.get('stride 3')})

            # cnn1
            print(f"""
        The hyperparameter search is complete for CNN01 on {ss_type.upper()} data.
        1st layer filters: {tuner01_best_hps.get('filters 1')}
        1st layer kernal: ({tuner01_best_hps.get('kernal 1')},{tuner01_best_hps.get('kernal 1')})
        2nd layer filters: {tuner01_best_hps.get('filters 2')}
        2nd layer kernal: ({tuner01_best_hps.get('kernal 2')},{tuner01_best_hps.get('kernal 2')})
        3rd layer filters: {tuner01_best_hps.get('filters 3')}
        3rd layer kernal: ({tuner01_best_hps.get('kernal 3')},{tuner01_best_hps.get('kernal 3')})
        drop 1: {tuner01_best_hps.get('dropout 1')}
        learning rate: {tuner01_best_hps.get('learning_rate')}
        decay steps: {tuner01_best_hps.get('decay_steps')}
        decay rate: {tuner01_best_hps.get('decay_rate')}
        The best number of epochs was {tuner01_best_epoch},
        which produced a validation accuracy of {tuner01_best_val_acc}.
        """)
            # if kwargs['report']:
            #     with open()

        if 'cnn2' in models_tuned:
            # cnn2 tuning
            tuner02 = kt.Hyperband(
                tune_cnn2,
                objective='val_accuracy',
                max_epochs=15,
                factor=3,
                seed=kwargs['state'],
                directory='TuningLogs',
                project_name='cnn2_tuning'
            )
            tuner02.search(X_trn, y_trn, epochs=125, validation_split=0.2, callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, min_delta=0.001)])
            tuner02_best_hps = tuner02.get_best_hyperparameters(num_trials=1)[0]
            model02 = tuner02.hypermodel.build(tuner02_best_hps)
            history02 = model02.fit(X_trn, y_trn, epochs=50, validation_split=0.2, callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, min_delta=0.001)])
            val_acc_per_epoch_02 = history02.history['val_accuracy']
            tuner02_best_epoch = val_acc_per_epoch_02.index(max(val_acc_per_epoch_02)) + 1
            tuner02_best_val_acc = max(val_acc_per_epoch_02)

            #    max pool 2: {tuner02_best_hps.get('pooling 2')}

            # cnn2
            print(f"""
        The hyperparameter search is complete for CNN02 on {ss_type.upper()} data.
        1st layer filters: {tuner02_best_hps.get('filters 1')}
        1st layer kernal: {tuner02_best_hps.get('kernal 1')}
        2nd layer filters: {tuner02_best_hps.get('filters 2')}
        2nd layer kernal: {tuner02_best_hps.get('kernal 2')}
        max pool 1: {tuner02_best_hps.get('pooling 1')}
        3rd layer filters: {tuner02_best_hps.get('filters 3')}
        3rd layer kernal: {tuner02_best_hps.get('kernal 3')}
        drop 1: {tuner02_best_hps.get('dropout 1')}
        learning rate: {tuner02_best_hps.get('learning_rate')}
        decay steps: {tuner02_best_hps.get('decay_steps')}
        decay rate: {tuner02_best_hps.get('decay_rate')}
        The best number of epochs was {tuner02_best_epoch},
        which produced a validation accuracy of {tuner02_best_val_acc}.
        """)

        if 'cnn3' in models_tuned:
            # cnn3 tuning
            tuner03 = kt.Hyperband(
                tune_cnn3,
                objective='val_accuracy',
                max_epochs=15,
                factor=3,
                seed=kwargs['state'],
                directory='TuningLogs',
                project_name='cnn3_tuning'
            )
            tuner03.search(X_trn, y_trn, epochs=125, validation_split=0.2, callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, min_delta=0.001)])
            tuner03_best_hps = tuner03.get_best_hyperparameters(num_trials=1)[0]
            model03 = tuner03.hypermodel.build(tuner03_best_hps)
            history03 = model03.fit(X_trn, y_trn, epochs=50, validation_split=0.2, callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, min_delta=0.001)])
            val_acc_per_epoch_03 = history03.history['val_accuracy']
            tuner03_best_epoch = val_acc_per_epoch_03.index(max(val_acc_per_epoch_03)) + 1
            tuner03_best_val_acc = max(val_acc_per_epoch_03)


        #max pool 2: {tuner03_best_hps.get('pooling 2')}
        #max pool 1: {tuner03_best_hps.get('pooling 1')}

            # cnn3
            print(f"""
        The hyperparameter search is complete for CNN03 on {ss_type.upper()} data.
        1st layer filters: {tuner03_best_hps.get('filters 1')}
        1st layer kernal: {tuner03_best_hps.get('kernal 1')}
        2nd layer filters: {tuner03_best_hps.get('filters 2')}
        2nd layer kernal: {tuner03_best_hps.get('kernal 2')}
        drop 1: {tuner03_best_hps.get('dropout 1')}
        learning rate: {tuner03_best_hps.get('learning_rate')}
        decay steps: {tuner03_best_hps.get('decay_steps')}
        decay rate: {tuner03_best_hps.get('decay_rate')}
        The best number of epochs was {tuner03_best_epoch},
        which produced a validation accuracy of {tuner03_best_val_acc}.
        """)

        if 'cnn4' in models_tuned:
            # cnn4 tuning
            tuner04 = kt.Hyperband(
                tune_cnn4,
                objective='val_accuracy',
                max_epochs=15,
                factor=3,
                seed=kwargs['state'],
                directory='TuningLogs',
                project_name='cnn4_tuning'
            )
            tuner04.search(X_trn, y_trn, epochs=200, validation_split=0.2, callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, min_delta=0.001)])
            tuner04_best_hps = tuner04.get_best_hyperparameters(num_trials=1)[0]
            model04 = tuner04.hypermodel.build(tuner04_best_hps)
            history04 = model04.fit(X_trn, y_trn, epochs=50, validation_split=0.2, callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, min_delta=0.001)])
            val_acc_per_epoch_04 = history04.history['val_accuracy']
            tuner04_best_epoch = val_acc_per_epoch_04.index(max(val_acc_per_epoch_04)) + 1
            tuner04_best_val_acc = max(val_acc_per_epoch_04)

            # cnn4
            print(f"""
        The hyperparameter search is complete for CNN04.
        1st layer filters: {tuner04_best_hps.get('filters 1')}
        1st layer kernal: {tuner04_best_hps.get('kernal 1')}
        max pool 1: {tuner04_best_hps.get('pooling 1')}
        drop 1: {tuner04_best_hps.get('dropout 1')}
        learning rate: {tuner04_best_hps.get('learning_rate')}
        decay steps: {tuner04_best_hps.get('decay_steps')}
        decay rate: {tuner04_best_hps.get('decay_rate')}
        The best number of epochs was {tuner04_best_epoch},
        which produced a validation accuracy of {tuner04_best_val_acc}.
        """)

        if 'dnn1' in models_tuned:
            # dnn1 tuning
            tuner05 = kt.Hyperband(
                tune_dnn1,
                objective='val_accuracy',
                max_epochs=15,
                factor=3,
                seed=kwargs['state'],
                directory='TuningLogs',
                project_name='dnn1_tuning'
            )
            tuner05.search(
                X_trn,
                y_trn,
                epochs=200,
                validation_split=0.2,
                callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, min_delta=0.001)]
            )
            tuner05_best_hps = tuner05.get_best_hyperparameters(num_trials=1)[0]
            model05 = tuner05.hypermodel.build(tuner05_best_hps)
            history05 = model05.fit(X_trn, y_trn, epochs=50, validation_split=0.2, callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, min_delta=0.001)])
            val_acc_per_epoch_05 = history05.history['val_accuracy']
            tuner05_best_epoch = val_acc_per_epoch_05.index(max(val_acc_per_epoch_05)) + 1
            tuner05_best_val_acc = max(val_acc_per_epoch_05)

            #
            # 1st layer reg.: {tuner05_best_hps.get('regularizer 1')}
            # 2nd reg.: {tuner05_best_hps.get('regularizer 2')}
            # 3rd reg.: {tuner05_best_hps.get('regularizer 3')}
            #drop 2: {tuner05_best_hps.get('dropout 2')}
            #        drop 2: {tuner05_best_hps.get('dropout 2')}
            # 4th layer units: {tuner05_best_hps.get('units 4')}
            # 4th reg.: {tuner05_best_hps.get('regularizer 4')}
            # 5th layer units: {tuner05_best_hps.get('units 5')}
            # 5th reg.: {tuner05_best_hps.get('regularizer 5')}
            # 6th layer units: {tuner05_best_hps.get('units 6')}
            # 6th reg.: {tuner05_best_hps.get('regularizer 6')}
            # 1st layer reg.: {tuner05_best_hps.get('regularizer 1')}
            #         3rd reg.: {tuner05_best_hps.get('regularizer 3')}
            #2nd reg.: {tuner05_best_hps.get('regularizer 2')}
            # dnn1
            print(f"""
        The hyperparameter search is complete for DNN01 on {ss_type.upper()} data.
        1st layer units: {tuner05_best_hps.get('units 1')}
        1st layer reg.: {tuner05_best_hps.get('regularizer 1')}
        2nd layer units: {tuner05_best_hps.get('units 2')}
        2nd reg.: {tuner05_best_hps.get('regularizer 2')}
        drop 1: {tuner05_best_hps.get('dropout 1')}
        3rd layer units: {tuner05_best_hps.get('units 3')}
        drop 2: {tuner05_best_hps.get('dropout 2')}
        learning rate: {tuner05_best_hps.get('learning_rate')}
        decay steps: {tuner05_best_hps.get('decay_steps')}
        decay rate: {tuner05_best_hps.get('decay_rate')}
        The best number of epochs was {tuner05_best_epoch},
        which produced a validation accuracy of {tuner05_best_val_acc}.
        """)

        if 'dnn2' in models_tuned:
            # dnn2 tuning
            tuner06 = kt.Hyperband(
                tune_dnn2,
                objective='val_accuracy',
                max_epochs=15,
                factor=3,
                seed=kwargs['state'],
                directory='TuningLogs',
                project_name='dnn2_tuning'
            )
            tuner06.search(X_trn, y_trn, epochs=200, validation_split=0.2, callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, min_delta=0.001)])
            tuner06_best_hps = tuner06.get_best_hyperparameters(num_trials=1)[0]
            model06 = tuner06.hypermodel.build(tuner06_best_hps)
            history06 = model06.fit(X_trn, y_trn, epochs=50, validation_split=0.2, callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, min_delta=0.001)])
            val_acc_per_epoch_06 = history06.history['val_accuracy']
            tuner06_best_epoch = val_acc_per_epoch_06.index(max(val_acc_per_epoch_06)) + 1
            tuner06_best_val_acc = max(val_acc_per_epoch_06)

            # dnn2
            print(f"""
        The hyperparameter search is complete for DNN02 on {ss_type.upper()} data.
        1st layer units: {tuner06_best_hps.get('units 1')}
        1st layer reg.: {tuner06_best_hps.get('regularizer 1')}
        2nd layer units: {tuner06_best_hps.get('units 2')}
        2nd reg.: {tuner06_best_hps.get('regularizer 2')}
        3rd layer units: {tuner06_best_hps.get('units 3')}
        3rd reg.: {tuner06_best_hps.get('regularizer 3')}
        drop 1: {tuner06_best_hps.get('dropout 1')}
        learning rate: {tuner06_best_hps.get('learning_rate')}
        decay steps: {tuner06_best_hps.get('decay_steps')}
        decay rate: {tuner06_best_hps.get('decay_rate')}
        The best number of epochs was {tuner06_best_epoch},
        which produced a validation accuracy of {tuner06_best_val_acc}.
        """)

        if 'dnn3' in models_tuned:
            # dnn3 tuning
            tuner07 = kt.Hyperband(
                tune_dnn3,
                objective='val_accuracy',
                max_epochs=15,
                factor=3,
                seed=kwargs['state'],
                directory='TuningLogs',
                project_name='dnn3_tuning'
            )
            tuner07.search(X_trn, y_trn, epochs=200, validation_split=0.2, callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, min_delta=0.001)])
            tuner07_best_hps = tuner07.get_best_hyperparameters(num_trials=1)[0]
            model07 = tuner07.hypermodel.build(tuner07_best_hps)
            history07 = model07.fit(X_trn, y_trn, epochs=50, validation_split=0.2, callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, min_delta=0.001)])
            val_acc_per_epoch_07 = history07.history['val_accuracy']
            tuner07_best_epoch = val_acc_per_epoch_07.index(max(val_acc_per_epoch_07)) + 1
            tuner07_best_val_acc = max(val_acc_per_epoch_07)

            # dnn3
            print(f"""
        The hyperparameter search is complete for DNN03 on {ss_type.upper()} data.
        1st layer units: {tuner07_best_hps.get('units 1')}
        1st layer reg.: {tuner07_best_hps.get('regularizer 1')}
        2nd layer units: {tuner07_best_hps.get('units 2')}
        2nd layer reg.: {tuner07_best_hps.get('regularizer 2')}
        3rd layer units: {tuner07_best_hps.get('units 3')}
        3rd layer reg.: {tuner07_best_hps.get('regularizer 3')}
        4th layer units: {tuner07_best_hps.get('units 4')}
        drop 1: {tuner07_best_hps.get('dropout 1')}
        learning rate: {tuner07_best_hps.get('learning_rate')}
        decay steps: {tuner07_best_hps.get('decay_steps')}
        decay rate: {tuner07_best_hps.get('decay_rate')}
        The best number of epochs was {tuner07_best_epoch},
        which produced a validation accuracy of {tuner07_best_val_acc}.
        """)

        if 'cnn5' in models_tuned:
            # cnn5 tuning
            tuner08 = kt.Hyperband(
                tune_cnn5,
                objective='val_accuracy',
                max_epochs=15,
                factor=3,
                seed=kwargs['state'],
                directory='TuningLogs',
                project_name='cnn5_tuning'
            )
            tuner08.search(conv2d_X_trn, y_trn, epochs=100, validation_split=0.2, callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)])
            tuner08_best_hps = tuner08.get_best_hyperparameters(num_trials=1)[0]
            model08 = tuner08.hypermodel.build(tuner08_best_hps)
            history08 = model08.fit(conv2d_X_trn, y_trn, epochs=45, validation_split=0.2, callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)])
            val_acc_per_epoch_08 = history08.history['val_accuracy']
            tuner08_best_epoch = val_acc_per_epoch_08.index(max(val_acc_per_epoch_08)) + 1
            tuner08_best_val_acc = max(val_acc_per_epoch_08)

            # cnn5
            print(f"""
        The hyperparameter search is complete for CNN015
        1st layer filters: {tuner08_best_hps.get('filters 1')}
        1st layer kernal: ({tuner08_best_hps.get('kernal 1')},{tuner08_best_hps.get('kernal 1')})
        2nd layer filters: {tuner08_best_hps.get('filters 2')}
        2nd layer kernal: ({tuner08_best_hps.get('kernal 2')},{tuner08_best_hps.get('kernal 2')})
        3rd layer filters: {tuner08_best_hps.get('filters 3')}
        3rd layer kernal: ({tuner08_best_hps.get('kernal 3')}
        max pool 1: ({tuner08_best_hps.get('pooling 1')}, {tuner08_best_hps.get('pooling 1')})
        4th layer filters: {tuner08_best_hps.get('filters 4')}
        4th layer kernal: ({tuner08_best_hps.get('kernal 4')}
        drop 1: {tuner08_best_hps.get('dropout 1')}
        learning rate: {tuner08_best_hps.get('learning_rate')}
        decay steps: {tuner08_best_hps.get('decay_steps')}
        decay rate: {tuner08_best_hps.get('decay_rate')}
        The best number of epochs was {tuner08_best_epoch},
        which produced a validation accuracy of {tuner08_best_val_acc}.
        """)

        if 'rnn1' in models_tuned:
            # rnn1 tuning
            tuner09 = kt.Hyperband(
                tune_rnn1,
                objective='val_accuracy',
                max_epochs=15,
                factor=3,
                seed=kwargs['state'],
                directory='TuningLogs',
                project_name='rnn1_tuning'
            )
            tuner09.search(X_trn, y_trn, epochs=125, validation_split=0.2, callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, min_delta=0.001)])
            tuner09_best_hps = tuner09.get_best_hyperparameters(num_trials=1)[0]
            model09 = tuner09.hypermodel.build(tuner09_best_hps)
            history09 = model09.fit(X_trn, y_trn, epochs=50, validation_split=0.2, callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, min_delta=0.001)])
            val_acc_per_epoch_09 = history09.history['val_accuracy']
            tuner09_best_epoch = val_acc_per_epoch_09.index(max(val_acc_per_epoch_09)) + 1
            tuner09_best_val_acc = max(val_acc_per_epoch_09)
            #drop 2: {tuner09_best_hps.get('dropout 2')}
            #3rd layer reg.: {tuner09_best_hps.get('regularizer 1')}
            # rnn1
            print(f"""
        The hyperparameter search is complete for RNN01.
        1st layer units: {tuner09_best_hps.get('units 1')}
        2nd layer units: {tuner09_best_hps.get('units 2')}
        drop 1: {tuner09_best_hps.get('dropout 1')}
        3rd layer units: {tuner09_best_hps.get('units 3')}
        learning rate: {tuner09_best_hps.get('learning_rate')}
        decay steps: {tuner09_best_hps.get('decay_steps')}
        decay rate: {tuner09_best_hps.get('decay_rate')}
        The best number of epochs was {tuner09_best_epoch},
        which produced a validation accuracy of {tuner09_best_val_acc}.
        """)

        if 'rnn2' in models_tuned:
            # rnn2 tuning
            tuner10 = kt.Hyperband(
                tune_rnn2,
                objective='val_accuracy',
                max_epochs=15,
                factor=3,
                seed=kwargs['state'],
                directory='TuningLogs',
                project_name='rnn2_tuning'
            )
            tuner10.search(X_trn, y_trn, epochs=100, validation_split=0.2, callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)])
            tuner10_best_hps = tuner10.get_best_hyperparameters(num_trials=1)[0]
            model10 = tuner10.hypermodel.build(tuner10_best_hps)
            history10 = model10.fit(X_trn, y_trn, epochs=45, validation_split=0.2, callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)])
            val_acc_per_epoch_10 = history10.history['val_accuracy']
            tuner10_best_epoch = val_acc_per_epoch_10.index(max(val_acc_per_epoch_10)) + 1
            tuner10_best_val_acc = max(val_acc_per_epoch_10)
            # rnn2
            print(f"""
        The hyperparameter search is complete for RNN02.
        1st layer units: {tuner10_best_hps.get('units 1')}
        2nd layer units: {tuner10_best_hps.get('units 2')}
        3rd layer units: {tuner10_best_hps.get('units 3')}
        drop 1: {tuner10_best_hps.get('dropout 1')}
        learning rate: {tuner10_best_hps.get('learning_rate')}
        decay steps: {tuner10_best_hps.get('decay_steps')}
        decay rate: {tuner10_best_hps.get('decay_rate')}
        The best number of epochs was {tuner10_best_epoch},
        which produced a validation accuracy of {tuner10_best_val_acc}.
        """)

        if 'rnn3' in models_tuned:
            # rnn3 tuning
            tuner11 = kt.Hyperband(
                tune_rnn3,
                objective='val_accuracy',
                max_epochs=15,
                factor=3,
                seed=kwargs['state'],
                directory='TuningLogs',
                project_name='rnn3_tuning'
            )
            tuner11.search(X_trn, y_trn, epochs=100, validation_split=0.2, callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)])
            tuner11_best_hps = tuner11.get_best_hyperparameters(num_trials=1)[0]
            model11 = tuner11.hypermodel.build(tuner11_best_hps)
            history11 = model11.fit(X_trn, y_trn, epochs=45, validation_split=0.2, callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)])
            val_acc_per_epoch_11 = history11.history['val_accuracy']
            tuner11_best_epoch = val_acc_per_epoch_11.index(max(val_acc_per_epoch_11)) + 1
            tuner11_best_val_acc = max(val_acc_per_epoch_11)

            # rnn3
            print(f"""
        The hyperparameter search is complete for RNN03.
        1st layer units: {tuner11_best_hps.get('units 1')}
        2nd layer units: {tuner11_best_hps.get('units 2')}
        2nd layer reg.: {tuner11_best_hps.get('regularizer 1')}
        3rd layer units: {tuner11_best_hps.get('units 3')}
        4th layer units: {tuner11_best_hps.get('units 4')}
        4th layer reg.: {tuner11_best_hps.get('regularizer 2')}
        drop 1: {tuner11_best_hps.get('dropout 1')}
        learning rate: {tuner11_best_hps.get('learning_rate')}
        decay steps: {tuner11_best_hps.get('decay_steps')}
        decay rate: {tuner11_best_hps.get('decay_rate')}
        The best number of epochs was {tuner11_best_epoch},
        which produced a validation accuracy of {tuner11_best_val_acc}.
        """)

    # print(tuner01.results_summary())
    # print(tuner02.results_summary())
    # print(tuner03.results_summary())
    # print(tuner04.results_summary())
    # print(tuner05.results_summary())
    # print(tuner06.results_summary())
    # print(tuner07.results_summary())
    # print(tuner08.results_summary())
