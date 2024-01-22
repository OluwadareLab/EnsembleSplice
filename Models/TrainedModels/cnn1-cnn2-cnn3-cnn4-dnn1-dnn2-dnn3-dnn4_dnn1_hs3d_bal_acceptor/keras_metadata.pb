
�@root"_tf_keras_sequential*�?{"name": "sequential_4", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "class_name": "Sequential", "config": {"name": "sequential_4", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 140, 4]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_5"}}, {"class_name": "Flatten", "config": {"name": "flatten_4", "trainable": true, "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "Dense", "config": {"name": "dense_4", "trainable": true, "dtype": "float32", "units": 704, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.02500000037252903}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_5", "trainable": true, "dtype": "float32", "units": 224, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.02500000037252903}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dropout", "config": {"name": "dropout_4", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}}, {"class_name": "Dense", "config": {"name": "dense_6", "trainable": true, "dtype": "float32", "units": 512, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dropout", "config": {"name": "dropout_5", "trainable": true, "dtype": "float32", "rate": 0.15, "noise_shape": null, "seed": null}}, {"class_name": "Dense", "config": {"name": "dense_7", "trainable": true, "dtype": "float32", "units": 2, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "shared_object_id": 18, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 140, 4]}, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 140, 4]}, "is_graph_network": true, "full_save_spec": {"class_name": "__tuple__", "items": [[{"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 140, 4]}, "float32", "input_5"]}], {}]}, "save_spec": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 140, 4]}, "float32", "input_5"]}, "keras_version": "2.8.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential_4", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 140, 4]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_5"}, "shared_object_id": 0}, {"class_name": "Flatten", "config": {"name": "flatten_4", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "shared_object_id": 1}, {"class_name": "Dense", "config": {"name": "dense_4", "trainable": true, "dtype": "float32", "units": 704, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 2}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 3}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.02500000037252903}, "shared_object_id": 4}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 5}, {"class_name": "Dense", "config": {"name": "dense_5", "trainable": true, "dtype": "float32", "units": 224, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 6}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 7}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.02500000037252903}, "shared_object_id": 8}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 9}, {"class_name": "Dropout", "config": {"name": "dropout_4", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}, "shared_object_id": 10}, {"class_name": "Dense", "config": {"name": "dense_6", "trainable": true, "dtype": "float32", "units": 512, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 11}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 12}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 13}, {"class_name": "Dropout", "config": {"name": "dropout_5", "trainable": true, "dtype": "float32", "rate": 0.15, "noise_shape": null, "seed": null}, "shared_object_id": 14}, {"class_name": "Dense", "config": {"name": "dense_7", "trainable": true, "dtype": "float32", "units": 2, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 15}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 16}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 17}]}}, "training_config": {"loss": "binary_crossentropy", "metrics": [[{"class_name": "BinaryAccuracy", "config": {"name": "bin_acc", "dtype": "float32", "threshold": 0.5}, "shared_object_id": 20}, {"class_name": "BinaryCrossentropy", "config": {"name": "bin_cross", "dtype": "float32", "from_logits": false, "label_smoothing": 0}, "shared_object_id": 21}, {"class_name": "AUC", "config": {"name": "auc", "dtype": "float32", "num_thresholds": 200, "curve": "ROC", "summation_method": "interpolation", "multi_label": false, "label_weights": null}, "shared_object_id": 22}, {"class_name": "Precision", "config": {"name": "pre", "dtype": "float32", "thresholds": null, "top_k": null, "class_id": null}, "shared_object_id": 23}, {"class_name": "Recall", "config": {"name": "rec", "dtype": "float32", "thresholds": null, "top_k": null, "class_id": null}, "shared_object_id": 24}, {"class_name": "TruePositives", "config": {"name": "tru_pos", "dtype": "float32", "thresholds": null}, "shared_object_id": 25}, {"class_name": "TrueNegatives", "config": {"name": "tru_neg", "dtype": "float32", "thresholds": null}, "shared_object_id": 26}, {"class_name": "FalsePositives", "config": {"name": "fal_pos", "dtype": "float32", "thresholds": null}, "shared_object_id": 27}, {"class_name": "FalseNegatives", "config": {"name": "fal_neg", "dtype": "float32", "thresholds": null}, "shared_object_id": 28}]], "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": {"class_name": "InverseTimeDecay", "config": {"initial_learning_rate": 0.002, "decay_steps": 80, "decay_rate": 1.4, "staircase": false, "name": null}, "shared_object_id": 29}, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}2
�root.layer-0"_tf_keras_layer*�{"name": "flatten_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Flatten", "config": {"name": "flatten_4", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "shared_object_id": 1, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}, "shared_object_id": 30}}2
�root.layer_with_weights-0"_tf_keras_layer*�{"name": "dense_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense_4", "trainable": true, "dtype": "float32", "units": 704, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 2}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 3}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.02500000037252903}, "shared_object_id": 4}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 5, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 560}}, "shared_object_id": 31}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 560]}}2
�root.layer_with_weights-1"_tf_keras_layer*�{"name": "dense_5", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense_5", "trainable": true, "dtype": "float32", "units": 224, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 6}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 7}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.02500000037252903}, "shared_object_id": 8}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 9, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 704}}, "shared_object_id": 32}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 704]}}2
�root.layer-3"_tf_keras_layer*�{"name": "dropout_4", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dropout", "config": {"name": "dropout_4", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}, "shared_object_id": 10, "build_input_shape": {"class_name": "TensorShape", "items": [null, 224]}}2
�root.layer_with_weights-2"_tf_keras_layer*�{"name": "dense_6", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense_6", "trainable": true, "dtype": "float32", "units": 512, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 11}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 12}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 13, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 224}}, "shared_object_id": 33}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 224]}}2
�root.layer-5"_tf_keras_layer*�{"name": "dropout_5", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dropout", "config": {"name": "dropout_5", "trainable": true, "dtype": "float32", "rate": 0.15, "noise_shape": null, "seed": null}, "shared_object_id": 14, "build_input_shape": {"class_name": "TensorShape", "items": [null, 512]}}2
�root.layer_with_weights-3"_tf_keras_layer*�{"name": "dense_7", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense_7", "trainable": true, "dtype": "float32", "units": 2, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 15}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 16}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 17, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 512}}, "shared_object_id": 34}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 512]}}2
�troot.keras_api.metrics.0"_tf_keras_metric*�{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}, "shared_object_id": 35}2
�uroot.keras_api.metrics.1"_tf_keras_metric*�{"class_name": "BinaryAccuracy", "name": "bin_acc", "dtype": "float32", "config": {"name": "bin_acc", "dtype": "float32", "threshold": 0.5}, "shared_object_id": 20}2
�vroot.keras_api.metrics.2"_tf_keras_metric*�{"class_name": "BinaryCrossentropy", "name": "bin_cross", "dtype": "float32", "config": {"name": "bin_cross", "dtype": "float32", "from_logits": false, "label_smoothing": 0}, "shared_object_id": 21}2
�wroot.keras_api.metrics.3"_tf_keras_metric*�{"class_name": "AUC", "name": "auc", "dtype": "float32", "config": {"name": "auc", "dtype": "float32", "num_thresholds": 200, "curve": "ROC", "summation_method": "interpolation", "multi_label": false, "label_weights": null}, "shared_object_id": 22}2
�xroot.keras_api.metrics.4"_tf_keras_metric*�{"class_name": "Precision", "name": "pre", "dtype": "float32", "config": {"name": "pre", "dtype": "float32", "thresholds": null, "top_k": null, "class_id": null}, "shared_object_id": 23}2
�yroot.keras_api.metrics.5"_tf_keras_metric*�{"class_name": "Recall", "name": "rec", "dtype": "float32", "config": {"name": "rec", "dtype": "float32", "thresholds": null, "top_k": null, "class_id": null}, "shared_object_id": 24}2
�zroot.keras_api.metrics.6"_tf_keras_metric*�{"class_name": "TruePositives", "name": "tru_pos", "dtype": "float32", "config": {"name": "tru_pos", "dtype": "float32", "thresholds": null}, "shared_object_id": 25}2
�{root.keras_api.metrics.7"_tf_keras_metric*�{"class_name": "TrueNegatives", "name": "tru_neg", "dtype": "float32", "config": {"name": "tru_neg", "dtype": "float32", "thresholds": null}, "shared_object_id": 26}2
�|root.keras_api.metrics.8"_tf_keras_metric*�{"class_name": "FalsePositives", "name": "fal_pos", "dtype": "float32", "config": {"name": "fal_pos", "dtype": "float32", "thresholds": null}, "shared_object_id": 27}2
�}root.keras_api.metrics.9"_tf_keras_metric*�{"class_name": "FalseNegatives", "name": "fal_neg", "dtype": "float32", "config": {"name": "fal_neg", "dtype": "float32", "thresholds": null}, "shared_object_id": 28}2