import numpy as np
import tensorflow as tf

def model_fn(features, labels, mode, params):
'''
    Logic to do the following:
    1. Configure the model via TensorFlow operations
    2. Define the loss function for training/evaluation
    3. Define the training operation/optimizer
    4. Generate predictions
    5. Return predictions/loss/train_op/eval_metric_ops in EstimatorSpec object

    The architectures of cnn:
    [conv-relu-conv-relu-pool]xN -> [affine-relu-drop]xM -> [fc]-> [softmax or SVM]

    Try filter size: 3 * 3
    Try more filters: 64
    Batch normalization: Try adding spatial batch normalization after convolution layers
    and vanilla batch normalization after affine layers.
    Use Learning Rate Decay
    Global Average Pooling: Instead of flattening and then having multiple affine layers, perform convolutions
    until your image gets small (7x7 or so) and then perform an average pooling operation to get to a 1x1 image
    picture (1, 1 , Filter#), which is then reshaped into a (Filter#) vector. This is used in Google's Inception
    Network (See Table 1 for their architecture)
    Dropout: use dropout after affine/dense layer, dropout rate is 0.4
'''
    N = params.get('N', 1)
    M = params.get('M', 1)
    n_layer_list = []
    m_layer_list = []
    input_layer = features
    filters_num = params.get('filters_num', 64)
    kernel_size = params.get('kernel_size', [3,3])
    pool_size = params.get('pool_size', [2,2])
    strides = params.get('strides', [2,2])
    dense_size = params.get('dense_size',1024)
    dropout_rate = params.get('dropout_rate',0.4)
    class_num = params.get('class_num',10)


    # [conv-relu-conv-relu-pool]xN
    for i in range(N):
        # Convolutional Relu layer
        conv1 = tf.layers.conv2d(
            inputs=input_layer,
            filters=filters_num,
            kernel_size=kernel_size,
            padding="same",
            activation=tf.nn.relu)

        # Convolutional Relu layer
        conv2 = tf.layers.conv2d(
            inputs=conv1,
            filters=filters_num,
            kernel_size=kernel_size,
            padding="same",
            activation=tf.nn.relu)

        # Max pool layer
        pool1 = tf.layers.max_pooling2d(inputs=conv2, pool_size=pool_size, strides=strides)

        input_layer = pool1

    # [affine-relu-drop]xM
    shape = input_layer.shape.as_list()
    dim = 1
    for i in range(1, len(shape)):
        dim = dim * shape[i]
    for i in range(M):
        # Dense Layer
        input_layer_flat = tf.reshape(input_layer, [-1, dim])
        dense = tf.layers.dense(inputs=input_layer_flat, units=dense_size, activation=tf.nn.relu)
        dropout = tf.layers.dropout(
            inputs=dense, rate=dropout_rate, training=mode == tf.estimator.ModeKeys.TRAIN)
        input_layer = dropout
        dim = dense_size

    # [fc]
    scores = tf.layers.dense(inputs=input_layer, units=class_num)

    # For prediction
    predictions = {
        # Generate predictions (for PREDICT and EVAL mode)
        "classes": tf.argmax(input=scores, axis=1),
        # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
        # `logging_hook`.
        "probabilities": tf.nn.softmax(scores, name="softmax_tensor")
    }
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # Calculate Loss (for both TRAIN and EVAL modes)
    onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=class_num)
    loss = tf.losses.softmax_cross_entropy(
        onehot_labels=onehot_labels, logits=scores)

    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    return EstimatorSpec(mode, predictions, loss, train_op, eval_metric_ops)