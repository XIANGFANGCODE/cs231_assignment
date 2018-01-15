import tensorflow as tf

def cnn_fn(features, labels, mode, params):
    """
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
    """

    N = params.get('N', 1)
    M = params.get('M', 1)
    input_layer = features["x"]
    filters_num = params.get('filters_num', 64)
    kernel_size = params.get('kernel_size', [3, 3])
    pool_size = params.get('pool_size', [2, 2])
    strides = params.get('strides', [2, 2])
    dense_size = params.get('dense_size', 1024)
    dropout_rate = params.get('dropout_rate', 0.4)
    class_num = params.get('class_num', 10)
    learning_rate = params.get('learning_rate', 5e-4)

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
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    # Add evaluation metrics (for EVAL mode)
    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(labels=labels, predictions=predictions["classes"])}
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


def run_model(session, predict, loss_val, Xd, yd,
              epochs=1, batch_size=64, print_every=100,
              training=None, plot_losses=False):
    # have tensorflow compute accuracy
    correct_prediction = tf.equal(tf.argmax(predict, 1), y)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # shuffle indicies
    train_indicies = np.arange(Xd.shape[0])
    np.random.shuffle(train_indicies)

    training_now = training is not None

    # setting up variables we want to compute (and optimizing)
    # if we have a training function, add that to things we compute
    variables = [mean_loss, correct_prediction, accuracy]
    if training_now:
        variables[-1] = training

    # counter
    iter_cnt = 0
    for e in range(epochs):
        # keep track of losses and accuracy
        correct = 0
        losses = []
        # make sure we iterate over the dataset once
        for i in range(int(math.ceil(Xd.shape[0] / batch_size))):
            # generate indicies for the batch
            start_idx = (i * batch_size) % Xd.shape[0]
            idx = train_indicies[start_idx:start_idx + batch_size]

            # create a feed dictionary for this batch
            feed_dict = {X: Xd[idx, :],
                         y: yd[idx],
                         is_training: training_now}
            # get batch size
            actual_batch_size = yd[idx].shape[0]

            # have tensorflow compute loss and correct predictions
            # and (if given) perform a training step
            loss, corr, _ = session.run(variables, feed_dict=feed_dict)

            # aggregate performance stats
            losses.append(loss * actual_batch_size)
            correct += np.sum(corr)

            # print every now and then
            if training_now and (iter_cnt % print_every) == 0:
                print("Iteration {0}: with minibatch training loss = {1:.3g} and accuracy of {2:.2g}" \
                      .format(iter_cnt, loss, np.sum(corr) / actual_batch_size))
            iter_cnt += 1
        total_correct = correct / Xd.shape[0]
        total_loss = np.sum(losses) / Xd.shape[0]
        print("Epoch {2}, Overall loss = {0:.3g} and accuracy of {1:.3g}" \
              .format(total_loss, total_correct, e + 1))
        if plot_losses:
            plt.plot(losses)
            plt.grid(True)
            plt.title('Epoch {} Loss'.format(e + 1))
            plt.xlabel('minibatch number')
            plt.ylabel('minibatch loss')
            plt.show()
    return total_loss, total_correct