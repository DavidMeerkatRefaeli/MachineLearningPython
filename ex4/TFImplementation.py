# This is a TF implementation of the NN exercises - note I still haven't figured an easy way to set pretrained weight
# on the network. 
# Also note that I didn't use the in-built tf.metrics.accuracy because it gives weird results.
# Also note that it might have been better to use sparse_softmax loss function, which would save the need to do one_hot, 
# but I tried to be as loyal to the original calculations as possible

import scipy.io
import numpy as np
import tensorflow as tf


# Load data
#     - X is a (5000,400) matrix with 5000 images of 20x20 pixels (=400)
#     - Y is a (5000,1) vector, with output from 1 to 10, where 10 means 0
mat = scipy.io.loadmat('./Data/ex3data1.mat')
X = mat['X']
y = np.int32(mat['y'])
y = np.where(y == 10, 0, y).squeeze()  # to avoid confusion, let's change 10 to what it really is: 0
m = X.shape[0]

# X must be casted from float64 to float32 as it is the default dtype in tf
tf_x = tf.placeholder(tf.float32, (None, X.shape[1]))  # input x

# Ex. 3 - part 2 - Load some preprocessed weights for initial feed-forward exploration
#     - Theta1 - (25,401) - the extra 1 is for the bias term, reduces to 25 dimensions in the latent space
#     - Theta2 - (10,26) - the extra 1 is for the bias term, reduces to 10 dimensions in the output space
mat_w = scipy.io.loadmat('./Data/ex3weights.mat')
Theta1 = mat_w['Theta1'][:, 1:]  # No need to add the bias
Theta2 = mat_w['Theta2'][:, 1:]  # No need to add the bias
w1 = tf.constant_initializer(Theta1.T)
w2 = tf.constant_initializer(Theta2.T)

# neural network layers
layer1 = tf.layers.dense(tf_x, 25, tf.nn.sigmoid, kernel_initializer=w1)   # hidden layer
output = tf.layers.dense(layer1, 10, kernel_initializer=w2)                # output layer

# Note we didn't set any activation function on the output layer. This is because the sigmoid_cross_entropy
# already does it when calculating the loss ... (:-?) so we will have to do it ourselves when computing accuracy

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    pred = sess.run(tf.argmax(tf.sigmoid(output), axis=1), feed_dict={tf_x: X}) + 1  # +1 to make it as y (1-10)
    pred = np.where(pred == 10, 0, pred)  # Change 10 to 0
    pred_y = pred == y
    print(f'Pre-Trained Accuracy: {np.mean(pred_y) * 100:.2f}%')

    i = np.random.randint(m)
    X_i = X[i, :].reshape((1, -1))
    single_pred = sess.run(tf.argmax(tf.sigmoid(output), axis=1), feed_dict={tf_x: X_i}) + 1
    print(f"single prediction: {single_pred % 10} | expected: {y[i]}")

# Ex. 4 - backpropagation
num_steps = 500

# We have to turn the y vector to one-hot encoding, because of how sigmoid_cross_entropy works
y_one_hot = np.zeros((len(y), 10))
y_one_hot[np.arange(len(y)), y] = 1

tf_y_one_hot = tf.placeholder(tf.int32, (None, y_one_hot.shape[1]))  # input y_one_hot

# neural network layers - this time without initialization
layer1 = tf.layers.dense(tf_x, 25, tf.nn.sigmoid)   # hidden layer
output = tf.layers.dense(layer1, 10)                # output layer

# Choose cost, optimizer and accuracy metric
loss_op = tf.losses.sigmoid_cross_entropy(multi_class_labels=y_one_hot, logits=output)
optimizer = tf.train.AdamOptimizer()
train_op = optimizer.minimize(loss_op)

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    for step in range(1, num_steps+1):
        # train and net output
        loss = sess.run(loss_op, feed_dict={tf_x: X, tf_y_one_hot: y_one_hot})
        sess.run(train_op, feed_dict={tf_x: X, tf_y_one_hot: y_one_hot})
        if (step % 50 == 0) | (step == 1):
            print(f'step {step} - loss: {loss}')

    print('Optimization finished')
    prediction = sess.run(tf.argmax(tf.sigmoid(output), axis=1), feed_dict={tf_x: X})
    pred_y = prediction == y
    print(f'Training Set Accuracy after training: {np.mean(pred_y) * 100}%')
