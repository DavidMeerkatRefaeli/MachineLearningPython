import scipy.io
import numpy as np
import tensorflow as tf  # You need to `pip install tensorflow` for this
from tensorflow import keras  # ... and this


# Load data
#     - X is a (5000,400) matrix with 5000 images of 20x20 pixels (=400)
#     - Y is a (5000,1) vector, with output from 1 to 10, where 10 means 0
from tensorflow.python.keras import regularizers
mat = scipy.io.loadmat('./Data/ex3data1.mat')
X = mat['X']
y = mat['y']
y = np.where(y == 10, 0, y)  # to avoid confusion, let's change 10 to what it really is: 0


def set_model():
    model = keras.Sequential([
        keras.layers.Input(400),
        keras.layers.Dense(25, activation=tf.nn.sigmoid),
        keras.layers.Dense(10, activation=tf.nn.softmax)
    ])
    model.compile(optimizer='adam',  # Conjugate Gradient is not available in keras
                  loss='sparse_categorical_crossentropy',  # does the 1-hot encoding for us
                  metrics=['accuracy'])

    return model


# Define the model
model = set_model()

# # Ex. 3 - part 2 - Load some preprocessed weights for initial feed-forward exploration
#     - Theta1 - (25,401) - the extra 1 is for the bias term, reduces to 25 dimensions in the latent space
#     - Theta2 - (10,26) - the extra 1 is for the bias term, reduces to 10 dimensions in the output space
mat_w = scipy.io.loadmat('./Data/ex3weights.mat')
Theta1 = mat_w['Theta1'][:, 1:]  # No need to add the bias
Theta2 = mat_w['Theta2'][:, 1:]  # No need to add the bias

model.set_weights([Theta1.T, np.zeros(25), Theta2.T, np.zeros(10)])
prob = model.predict(X)
# The pre-trained weights are set to give the first rows (0) the last category (9) - so we have to fix this:
pred = np.argmax(prob, axis=1).reshape(-1, 1) + 1  # +1 to make it as y (1-10)
pred = np.where(pred == 10, 0, pred)  # Change 10 to 0
pred_y = pred == y
print(f'Training Set Accuracy: {np.mean(pred_y)*100:.2f}%')  # Should also be around 97%

# unfortunately it's really a pain to calculate loss without actually running fit on the model, so no loss comparison

# Ex. 4 - backpropogation
model.fit(X, y, epochs=50)
prob = model.predict(X)
# Here no change is needed, because the model is fit to the correct y (i.e. range from 0 to 9)
pred = np.argmax(prob, axis=1).reshape(-1, 1)
pred_y = pred == y
print(f'Training Set Accuracy: {np.mean(pred_y)*100:.2f}%')
