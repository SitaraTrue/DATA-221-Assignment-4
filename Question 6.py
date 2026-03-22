''' Convolutional Neural Network '''

# Load dataset
from tensorflow.keras.datasets import fashion_mnist
import tensorflow as tf
from tensorflow.keras import layers,models
(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

# Normalize the dataset
mu = X_train.mean(axis=(0,1))
sigma = X_train.std(axis=(0,1))
x_std = (X_train - mu)/(sigma + 1e-8)

# Reshape the images to include the channel dimension
H,W,C = 28,28,3
input_dim = H*W*C

# Build CNN
cnn_model = models.Sequential([
    layers.Input(shape=(28,28,3)),
    layers.Conv2D(16,3,padding="same", activation="relu"),
    layers.MaxPool2D(),
    layers.Conv2D(32,3,padding="same", activation="relu"),
    layers.MaxPool2D()
])
cnn_model.add(layers.Flatten())
cnn_model.add(layers.Dense(128,activation="relu"))
cnn_model.add(layers.Dropout(0.3))
cnn_model.add(layers.Dense(10,activation="softmax"))

cnn_model.summary()