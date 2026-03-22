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
H,W,C = 28,28,1

# Build CNN
cnn_model = models.Sequential([
    layers.Conv2D(16,3,padding="same", activation="relu", input_shape=(H,W,C)),
    layers.MaxPool2D(),
    layers.Conv2D(32,3,padding="same", activation="relu"),
    layers.MaxPool2D()
])
cnn_model.add(layers.Flatten())
cnn_model.add(layers.Dense(128,activation="relu"))
cnn_model.add(layers.Dropout(0.3))
cnn_model.add(layers.Dense(10,activation="softmax"))

cnn_model.summary()

# Train CNN
cnn_model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=["accuracy"])
cnn_model.fit(X_train, y_train, epochs=15, batch_size=64)

# Evaluate
test_loss, test_acc = cnn_model.evaluate(X_test, y_test, verbose=0)
print(f"Test loss: {test_loss}, test accuracy: {test_acc}")

''' CNNs
- CNNs are preferred because a neural network would have too many input neurons.
    CNNs learn pattern recognition in localized areas instead of over the whole photo.
- The Conv2D layer is learning the patterns in each area of the image.
'''