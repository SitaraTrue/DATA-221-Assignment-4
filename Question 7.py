''' CNN Error Analysis and Misclassification '''

# Load dataset
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras import layers,models
import tensorflow as tf
from sklearn import metrics
import matplotlib.pyplot as plt

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

# Predictions and Confusion Matrix
predicted_probs = cnn_model.predict(X_test, verbose=0)
predicted_labels = []
for prob in predicted_probs:
    label = int(tf.argmax(prob))
    predicted_labels.append(label)
confusion_matrix = metrics.confusion_matrix(y_test, predicted_labels)
print(confusion_matrix)

# Find 3 misclassified images
counter = 0
for i in range(0,len(predicted_labels)):
    if predicted_labels[i] != y_test[i]:
        plt.figure(figsize=[8,8])
        plt.imshow(X_test[i,:,:], cmap='gray')
        plt.show()
        print(f"Predicted: {predicted_labels[i]}, Actual: {y_test[i]})")
        counter += 1
    if counter == 3:
        break

''' Misclassification
- The pattern in the misclassifications is that the prediction was the next closest thing.
    Predicted sandal, actually sneaker (both shoes)
    Predicted pullover, actually coat (both tops/outer layers)
    Predicted t-shirt, was actually shirt (both shirts)
- This could be improved by having more data, which would help the model learn the differences between similar categories.
'''