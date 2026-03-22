''' Neural Network for Binary Classification '''

# Load dataset
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, InputLayer
from sklearn.metrics import accuracy_score
data = load_breast_cancer()
tf.random.set_seed(1)

# Construct feature matrix X and target vector y
features_x = data.data
target_y = data.target

# Standardize the data
mu = features_x.mean(axis=(0,1))
sigma = features_x.std(axis=(0,1))
x_std = (features_x-mu) / (sigma + 1e-8)

# Create train-test split
features_train, features_test, labels_train, labels_test = train_test_split(x_std, target_y, test_size=.2, random_state=19)

# Train neural network
neural_network_model = Sequential()
input_layer = InputLayer(input_shape=(30,))
neural_network_model.add(input_layer)
hidden_layer=Dense(15)
neural_network_model.add(hidden_layer)
output_layer = Dense(1, activation='sigmoid')
neural_network_model.add(output_layer)

# Compile
neural_network_model.compile(loss='binary_crossentropy', metrics=['accuracy'])

# Fit
neural_network_model.fit(x_std, target_y, epochs=10)

# Evaluate
test_accuracy = neural_network_model.evaluate(features_test, labels_test)
train_accuracy = neural_network_model.evaluate(features_train, labels_train)
print(f"Test accuracy: {test_accuracy[1]}")
print(f"Train accuracy: {train_accuracy[1]}")

''' Feature scaling
- Feature scaling puts all of the values in a consistent range, helping the neural network learn.
- Epochs are how many times the model repeats predicting, validating, computing loss, and updating the weights and biases to minimize loss.
'''