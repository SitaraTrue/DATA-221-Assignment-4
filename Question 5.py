''' Model Evaluation and Comparison '''

# Load dataset
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import tensorflow as tf
from sklearn.tree import DecisionTreeClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, InputLayer
from sklearn import metrics
import matplotlib.pyplot as plt
import numpy as np

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

''' Neural Network '''

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

# Predict
neural_predicted = neural_network_model.predict(features_test)

''' Neural network confusion matrix '''
neural_predicted = np.round(neural_predicted).tolist()
nn_confusion_matrix = metrics.confusion_matrix(labels_test, neural_predicted)
print(f"Neural network confusion matrix: \n{nn_confusion_matrix}")

# Create decision tree model
decision_tree_classifier = DecisionTreeClassifier(criterion='entropy', max_depth=8, min_impurity_decrease=0.01)
decision_tree_classifier.fit(features_train, labels_train)

# Predict
decision_predicted = decision_tree_classifier.predict(features_test)

''' Decision tree confusion matrix '''
decision_confusion_matrix = metrics.confusion_matrix(labels_test, decision_predicted)
print(f"Decision tree confusion matrix: \n{decision_confusion_matrix}")


