''' Model Evaluation and Comparison '''

# Load dataset
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import tensorflow as tf
from sklearn.tree import DecisionTreeClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, InputLayer
from sklearn import metrics
import matplotlib as plt

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

# Make the confusion matrix
nn_confusion_matrix = metrics.confusion_matrix(labels_test, neural_predicted)
nn_cm = metrics.ConfusionMatrixDisplay(confusion_matrix=nn_confusion_matrix, display_labels=[0,1])
nn_cm.plot()
plt.show()

# Create decision tree model
decision_tree_classifier = DecisionTreeClassifier(criterion='entropy', max_depth=8, min_impurity_decrease=0.01)
decision_tree_classifier.fit(features_train, labels_train)