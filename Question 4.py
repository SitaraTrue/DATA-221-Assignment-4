''' Neural Network for Binary Classification '''

# Load dataset
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
data = load_breast_cancer()

# Construct feature matrix X and target vector y
features_x = data.data
target_y = data.target

# Create train-test split
features_train, features_test, labels_train, labels_test = train_test_split(features_x, target_y, test_size=.2, random_state=19)