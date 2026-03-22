''' Controlling Tree Complexity and Interpretability '''

# Load dataset
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
data = load_breast_cancer()

# Construct feature matrix X and target vector y
features_x = data.data
target_y = data.target

# Create train-test split
features_train, features_test, labels_train, labels_test = train_test_split(features_x, target_y, test_size=.2, random_state=19)

# Create decision tree model
decision_tree_classifier = DecisionTreeClassifier(criterion='entropy')
decision_tree_classifier.fit(features_train, labels_train)