''' Controlling Tree Complexity and Interpretability '''

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

# Create decision tree model
decision_tree_classifier = DecisionTreeClassifier(criterion='entropy', max_depth=8, min_impurity_decrease=0.01, random_state=19)
decision_tree_classifier.fit(features_train, labels_train)

# Predict
predicted_labels_test = decision_tree_classifier.predict(features_test)
test_accuracy = accuracy_score(labels_test, predicted_labels_test)

predicted_labels_train = decision_tree_classifier.predict(features_train)
train_accuracy = accuracy_score(labels_train, predicted_labels_train)

print(f"Test accuracy: {test_accuracy}")
print(f"Training accuracy: {train_accuracy}")

# Report on top five most important features
print(decision_tree_classifier.feature_importances_)

''' Top 5 most important features are features #8, 23, 22, 21, 28, which are:
- concave_points1
- perimeter3
- texture3
- radius3
- compactness3
'''

''' Model complexity
- Limiting the maximum depth and setting a minimum impurity decrease increases the model's accuracy on the test data.
    It prevents the model from creating decision nodes until it's perfectly fit to the training data.
- Knowing feature importance helps us understand which features contribute to the final classification and which do not.
'''