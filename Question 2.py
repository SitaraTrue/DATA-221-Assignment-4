''' Decision Tree Model Using Entropy '''

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
decision_tree_classifier = DecisionTreeClassifier(criterion='entropy', random_state=19)
decision_tree_classifier.fit(features_train, labels_train)

# Predict
predicted_labels_test = decision_tree_classifier.predict(features_test)
test_accuracy = accuracy_score(labels_test, predicted_labels_test)

predicted_labels_train = decision_tree_classifier.predict(features_train)
train_accuracy = accuracy_score(labels_train, predicted_labels_train)

print(f"Test accuracy: {test_accuracy}")
print(f"Training accuracy: {train_accuracy}")

''' Entropy
- Entropy measures the goodness of a partition (lower=better)
- The observed results show high accuracy on testing data, so it shows good generalization.
    Since the training has an accuracy of 1.0, it may be overfitted.
'''