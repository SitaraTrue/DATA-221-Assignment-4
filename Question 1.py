''' Dataset Exploration and Understanding '''

# Load dataset
from sklearn.datasets import load_breast_cancer
data = load_breast_cancer()

# Construct feature matrix X and target vector y

features_x = data.data
target_y = data.target

# Report on shape of X and y
print(f"X shape: {features_x.shape}") # (569,30)
print(f"y shape: {target_y.shape}") # (569,)

# Report the number of samples belonging to each class
counter0 = 0
counter1 = 0
for row in target_y:
    if row == 0:
        counter0 += 1
    else:
        counter1 += 1
print(f"Class 0: {counter0} samples")
print(f"Class 1: {counter1} samples")

''' Balance
- This dataset is imbalanced, as class 0 takes up about a third and class 1 is two-thirds.
- Balance is important so that the model has enough information to learn about each class equally.
    A balanced dataset leads to higher quality predictions.
'''