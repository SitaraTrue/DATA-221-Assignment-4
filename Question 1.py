''' Dataset Exploration and Understanding '''

# Load dataset
from sklearn.datasets import load_breast_cancer
data = load_breast_cancer()

# Construct feature matrix X and target vector y

features_x = data.data
target_y = data.target
print(features_x, target_y)

''' Shape of X and y


'''
