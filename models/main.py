from LinearRegression import decisionTree
import numpy as np

# Create a toy dataset
np.random.seed(0)

# Feature 1: Random values between 1 and 10
feature1 = np.random.uniform(1, 10, 100)

# Feature 2: Random values between 1 and 10
feature2 = np.random.uniform(1, 10, 100)

# Target: Class 0 if the sum of the features is less than 10, otherwise Class 1
target = (feature1 + feature2 >= 10).astype(int)

# Combine the features into a single array
X = np.column_stack((feature1, feature2))
y = target



clf = decisionTree(min_samples_split=2, max_depth=3, min_impurity_decrease=0.01)
clf.fit(X, y)

    # Predictions
print("Predictions:", clf.predict(X))

        # Feature importance
print("Feature importances:", clf.feature_importances())

            # Visualize the tree
#f.visualize(feature_names=["Feature 1", "Feature 2"], class_names=["Class 0", "Class 1"])
clf.export_graphviz()