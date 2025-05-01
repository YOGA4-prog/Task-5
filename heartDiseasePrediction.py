# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report

# 1. Load and Prepare the Dataset
# Load your dataset (replace 'your_dataset.csv' with the actual file name)
data = pd.read_csv('heart.csv')

# Check for missing values and handle them if necessary
print("Missing values:\n", data.isnull().sum()) 
# Handle missing values if any, like data.fillna() or data.dropna()

# Define features (X) and target (y)
X = data.drop('target', axis=1)
y = data['target']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 2. Train a Decision Tree Classifier
dt_classifier = DecisionTreeClassifier(random_state=42)
dt_classifier.fit(X_train, y_train)

# 3. Visualize the Decision Tree
plt.figure(figsize=(20, 10))
plot_tree(dt_classifier, filled=True, feature_names=X.columns, class_names=['No Disease', 'Disease'], rounded=True)
plt.title("Decision Tree Visualization")
plt.show()

# 4. Analyze Overfitting and Control Tree Depth
train_scores = []
test_scores = []
depths = range(1, 21)  # Explore tree depths from 1 to 20

for depth in depths:
    dt_classifier = DecisionTreeClassifier(max_depth=depth, random_state=42)
    dt_classifier.fit(X_train, y_train)
    train_scores.append(dt_classifier.score(X_train, y_train))
    test_scores.append(dt_classifier.score(X_test, y_test))

# Plot accuracy vs. tree depth to visualize overfitting
plt.figure(figsize=(10, 6))
plt.plot(depths, train_scores, label='Train Accuracy', marker='o')
plt.plot(depths, test_scores, label='Test Accuracy', marker='s')
plt.xlabel('Tree Depth')
plt.ylabel('Accuracy')
plt.title('Overfitting Analysis: Decision Tree Depth vs Accuracy')
plt.legend()
plt.grid(True)
plt.show()

# Choose an optimal depth based on the analysis (e.g., where test accuracy is highest)
optimal_depth = depths[test_scores.index(max(test_scores))]
print(f"Optimal Tree Depth: {optimal_depth}")

# Retrain Decision Tree with the optimal depth
dt_classifier = DecisionTreeClassifier(max_depth=optimal_depth, random_state=42)
dt_classifier.fit(X_train, y_train)

# 5. Train a Random Forest and Compare Accuracy
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train, y_train)

# Evaluate and compare accuracies
dt_accuracy = dt_classifier.score(X_test, y_test)
rf_accuracy = rf_classifier.score(X_test, y_test)
print(f"Decision Tree Accuracy: {dt_accuracy}")
print(f"Random Forest Accuracy: {rf_accuracy}")

# 6. Interpret Feature Importances (Random Forest)
feature_importances = rf_classifier.feature_importances_
features = X.columns

# Create a DataFrame for better visualization
importances_df = pd.DataFrame({'Feature': features, 'Importance': feature_importances})
importances_df = importances_df.sort_values(by='Importance', ascending=False)
print("\nFeature Importances:")
print(importances_df)

# Plot Feature Importances
plt.figure(figsize=(12, 6))
plt.barh(importances_df['Feature'], importances_df['Importance'])
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title('Feature Importances from Random Forest')
plt.show()

# 7. Evaluate using Cross-Validation
dt_cv_scores = cross_val_score(dt_classifier, X, y, cv=5)
rf_cv_scores = cross_val_score(rf_classifier, X, y, cv=5)

print(f"\nDecision Tree Cross-Validation Scores: {dt_cv_scores}")
print(f"Mean Decision Tree CV Accuracy: {np.mean(dt_cv_scores)}")
print(f"\nRandom Forest Cross-Validation Scores: {rf_cv_scores}")
print(f"Mean Random Forest CV Accuracy: {np.mean(rf_cv_scores)}")
