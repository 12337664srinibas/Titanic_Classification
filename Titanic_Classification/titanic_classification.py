# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
titanic_data = pd.read_csv('train.csv')

# Show the first few rows
print(titanic_data.head())

# Get basic info about the dataset
print(titanic_data.info())

# Check for missing values
print(titanic_data.isnull().sum())
# Fill missing Age values with the median
titanic_data['Age'].fillna(titanic_data['Age'].median(), inplace=True)

# Drop the Cabin column
titanic_data.drop(columns=['Cabin'], inplace=True)

# Fill missing Embarked values with the most frequent value
titanic_data['Embarked'].fillna(titanic_data['Embarked'].mode()[0], inplace=True)

# Convert categorical variables to numeric
titanic_data['Sex'] = titanic_data['Sex'].map({'male': 0, 'female': 1})
titanic_data['Embarked'] = titanic_data['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})

# Drop unnecessary columns
titanic_data.drop(columns=['Name', 'Ticket', 'PassengerId'], inplace=True)

# Check the processed data
print(titanic_data.head())
# Plot survival rate by gender
sns.countplot(x='Survived', hue='Sex', data=titanic_data)
plt.title("Survival Rate by Gender")
plt.show()

# Plot survival rate by passenger class
sns.countplot(x='Survived', hue='Pclass', data=titanic_data)
plt.title("Survival Rate by Class")
plt.show()

from sklearn.model_selection import train_test_split

# Define features (X) and target (y)
X = titanic_data.drop('Survived', axis=1)
y = titanic_data['Survived']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Check the shape of the training and test sets
print(f"X_train: {X_train.shape}, X_test: {X_test.shape}")

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Logistic Regression
logreg = LogisticRegression(max_iter=1000)
logreg.fit(X_train, y_train)
y_pred_logreg = logreg.predict(X_test)

# Random Forest Classifier
rfc = RandomForestClassifier(n_estimators=100, random_state=42)
rfc.fit(X_train, y_train)
y_pred_rfc = rfc.predict(X_test)

# Evaluate the models
print("Logistic Regression Accuracy:", accuracy_score(y_test, y_pred_logreg))
print("Random Forest Accuracy:", accuracy_score(y_test, y_pred_rfc))

# Classification Report for Logistic Regression
print("\nLogistic Regression Classification Report:")
print(classification_report(y_test, y_pred_logreg))

# Classification Report for Random Forest
print("\nRandom Forest Classification Report:")
print(classification_report(y_test, y_pred_rfc))

# Confusion Matrix for Random Forest
print("Random Forest Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_rfc))

# Feature importance for Random Forest
feature_importances = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance': rfc.feature_importances_
}).sort_values(by='Importance', ascending=False)

print(feature_importances)

# Plot the feature importance
sns.barplot(x='Importance', y='Feature', data=feature_importances)
plt.title("Feature Importance")
plt.show()

from sklearn.model_selection import cross_val_score

# Cross-validation for Random Forest
scores = cross_val_score(rfc, X, y, cv=5, scoring='accuracy')
print("Random Forest Cross-Validation Accuracy: {:.2f}%".format(scores.mean() * 100))
