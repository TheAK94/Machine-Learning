import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn import tree

data = pd.read_csv('Dataset.csv')
print("Dataset Length:", len(data))
print("Dataset Shape:", data.shape)
print(data.head())

print("\nMissing values in each column:\n", data.isnull().sum())

data = data.drop(columns=['sum'])
print("\nDropped the column 'sum' ")
print(data.head())

X = data.drop('result', axis=1)
y = data['result']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf_entropy = DecisionTreeClassifier(criterion="entropy", random_state=42, max_depth=3, min_samples_leaf=5)
clf_entropy.fit(X_train, y_train)

y_pred = clf_entropy.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
