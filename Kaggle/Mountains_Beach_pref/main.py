import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score

# Loading Dataset and Analysis
data = pd.read_csv("mountains_vs_beaches_preferences.csv")
print(data.head())
print(data.info())
print(data.isnull().sum())

# one hot encoding for "Location", "Favourite_Season"
data_one_hot = pd.get_dummies(data, columns=['Location', 'Favorite_Season', 'Gender'], drop_first=True)

# for the feature "Preferred_Activities"
target_mean = data_one_hot.groupby('Preferred_Activities')['Preference'].mean()
data_one_hot['Preferred_Activities_Encoded'] = data_one_hot['Preferred_Activities'].map(target_mean)

# encoding for "Education_Level"
education_mapping = {'high school': 0, 'bachelor': 1, 'master': 2, 'doctorate': 3}
data_one_hot['Education_Level_Encoded'] = data_one_hot['Education_Level'].map(education_mapping)

# dropping originial columns that are encoded
data_one_hot = data_one_hot.drop(["Education_Level", "Preferred_Activities"], axis=1)

# Checking data
print(data_one_hot.info())
data_one_hot.isnull().sum()
data_one_hot.head()

# Checking for class imbalance
print(data_one_hot['Preference'].value_counts())

# Data split into X (features) and y (label)
X = data_one_hot.drop("Preference", axis=1)
y = data_one_hot["Preference"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Training the model
logistic_model = LogisticRegression(class_weight={0: 1, 1: 3}, max_iter=10000, C=0.1)
logistic_model.fit(X_train, y_train)

# Predictions and Evaluations
y_pred = logistic_model.predict(X_test)
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion matrix:\n", confusion_matrix(y_test, y_pred))

# Get the classification report (precision, recall, f1-score)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Get ROC-AUC score (for binary classification)
roc_auc = roc_auc_score(y_test, logistic_model.predict_proba(X_test)[:, 1])
print(f"\nROC-AUC Score: {roc_auc}")
