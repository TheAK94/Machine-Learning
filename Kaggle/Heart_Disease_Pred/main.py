import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import NearMiss
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score

# Load Dataset 
data = pd.read_csv("framingham.csv")
print(data.info())
print("Dataset loaded successfully.\n")

# Find if there are missing values
print(data.isnull().sum())

# Dropping column which is not relevant to training
data = data.drop(columns=["education"], axis=1)
print("Dropped the column 'education.\n")

# Filling missing values
data["cigsPerDay"] = data["cigsPerDay"].fillna(data["cigsPerDay"].median())
data["BMI"] = data["BMI"].fillna(data["BMI"].median())
data["glucose"] = data["glucose"].fillna(data["glucose"].median())
data["BPMeds"] = data["BPMeds"].fillna(method="ffill")
data["totChol"] = data["totChol"].fillna(method="ffill")
data = data.dropna(subset=['heartRate'])
print("Missing Values now:\n", data.isnull().sum(), "\n")

# Checking for class imbalance
print(data['TenYearCHD'].value_counts())

# Splitting into features(X) and class(y)
X = data.drop("TenYearCHD", axis=1)
y = data["TenYearCHD"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Training the model
logistic_model = LogisticRegression(class_weight={0: 1, 1: 3}, max_iter=1000)
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