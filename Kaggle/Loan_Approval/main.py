import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score

# Load Dataset
loan_data = pd.read_csv("loan_data.csv")
print(loan_data.info())
print("Dataset loaded successfully.\n")

# Finding Missing Values
print("Missing values in each column:\n", loan_data.isnull().sum(), "\n")

# Label Encoding
loan_data["person_gender"] = loan_data["person_gender"].map({'male':1, 'female':0})
print("Converted 'person_gender' column:\n", loan_data["person_gender"].head(), "\n")

loan_data["previous_loan_defaults_on_file"] = loan_data["previous_loan_defaults_on_file"].map({'Yes':1, 'No':0})
print("Converted 'previous_loan_defaults_on_file' column:\n", loan_data["previous_loan_defaults_on_file"].head(), "\n")

label_encoder = LabelEncoder()
loan_data["person_education"] = label_encoder.fit_transform(loan_data["person_education"])

loan_data = pd.get_dummies(loan_data, columns=['person_home_ownership'], drop_first=True)
loan_data = pd.get_dummies(loan_data, columns=['loan_intent'], drop_first=True)

# Splitting into features(X) and (y)
X = loan_data.drop("loan_status", axis=1)
y = loan_data["loan_status"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # SMOTE for imbalance in loan_status
# smote = SMOTE(random_state=42)
# X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
# print(y_train_balanced.value_counts(normalize=True))

# Train Logistic Regression Model
logistic_model = LogisticRegression(solver='saga', max_iter=10000)
logistic_model.fit(X_train, y_train)

# Predictions and Evaluation
y_pred = logistic_model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))

# Get the classification report (precision, recall, f1-score)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Get ROC-AUC score (for binary classification)
roc_auc = roc_auc_score(y_test, logistic_model.predict_proba(X_test)[:, 1])
print(f"\nROC-AUC Score: {roc_auc}")