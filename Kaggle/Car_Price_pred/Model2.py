import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, root_mean_squared_error

# Loading Dataset and analyzing
data = pd.read_csv("CarPrice_.csv")
print(data.head())
print(data.info())

# Mapping Doornumber
data['doornumber'] = data['doornumber'].map({'two':0, 'four':1})

# Dropping irrelevant columns
columns_to_drop = ['car_ID', 'CarName', 'enginelocation']
data = data.drop(columns=columns_to_drop)

# (Symboling) Value of +3 indicates that the auto is risky, -3 that it is probably pretty safe		
data['symboling'] = data['symboling'].apply(
    lambda x: 3 if x >= 3 else (2 if -2 <= x <= 2 else 1)
)

# One-hot encoding
data = pd.get_dummies(data, columns=['fueltype', 'aspiration', 'fuelsystem', 'cylindernumber', 'enginetype', 'carbody', 'drivewheel'], drop_first=True)
print(data.info())

# Splitting into features and target variable
X = data.drop(columns='price')
y = data['price']

# Scaling numerical features
scaler = StandardScaler()
numerical_features = X.select_dtypes(include=['int64', 'float64']).columns
data[numerical_features] = scaler.fit_transform(data[numerical_features])

# Splitting into training and testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initializing Model and Training
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions and Evaluations
y_pred = model.predict(X_test)
print("\nR2 : ", r2_score(y_test, y_pred))
print("\nMSE: ", mean_squared_error(y_test, y_pred))
print("\nRMSE: ", root_mean_squared_error(y_test, y_pred))
