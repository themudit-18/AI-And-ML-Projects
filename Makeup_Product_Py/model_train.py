# model_train.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import joblib

# Load the dataset
data = pd.read_csv('data/makeup_sales_data.csv')

# Preprocess the data
data['promotion'] = data['promotion'].map({'Yes': 1, 'No': 0})  # Convert categorical to numeric
data = pd.get_dummies(data, columns=['season', 'product_name'], drop_first=True)

# Features and target
X = data.drop(columns=['date', 'units_sold'])
y = data['units_sold']

# Combine X and y for cleaning
data_cleaned = pd.concat([X, y], axis=1).dropna()

# Split cleaned data back into features and target
X = data_cleaned.drop(columns=['units_sold'])
y = data_cleaned['units_sold']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Save the model
joblib.dump(model, 'models/sales_model.pkl')
print("Model trained and saved!")
