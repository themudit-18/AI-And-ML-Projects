import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import pickle

# Sample dataset - You would replace this with your actual IPL match data
# This is just a mock dataset structure, adjust it to your real dataset
data = pd.DataFrame({
    'batting_team': ['Sunrisers Hyderabad', 'Mumbai Indians', 'Gujarat Lions'],
    'bowling_team': ['Mumbai Indians', 'Sunrisers Hyderabad', 'Royal Challengers Bangalore'],
    'city': ['Hyderabad', 'Mumbai', 'Bangalore'],
    'runs_left': [100, 80, 120],
    'balls_left': [60, 40, 100],
    'wickets': [5, 3, 7],
    'total_runs_x': [180, 160, 200],  # Target score
    'crr': [6.5, 7.0, 5.8],  # Current run rate
    'rrr': [8.5, 9.0, 7.2],  # Required run rate
    'winning_team': [1, 0, 1]  # 1 for batting team win, 0 for bowling team win
})

# Features and target variable
X = data.drop(columns=['winning_team'])
y = data['winning_team']

# Define the categorical features
categorical_features = ['batting_team', 'bowling_team', 'city']

# Preprocessing pipeline: OneHotEncode the categorical features
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ],
    remainder='passthrough'
)

# Define the model pipeline with RandomForestClassifier
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier())
])

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model.fit(X_train, y_train)

# Save the trained pipeline to a file
with open('pipe.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Model trained and saved as pipe.pkl")
