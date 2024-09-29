import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle
from sklearn.metrics import accuracy_score

# Load and preprocess the data
data = pd.read_csv('../data/world_cup_data.csv')

# Strip any extra spaces from column names
data.columns = data.columns.str.strip()

# Drop rows with missing values
data = data.dropna(subset=["team1", "team2", "venue", "toss_winner", "winner"])

# Select relevant columns for X and y
X = data[["team1", "team2", "venue", "toss_winner"]]
y = data['winner']

# Convert categorical data to numerical using one-hot encoding
X = pd.get_dummies(X)

# Check if there's enough data to perform train-test split
if len(X) > 1:
    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
else:
    # If the dataset is too small, use all data for training
    X_train, y_train = X, y
    X_test, y_test = X, y
    print("Using entire dataset for training since it's too small to split.")

# Train the model using a Random Forest Classifier
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Save the trained model
with open('../models/model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Model trained and saved!")
