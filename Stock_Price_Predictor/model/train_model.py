import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import joblib

def train_model(data_path):
    # Load the data
    stock_data = pd.read_csv(data_path)
    
    # Prepare the data
    X = stock_data[['Open', 'High', 'Low', 'Volume']]
    y = stock_data['Close']
    
    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train the model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Save the model
    joblib.dump(model, 'stock_model.pkl')
    print("Model trained and saved as 'stock_model.pkl'")

# Call the function to train the model
train_model('../data/stock_data.csv')
