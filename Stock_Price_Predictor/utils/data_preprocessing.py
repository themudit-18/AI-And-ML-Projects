import pandas as pd
from sklearn.preprocessing import StandardScaler

def preprocess_data(stock_data):
    stock_data['Date'] = pd.to_datetime(stock_data['Date'])
    
    stock_data.set_index('Date', inplace=True)
    
    features = stock_data[['Open', 'High', 'Low', 'Volume']]
    
    features.fillna(method='ffill', inplace=True)  # Forward fill to handle missing data
    
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    
    scaled_features_df = pd.DataFrame(scaled_features, columns=features.columns, index=features.index)
    
    return scaled_features_df
