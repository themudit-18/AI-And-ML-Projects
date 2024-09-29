import pandas as pd

def preprocess_input(input_data, data):
    # Convert input data to a DataFrame
    input_df = pd.DataFrame([input_data])
    
    # Use the columns from the original training data for one-hot encoding
    X = data[["team1", "team2", "venue", "toss_winner"]]
    X = pd.get_dummies(X)
    
    # Apply one-hot encoding to the input
    input_df_encoded = pd.get_dummies(input_df)
    
    # Ensure input has the same columns as training data (missing columns are set to 0)
    input_df_encoded = input_df_encoded.reindex(columns=X.columns, fill_value=0)
    
    return input_df_encoded.iloc[0].values  # Return the processed input as a flat array

print("Data preprocessing is successfully compeleted!!")