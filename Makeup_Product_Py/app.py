import streamlit as st
import pandas as pd
import joblib

# Load the model
model = joblib.load('models/sales_model.pkl')

# Define the input fields in your app
st.title("Makeup Sales Prediction")

# Example input fields
promotion = st.selectbox("Promotion", options=["Yes", "No"])
season = st.selectbox("Season", options=["Spring", "Summer", "Fall", "Winter"])
product_name = st.selectbox("Product Name", options=["Lipstick", "Foundation", "Mascara"])

# Get input data
input_data = {
    "promotion": 1 if promotion == "Yes" else 0,
    "season_Spring": 1 if season == "Spring" else 0,
    "season_Summer": 1 if season == "Summer" else 0,
    "season_Fall": 1 if season == "Fall" else 0,
    "season_Winter": 1 if season == "Winter" else 0,
    "product_name_Lipstick": 1 if product_name == "Lipstick" else 0,
    "product_name_Foundation": 1 if product_name == "Foundation" else 0,
    "product_name_Mascara": 1 if product_name == "Mascara" else 0,
    # Add more product names if needed
}

# Convert input data to DataFrame
input_df = pd.DataFrame([input_data])

# Ensure the input DataFrame has the same columns as the model
input_df = input_df.reindex(columns=model.feature_names_in_, fill_value=0)

# Make prediction
if st.button("Predict"):
    prediction = model.predict(input_df)
    st.write(f"Predicted Units Sold: {prediction[0]}")
