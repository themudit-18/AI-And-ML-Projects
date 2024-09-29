import streamlit as st
import pandas as pd
import pickle
from utils.data_processing import preprocess_input

# Load the trained model
with open('models/model.pkl', 'rb') as f:
    model = pickle.load(f)

st.title('Cricket World Cup Match Outcome Predictor')

# Load data for user selection
data = pd.read_csv('data/world_cup_data.csv')

# Ensure columns are stripped of any extra spaces
data.columns = data.columns.str.strip()

# Extract unique team names and venue
teams = pd.concat([data['team1'], data['team2']]).unique()
venues = data['venue'].unique()

# User input for team and match details
team1 = st.selectbox('Select Team 1', teams)
team2 = st.selectbox('Select Team 2', teams)
venue = st.selectbox('Select Venue', venues)
toss_winner = st.selectbox('Who won the toss?', [team1, team2])

# When the user clicks the Predict button
if st.button('Predict Outcome'):
    # Preprocess the input to match the training format
    input_data = {
        'team1': team1,
        'team2': team2,
        'venue': venue,
        'toss_winner': toss_winner
    }
    
    # Preprocess the input data (convert to dummies or match the model input)
    processed_input = preprocess_input(input_data, data)
    
    # Predict using the loaded model
    prediction = model.predict([processed_input])
    
    # Output the prediction
    st.write(f'The predicted winner is: {prediction[0]}')
