import streamlit as st
import pickle
import pandas as pd

# List of teams and cities
teams = ['Sunrisers Hyderabad', 'Mumbai Indians', 'Gujarat Lions',
         'Royal Challengers Bangalore', 'Kolkata Knight Riders',
         'Kings XI Punjab', 'Chennai Super Kings', 'Rajasthan Royals',
         'Delhi Capitals']

cities = ['Hyderabad', 'Bangalore', 'Mumbai', 'Indore', 'Kolkata', 'Delhi',
          'Chandigarh', 'Jaipur', 'Chennai', 'Cape Town', 'Port Elizabeth',
          'Durban', 'Centurion', 'East London', 'Johannesburg', 'Kimberley',
          'Bloemfontein', 'Ahmedabad', 'Cuttack', 'Nagpur', 'Dharamsala',
          'Visakhapatnam', 'Pune', 'Raipur', 'Ranchi', 'Abu Dhabi',
          'Sharjah', 'Mohali', 'Bengaluru']

# Load the pre-trained pipeline
pipe = pickle.load(open('pipe.pkl', 'rb'))

# Title of the Streamlit app
st.title('IPL Match Win Predictor')

# Input fields for user input
col1, col2 = st.columns(2)

with col1:
    batting_team = st.selectbox('Batting team', sorted(teams))
with col2:
    bowling_team = st.selectbox('Bowling team', sorted(teams))

city = st.selectbox('Select Venue', sorted(cities))

col3, col4 = st.columns(2)

with col3:
    score = st.number_input('Current Score', min_value=0)
with col4:
    wickets = st.number_input('Wickets out', min_value=0, max_value=10)

overs = st.number_input('Overs completed', min_value=0.0, max_value=20.0, step=0.1)
target = st.number_input('Target', min_value=0)

# If the 'Predict Probability' button is clicked
if st.button('Predict Probability'):
    # Calculate necessary features for prediction
    runs_left = target - score
    balls_left = 120 - (overs * 6)
    wickets_left = 10 - wickets
    crr = score / overs if overs > 0 else 0
    rrr = (runs_left * 6) / balls_left if balls_left > 0 else 0

    # Create a DataFrame for the model input
    input_df = pd.DataFrame({
        'batting_team': [batting_team],
        'bowling_team': [bowling_team],
        'city': [city],
        'runs_left': [runs_left],
        'balls_left': [balls_left],
        'wickets': [wickets_left],
        'total_runs_x': [target],
        'crr': [crr],
        'rrr': [rrr]
    })

    # Handle missing transformers or preprocessing issues
    try:
        # Predict the probability using the pipeline
        result = pipe.predict_proba(input_df)
        loss = result[0][0]
        win = result[0][1]

        # Display the results
        st.header(f"{batting_team} - {round(win * 100)}%")
        st.header(f"{bowling_team} - {round(loss * 100)}%")

    except Exception as e:
        # Show the error message for debugging
        st.error(f"An error occurred: {str(e)}")
        st.write("Ensure that the input data format matches the pipeline's expected format.")
