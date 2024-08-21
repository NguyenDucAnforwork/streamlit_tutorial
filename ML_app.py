import streamlit as st
import pandas as pd
import numpy as np
import pickle

st.title("CO2 Emissions Prediction App")
st.write("Predict CO2 Emissions based on vehicle features")

st.info("This is a Machine Learning app for regression")

# Load the saved regression model
with open('model.pkl', 'rb') as file:
    regressor = pickle.load(file)

with st.expander('Data'):
    st.write('Sample Data')
    df = pd.DataFrame({
        'MODELYEAR': [2014],
        'ENGINESIZE': [2.0],
        'CYLINDERS': [4],
        'FUELCONSUMPTION_CITY': [9.9],
        'FUELCONSUMPTION_HWY': [6.7],
        'FUELCONSUMPTION_COMB': [8.5],
        'FUELCONSUMPTION_COMB_MPG': [33],
        'CO2EMISSIONS': [196]
    })
    st.write(df)

with st.sidebar:
    st.header('Input Features')
    model_year = st.slider('Model Year', 1990, 2024, 2014)
    engine_size = st.slider('Engine Size', 0.0, 10.0, 2.0)
    cylinders = st.slider('Cylinders', 2, 16, 4)
    fuel_consumption_city = st.slider('Fuel Consumption City (L/100 km)', 1.0, 30.0, 9.9)
    fuel_consumption_hwy = st.slider('Fuel Consumption Hwy (L/100 km)', 1.0, 20.0, 6.7)
    fuel_consumption_comb = st.slider('Fuel Consumption Comb (L/100 km)', 1.0, 25.0, 8.5)
    fuel_consumption_comb_mpg = st.slider('Fuel Consumption Comb MPG', 5, 100, 33)

    input_data = {
        'MODELYEAR': model_year,
        'ENGINESIZE': engine_size,
        'CYLINDERS': cylinders,
        'FUELCONSUMPTION_CITY': fuel_consumption_city,
        'FUELCONSUMPTION_HWY': fuel_consumption_hwy,
        'FUELCONSUMPTION_COMB': fuel_consumption_comb,
        'FUELCONSUMPTION_COMB_MPG': fuel_consumption_comb_mpg
    }

    input_df = pd.DataFrame(input_data, index=[0])

with st.expander('Input Features'):
    st.write('**Input Data**')
    st.write(input_df)

# Predict CO2 emissions
prediction = regressor.predict(input_df)
st.subheader('Predicted CO2 Emissions')
st.write(f'{prediction[0]:.2f} g/km')
