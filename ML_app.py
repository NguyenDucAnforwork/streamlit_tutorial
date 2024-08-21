import streamlit as st
import pandas as pd
import pickle
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Load the dataset
df = pd.read_csv('FuelConsumptionCo2.csv')

with st.expander('Data'):
  st.write('Raw data')
  df
  st.write('**X**')
  X_raw = df.drop('CO2EMISSIONS', axis=1)
  X_raw
  
  st.write("**y**")
  y_raw = df['CO2EMISSIONS']
  y_raw

# Convert columns to numeric and drop columns with NaN values
df = df.apply(pd.to_numeric, errors='coerce')
df_cleaned = df.dropna(axis=1, how='any')

# Separate features and target
X = df_cleaned.iloc[:, :-1]  # All columns except the last one
y = df_cleaned.iloc[:, -1]   # The last column

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize and train the Decision Tree regressor
regressor = DecisionTreeRegressor()
regressor.fit(X_train, y_train)

# Make predictions on the test set
y_pred = regressor.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Save the model
with open('model.pkl', 'wb') as file:
    pickle.dump(regressor, file)

# Streamlit App
st.title("🚗 CO2 Emissions Prediction App")
st.write("Use this app to predict CO2 emissions based on vehicle features.")

st.info("This is a Machine Learning app that uses a Decision Tree Regressor for regression.")

# Load the saved model
with open('model.pkl', 'rb') as file:
    regressor = pickle.load(file)

# Sidebar for input features
st.sidebar.header('Input Vehicle Features')
model_year = st.sidebar.slider('Model Year', 1990, 2024, 2014)
engine_size = st.sidebar.slider('Engine Size (L)', 0.0, 10.0, 2.0)
cylinders = st.sidebar.slider('Cylinders', 2, 16, 4)
fuel_consumption_city = st.sidebar.slider('Fuel Consumption City (L/100 km)', 1.0, 30.0, 9.9)
fuel_consumption_hwy = st.sidebar.slider('Fuel Consumption Hwy (L/100 km)', 1.0, 20.0, 6.7)
fuel_consumption_comb = st.sidebar.slider('Fuel Consumption Comb (L/100 km)', 1.0, 25.0, 8.5)
fuel_consumption_comb_mpg = st.sidebar.slider('Fuel Consumption Comb MPG', 5, 100, 33)

# Collect the inputs into a DataFrame
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

# Display input data
st.subheader('Input Data')
st.write(input_df)

# Predict CO2 emissions
prediction = regressor.predict(input_df)
st.subheader('Predicted CO2 Emissions')
st.write(f'🚗 **{prediction[0]:.2f} g/km**')

# Display model evaluation metrics
with st.expander('Model Performance'):
    st.subheader('Model Evaluation Metrics')
    st.write(f'- **Mean Squared Error (MSE):** {mse:.2f}')
    st.write(f'- **Mean Absolute Error (MAE):** {mae:.2f}')
    st.write(f'- **R-squared (R²):** {r2:.2f}')
