import streamlit as st
import pandas as pd
st.title("App name")
st.write("Hello world")

st.info("This is a ML app") 

with st.expander('Data'):
  st.write('Raw data')
  df = pd.read_csv('https://raw.githubusercontent.com/dataprofessor/data/master/penguins_cleaned.csv')
  df

  st.write('**X**')
  X = df.drop('species', axis=1)
  X
  
  st.write("**y**")
  y = df['species']
  y
st.header("hehe")
with st.expander('Data visualization'):
  st.scatter_chart(data=df, x='bill_length_mm', y='body_mass_g', color='species')

add_selectbox = st.sidebar.selectbox(
  "how would you like to be contacted?",
  ("email", "home phone", "mobile phone")
)

with st.sidebar:
  st.header('input features')
  island = st.selectbox('Island', ('Biscoe', 'Dream', 'Torgensen'))
