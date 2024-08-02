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

with st.expander('Data visualization'):
  st.scatter_chart(data=df, x='bill_length_mm', y='body_mass_g', color='species')
