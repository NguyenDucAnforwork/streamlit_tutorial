import streamlit as st
import pandas as pd
st.title("App name")
st.write("Hello world")

st.info("This is a ML app") 

df = pd.read_csv('https://raw.githubusercontent.com/dataprofessor/data/master/penguins_cleaned.csv')
df