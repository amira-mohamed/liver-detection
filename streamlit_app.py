'''
import streamlit as st

st.title('🎈 App Name')

st.write('Hello world!')
'''
import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load the trained model
#model = joblib.load("Model_liver.pkl")  # Replace with your model file

# Function to make predictions


# Page Title
st.title("liver disease classification")

# create page columns 
col1, col2, col3, col4, col5= st.columns(5)

# Inputs
gender = col1.selectbox("Gender", ["Male", "Female"])
age = col2.number_input("AGE", min_value=0, max_value=120, step=1)
tb = col3.number_input("TB", min_value=0.0, max_value=10.0, step=0.1)
db = col4.number_input("DB", min_value=0.0, max_value=10.0, step=0.1)
alkphos = col5.number_input("Alkphos", min_value=0, max_value=1000, step=1)
sgpt = col1.number_input("SGPT", min_value=0, max_value=100, step=1)
sgot = col2.number_input("SGOT", min_value=0, max_value=100, step=1)
tp = col3.number_input("TP", min_value=0.0, max_value=10.0, step=0.1)
alb = col4.number_input("ALB", min_value=0.0, max_value=10.0, step=0.1)
ag_ratio = col5.number_input("A/G Ratio", min_value=0.0, max_value=5.0, step=0.1)



# Button for prediction
st.button("Predict"):
        st.error("The patient is classified as having liver disease.")
  
