import streamlit as st
import pandas as pd
import joblib
import numpy as np
from xgboost import XGBClassifier



# Load the trained model
model = joblib.load("Model_liver_xgb.pkl")  # including the trained model

# Function to make predictions
def predict(input_data):
    prediction = model.predict(input_df)
    return prediction[0]

# Page Title
st.title("liver disease classification")
st.info('This App for Liver Detection predicition')

# Load Data
with st.expander('Data'):
 st.write('Raw Data')
 df = pd.read_csv('liver_dataset.csv')
 df

# create page columns 
col1, col2, col3, col4, col5= st.columns(5)

# Inputs

age = col1.number_input("Age", min_value=0, max_value=120, step=1)
gender = col2.selectbox("Gender", ["Male", "Female"])
tb = col3.number_input("TB", min_value=0.0, max_value=10.0, step=0.1)
db = col4.number_input("DB", min_value=0.0, max_value=10.0, step=0.1)
alkphos = col5.number_input("Alkphos", min_value=0, max_value=1000, step=1)
sgpt = col1.number_input("Sgpt", min_value=0, max_value=100, step=1)
sgot = col2.number_input("Sgot", min_value=0, max_value=100, step=1)
tp = col3.number_input("TP", min_value=0.0, max_value=10.0, step=0.1)
alb = col4.number_input("ALB", min_value=0.0, max_value=10.0, step=0.1)
ag_ratio = col5.number_input("A/G Ratio", min_value=0.0, max_value=5.0, step=0.1)


# Process gender input (if needed for your model)
gender = 1 if gender == "Male" else 2  # Assume 1 = Male, 0 = Female

# Combine inputs into a single array

data = {'Age' : [age],
        'Gender': [gender],
        'TB' : [tb], 
        'DB': [db],
        'Alkphos' : [alkphos], 
        'SGPT': [sgpt], 
        'SGOT' : [sgot], 
        'TP' : [tp], 
        'ALB' : [alb], 
        'A/G Ratio' : [ag_ratio]}

input_df = pd.DataFrame(data, index = [0])

# input Data
with st.expander('Input Data'):
 st.write('**Input Data**')
 input_df

# Button for prediction
if st.button("Predict"):
    result = predict(input_df)
    if result == 1:  # Assuming 1 = Disease and 0 = No Disease
        st.error("The patient is classified as having liver disease.")
    else:
        st.success("The patient is classified as healthy.")
 
