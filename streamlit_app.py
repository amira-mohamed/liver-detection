import streamlit as st
import pandas as pd
import joblib
import numpy as np


# Load the trained model
model = joblib.load("Model_liver_xgb.pkl")  # including the trained model

# Function to make predictions
def predict(input_data):
    prediction = model.predict([input_data])
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


# Process gender input (if needed for your model)
gender_numeric = 1 if gender == "Male" else 2  # Assume 1 = Male, 0 = Female

# Combine inputs into a single array
##([gender_numeric, age, tb, db, alkphos, sgpt, sgot, tp, alb, ag_ratio])
data = {'gender_numeric'=gender_numeric,
        'age' = age,
        'tb' = tb, 
        'db'=db,
        'alkphos' = alkphos, 
        'sgpt'=sgpt, 
        'sgot' = sgot, 
        'tp' = tp, 
        'alb' = alb, 
        'ag_ratio' = ag_ratio}
input_df = pd.Dataframe(data, index = [0])
# input Data
with st.expander('Input Data'):
 st.write('**Input Data**')
 indput_df

# Button for prediction
if st.button("Predict"):
    result = predict(input_data)
    if result == 1:  # Assuming 1 = Disease and 0 = No Disease
        st.error("The patient is classified as having liver disease.")
    else:
        st.success("The patient is classified as healthy.")
 
