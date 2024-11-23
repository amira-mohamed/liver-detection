import streamlit as st
import pandas as pd
import joblib
import numpy as np


# Load the trained model
#model = joblib.load("Model_liver.pkl")  # Replace with your model file

# Function to make predictions


# Page Title
st.title("liver disease classification")
st.info('This App for Liver Detection predicition')

# Load Data
with st.expander('Data')
 st.write('Raw Data')
 df = pd.read_csv('liver_dataset.csv')
 df
