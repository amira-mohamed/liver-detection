

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
df = pd.read_csv('https://github.com/amira-mohamed/liver-detection/blob/master/liver_dataset.csv')
df
