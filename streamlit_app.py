import streamlit as st
import pandas as pd
import joblib
import numpy as np
from xgboost import XGBClassifier


try:
    get_ipython().run_line_magic('matplotlib', 'inline')
except NameError:
    pass  # Skip if get_ipython is not defined

import warnings
warnings.filterwarnings('ignore')



#------------------------------------------------------------------
# Page Title
st.title("liver disease classification")
st.info('This App for Liver Detection predicition')

# Load Data
with st.expander('Data'):
 st.write('Raw Data')
 df = pd.read_csv('liver_dataset.csv')
 df

# create page columns 
col1, col2, col3, col4, col5, col6= st.columns(6)

# Inputs

age = col1.number_input("Age", min_value=0, max_value=120, step=1)
gender = col2.selectbox("Gender", ["Male", "Female"])
tb = col3.number_input("TB", min_value=0.0, max_value=10.0, step=0.1)
alkphos = col4.number_input("Alkphos", min_value=0, max_value=1000, step=1)
sgpt = col5.number_input("Sgpt", min_value=0, max_value=100, step=1)
ag_ratio = col6.number_input("A/G Ratio", min_value=0.0, max_value=5.0, step=0.1)

# Process gender input (if needed for your model)
gender = 1 if gender == "Male" else 2  # Assume 1 = Male, 0 = Female

#---------------------------------------------------------------------------------
# Load Dataset

df = pd.read_csv('liver_dataset.csv')

df.columns = df.columns.map(str.lower)                            

# replacing missing values with mean
df['a/g ratio'].fillna(df['a/g ratio'].mean(), inplace=True)

df.drop(['db', 'sgot', 'tp', 'alb'], axis=1, inplace=True)

skewed_cols = ['a/g ratio','tb', 'alkphos', 'sgpt']

for c in skewed_cols:
    df[c] = df[c].apply('log1p')

from sklearn.preprocessing import RobustScaler
rs = RobustScaler()
for c in df[['age', 'gender', 'tb', 'alkphos', 'sgpt', 'a/g ratio']].columns:
    df[c] = rs.fit_transform(df[c].values.reshape(-1, 1))

from sklearn.utils import resample
#df['class'].value_counts()

minority = df[df['class'] ==0]
majority = df[df['class']==1]

minority_upsample = resample(minority, replace=True, n_samples=majority.shape[0])

df = pd.concat([minority_upsample, majority], axis=0)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df.drop('class', axis=1), df['class'], test_size=0.05, random_state=123)

from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

#----------------------------------------------------------------
# Combine inputs into a single array

data = {"age" : [age],
        "gender": [gender],
        "tb" : [tb], 
        "alkphos" : [alkphos], 
        "sgpt": [sgpt], 
        "a/g ratio" : [ag_ratio]}

input_df = pd.DataFrame(data, index = [0])

from xgboost import XGBClassifier

model = XGBClassifier(random_state=123)
model.fit(X_train, y_train)
#y_train_hat = model.predict(X_train)
y_test_hat = model.predict(input_df)

# input Data
with st.expander('Input Data'):
 st.write('**Input Data**')
 input_df

st.info(y_test_hat)
# Button for prediction

if st.button("Predict"):
    result = y_test_hat
    if result == 1:  # Assuming 1 = Disease and 0 = No Disease
        st.error("The patient is classified as having liver disease.")
    else:
        st.success("The patient is classified as healthy.")
 
