# model deployment using streamlit
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import joblib, json, os, sys
from sklearn import set_config
set_config(transform_output='pandas')

# Define the load train or test data function with caching
@st.cache_data
def load_Xy_data(fpath):
    train_path = fpath['data']['ml']['train']
    X_train, y_train =  joblib.load(train_path)
    test_path = fpath['data']['ml']['test']
    X_test, y_test = joblib.load(test_path)
    return X_train, y_train, X_test, y_test
 
# Load the filepaths
FILEPATHS_FILE = 'config/filepaths.json'
with open(FILEPATHS_FILE) as f:
    FPATHS = json.load(f)

# Define the load raw eda data function with caching
@st.cache_data
def load_data(fpath):
    df = pd.read_csv(fpath)
    df = df.set_index("movie_id")
    return df
    
# Define the load train or test data function with caching
@st.cache_data
def load_Xy_data(fpath):
    return joblib.load(fpath)
    
@st.cache_resource
def load_model_ml(fpath):
    return joblib.load(fpath)
    
### Start of App
st.title("IMDB Movies")
# Include the banner image
st.image(FPATHS['images']['banner'])

# Load & cache dataframe
df = load_data(fpath = FPATHS['data']['raw']['full'])
# Load training data
X_train, y_train = load_Xy_data(fpath=FPATHS['data']['ml']['train'])
# Load testing data
X_test, y_test = load_Xy_data(fpath=FPATHS['data']['ml']['test'])
# Load model
linreg = load_model_ml(fpath = FPATHS['models']['linear_regression'])

