import streamlit as st
# Add title
st.title("IMDB Movies")
import streamlit as st
import pandas as pd
import numpy as np
import joblib, json, os, sys
from sklearn import set_config
set_config(transform_output='pandas')
# Load the filepaths
FILEPATHS_FILE = 'config/filepaths.json'
with open(FILEPATHS_FILE) as f:
    FPATHS = json.load(f)
    
# Define the load raw eda data function with caching
@st.cache_data
def load_data(fpath):
    df = pd.read_csv(fpath)
    df = df.set_index("PID")
    return df
    
# Define the load train or test data function with caching
@st.cache_data
def load_Xy_data(fpath):
    return joblib.load(fpath)
    
@st.cache_resource
def load_model_ml(fpath):
    return joblib.load(fpath)
    
### Start of App
st.title('House Prices in Ames, Iowa')
# Include the banner image
st.image(FPATHS['images']['banner'])