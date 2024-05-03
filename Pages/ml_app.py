import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import os
import joblib
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np
import json
import tensorflow as tf
import sys, os
# import custom_functions as fn
from lime.lime_text import LimeTextExplainer

with open('config/filepaths.json') as f:
    FPATHS = json.load(f)
st.title("Predicting IMDB Movie Review Ratings")


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
@st.title("IMDB Movies")

# Include the banner image
@st.image(FPATHS['Images']['banner'])

# # Load & cache dataframe
# df = load_data(fpath = FPATHS['data']['raw']['full'])

# # Load training data
# X_train, y_train = load_Xy_data(fpath=FPATHS['data']['ml']['train'])

# # Load testing data
# X_test, y_test = load_Xy_data(fpath=FPATHS['data']['ml']['test'])

# # Get text to predict from the text input box
# X_to_pred = st.text_input("### Enter text to predict here:", 
#                           value="Great a must See!.")

# Loading the ML model
@st.cache_resource
def load_ml_model(fpath):
    loaded_model = joblib.load(fpath)
    return loaded_model

# Load model from FPATHS
model_fpath = FPATHS['models']['nb']
nb_pipe = load_ml_model(model_fpath)


# load target lookup dict
@st.cache_data
def load_lookup(fpath=FPATHS['data']['ml']['target_lookup']):
    return joblib.load(fpath)
@st.cache_resource
def load_encoder(fpath=FPATHS['data']['ml']['label_encoder'] ):
    return joblib.load(fpath)

# Load the target lookup dictionary
target_lookup = load_lookup()

# Load the encoder
encoder = load_encoder()

# Update the function to decode the prediction
def make_prediction(X_to_pred, nb_pipe=nb_pipe, lookup_dict= target_lookup):
    # Get Prediction
    pred_class = nb_pipe.predict([X_to_pred])[0]
    # Decode label
    pred_class = lookup_dict[pred_class]
    return pred_class

# Trigger prediction with a button
if st.button("Get prediction."):
    pred_class = make_prediction(X_to_pred)
    st.markdown(f"##### Predicted category:  {pred_class}") 
else: 
    st.empty()



@st.cache_resource
def get_explainer(class_names = None):
    lime_explainer = LimeTextExplainer(class_names=class_names)
    return lime_explainer
    
def explain_instance(explainer, X_to_pred, predict_func):
    explanation = explainer.explain_instance(X_to_pred, predict_func)
    return explanation.as_html(predict_proba=False)
# Create the lime explainer
explainer = get_explainer(class_names = encoder.classes_)


## Loading our training and test data
@st.cache_data
def load_Xy_data(joblib_fpath):
    return joblib.load(joblib_fpath)
    
# Load training data from FPATHS
train_data_fpath  = FPATHS['data']['ml']['train']
X_train, y_train = load_Xy_data(train_data_fpath)
# Load test data from FPATHS
test_data_fpath  = FPATHS['data']['ml']['test']
X_test, y_test = load_Xy_data(test_data_fpath)

# ​## To place the 3 checkboxes side-by-side
# col1,col2,col3 = st.columns(3)
# show_train = col1.checkbox("Show training data.", value=True)
# show_test = col2.checkbox("Show test data.", value=True)
# show_model_params =col3.checkbox("Show model params.", value=False)
# if st.button("Show model evaluation."):
#     pass # placeholder

@st.cache_resource
def load_network(fpath):
    model = tf.keras.models.load_model(fpath)
    return model

fpath_model = FPATHS['models']['gru']

## To place the 3 checkboxes side-by-side
col1,col2,col3 = st.columns(3)
show_train = col1.checkbox("Show training data.", value=True)
show_test = col2.checkbox("Show test data.", value=True)

@st.cache_resource
def load_tf_dataset(fpath):
    ds = tf.data.Dataset.load(fpath)
    return ds
    
# Loading train and test ds 
fpath_train_ds = FPATHS['data']['tf']['train_tf']
train_ds = load_tf_dataset(fpath_train_ds)

fpath_test_ds = FPATHS['data']['tf']['test_tf']
test_ds = load_tf_dataset(fpath_test_ds)
# Make prediction
st.sidebar.subheader('Make a prediction')


# Feature inputs
rating = st.sidebar.slider('Rating', min_value=0, max_value=10)
ratings = st.sidebar.radio('Movie Ratings', ['High_Rating', 'Low_Rating'])
original_title = st.sidebar.radio('Movie Title')
review = st.sidebar('Movie Review')
movie_id = st.slider('Movie ID')



explain = st.sidebar.checkbox('Explain Prediction')
predict = st.sidebar.button('Make Prediction')


