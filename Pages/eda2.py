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
from wordcloud import WordCloud
from nltk import casual_tokenize
import scattertext

# Changing the Layout
st.set_page_config( layout="wide")


with open('config/filepaths.json') as f:
    FPATHS = json.load(f)

st.title("Exploratory Data Analysis of IMDB Reviews")

@st.cache_data
def load_data(fpath):
    return joblib.load(fpath)


df = load_data(FPATHS['data']['processed'])
df.head(2)

st.image(fpath_wc)

import pandas as pd
import nltk
def get_ngram_measures_finder(tokens, ngrams=2, measure='raw_freq', top_n=None, min_freq = 1,
                             words_colname='Words'):
    import nltk
    if ngrams == 4:
        MeasuresClass = nltk.collocations.QuadgramAssocMeasures
        FinderClass = nltk.collocations.QuadgramCollocationFinder
        
    elif ngrams == 3: 
        MeasuresClass = nltk.collocations.TrigramAssocMeasures
        FinderClass = nltk.collocations.TrigramCollocationFinder
    else:
        MeasuresClass = nltk.collocations.BigramAssocMeasures
        FinderClass = nltk.collocations.BigramCollocationFinder

    measures = MeasuresClass()
    
   
    finder = FinderClass.from_words(tokens)
    finder.apply_freq_filter(min_freq)
    if measure=='pmi':
        scored_ngrams = finder.score_ngrams(measures.pmi)
    else:
        measure='raw_freq'
        scored_ngrams = finder.score_ngrams(measures.raw_freq)

    df_ngrams = pd.DataFrame(scored_ngrams, columns=[words_colname, measure.replace("_",' ').title()])
    if top_n is not None:
        return df_ngrams.head(top_n)
    else:
        return df_ngrams

st.subheader("n-grams")
# Making 2 columns, one for controls one for dataframe
col1, col2 = st.columns([0.3,0.7])
# Making a widget for ngrams arg
ngram_type = col1.radio("Type of n-gram", options= [2,3,4], index=1, horizontal=True)
# Making a widget for the measures arg
measure_type = col1.selectbox('Measure to Compare', options=['raw_freq','pmi'], index=0)
# Making a widget for the top_n:
top_n = col1.slider("Top # of ngrams", min_value=5, max_value=50,value=20,)
# Making a widget for the top_n:
min_freq = col1.slider("Frequency Filter", min_value=1, max_value=50,value=3)


try:
    st.dataframe(df_compare_bigrams)
except Exception as e:
    display(e)