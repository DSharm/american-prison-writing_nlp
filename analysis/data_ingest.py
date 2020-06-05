'''
Data Ingest.
Note, a lot of code is heavily borrowed and adapted from materials provided by the course staff
of Content Analysis
'''
#Special module written for this class
#This provides access to data and to helper functions from previous weeks
import lucem_illud_2020 #pip install git+git://github.com/Computational-Content-Analysis-2020/lucem_illud_2020.git

import pandas as pd #gives us DataFrames
import re #for regexs
import numpy as np #For divergences/distances

# #All these packages need to be installed from pip
# import requests #for http requests
# import matplotlib.pyplot as plt #For graphics
# import wordcloud #Makes word clouds
# import scipy #For divergences/distances
# import seaborn as sns #makes our plots look nicer
# import sklearn.manifold #For a manifold plot
# import json #For API responses
# import urllib.parse #For joining urls
# import bs4 #called `beautifulsoup4`, an html parser

# # comp-linguistics
# import spacy

# #Displays the graphs
# import graphviz #You also need to install the command line graphviz

# #These are from the standard library
# import os.path
# import zipfile
# import subprocess
# import io
# import tempfile

# #These come with Python
# import urllib.parse #For joining urls
# import io #for making http requests look like files
# import json #For Tumblr API responses
# import os.path #For checking if files exist
# import os #For making directories

def load_prep_data(file):
    '''
    Loads data, performs regex functions to clean it up, and creates
    tokenized text, normalized text, and normalized sentence columns.

    Assumes  text saved in column called 'text'
    '''
    df = pd.read_csv(file)
    df['text'] =  [re.sub(r'[\n]+',' ', str(x)) for x in df['text']]
    df['text'] =  [re.sub(r'[//]+',' ', str(x)) for x in df['text']]
    df['text'] =  [re.sub(r'[\\]','', str(x)) for x in df['text']]
    df['text'] =  [re.sub(r'[^a-zA-Z0-9 \']',' ', str(x)) for x in df['text']]

    return df

def agg_groups(df, original_var,new_var,list_of_cats):
    
    df[new_var] = df[original_var]
    df.loc[~df[original_var].isin(list_of_cats),new_var]= "Other"

    return df

def norm_text(df):
    # Tokenized and normalized texts
    df['tokenized_text'] = df['text'].apply(lambda x: lucem_illud_2020.word_tokenize(x))
    df['normalized_text'] = df['tokenized_text'].apply(lambda x: lucem_illud_2020.normalizeTokens(x))

    return df

def norm_sent(df):
    # Tokenized and normalized sents
    df['tokenized_sents'] = df['text'].apply(lambda x: [lucem_illud_2020.word_tokenize(s) for s in lucem_illud_2020.sent_tokenize(x)])
    df['normalized_sents'] = df['tokenized_sents'].apply(lambda x: [lucem_illud_2020.normalizeTokens(s, lemma=False) for s in x])

    return df
