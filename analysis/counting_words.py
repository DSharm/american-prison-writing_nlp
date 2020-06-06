'''
Helper functions to conduct word counting and divergence analysis
'''
#Special module written for this class
#This provides access to data and to helper functions from previous weeks
import lucem_illud_2020 #pip install git+git://github.com/Computational-Content-Analysis-2020/lucem_illud_2020.git

import pandas as pd #gives us DataFrames
import re #for regexs
import numpy as np #For divergences/distances

#All these packages need to be installed from pip
import requests #for http requests
import matplotlib.pyplot as plt #For graphics
import wordcloud #Makes word clouds
import scipy #For divergences/distances
import seaborn as sns #makes our plots look nicer
import sklearn.manifold #For a manifold plot
import json #For API responses
import urllib.parse #For joining urls
import bs4 #called `beautifulsoup4`, an html parser

# comp-linguistics
import spacy

#Displays the graphs
import graphviz #You also need to install the command line graphviz

#These are from the standard library
import os.path
import zipfile
import subprocess
import io
import tempfile

#These come with Python
import urllib.parse #For joining urls
import io #for making http requests look like files
import json #For Tumblr API responses
import os.path #For checking if files exist
import os #For making directories

import nltk


def wordCounter(normalized_text):
    wordLst = normalized_text.sum()
    wordCounts = {}
    for word in wordLst:
        #We usually need to normalize the case
        wLower = word.lower()
        if wLower in wordCounts:
            wordCounts[wLower] += 1
        else:
            wordCounts[wLower] = 1
    #convert to DataFrame
    countsForFrame = {'word' : [], 'count' : []}
    for w, c in wordCounts.items():
        countsForFrame['word'].append(w)
        countsForFrame['count'].append(c)
    
    df = pd.DataFrame(countsForFrame)
    df.sort_values(by="count", ascending = False, inplace= True)
    return df


def word_cloud(normalized_text):
    wc = wordcloud.WordCloud(background_color="white", max_words=500, width= 1000, height = 1000, mode ='RGBA', scale=.5).generate(' '.join(normalized_text.sum()))
    plt.imshow(wc)
    plt.axis("off")
    #plt.savefig("whitehouse_word_cloud.pdf", format = 'pdf')

def make_group_corpora(df,group_col,text_col='normalized_text'):

    nlp = spacy.load("en")
    # aggregate texts by group:
    df_agg  = df.groupby(group_col).size().reset_index(name='count')
    df_agg_text = pd.DataFrame(df.groupby(group_col)[text_col].sum()).reset_index()

    df2 = df_agg_text.merge(df_agg)
    df2.sort_values(by='count', inplace=True, ascending=False)
    df2 = df2[:20]
    fileids = list(df2[group_col])
    print(fileids)
    corpora = []
    for index, row in df2.iterrows():
        corpora.append(row[text_col])

    
    return fileids, corpora, df2[[group_col,'count']]

def kl_divergence(X, Y):
    P = X.copy()
    Q = Y.copy()
    P.columns = ['P']
    Q.columns = ['Q']
    df = Q.join(P).fillna(0)
    p = df.iloc[:,1]
    q = df.iloc[:,0]
    D_kl = scipy.stats.entropy(p, q)
    return D_kl

def chi2_divergence(X,Y):
    P = X.copy()
    Q = Y.copy()
    P.columns = ['P']
    Q.columns = ['Q']
    df = Q.join(P).fillna(0)
    p = df.iloc[:,1]
    q = df.iloc[:,0]
    return scipy.stats.chisquare(p, q).statistic

def Divergence(corpus1, corpus2, difference="KL"):
    """Difference parameter can equal KL, Chi2, or Wass"""
    freqP = nltk.FreqDist(corpus1)
    P = pd.DataFrame(list(freqP.values()), columns = ['frequency'], index = list(freqP.keys()))
    freqQ = nltk.FreqDist(corpus2)
    Q = pd.DataFrame(list(freqQ.values()), columns = ['frequency'], index = list(freqQ.keys()))
    if difference == "KL":
        return kl_divergence(P, Q)
    elif difference == "Chi2":
        return chi2_divergence(P, Q)
    elif difference == "KS":
        try:
            return scipy.stats.ks_2samp(P['frequency'], Q['frequency']).statistic
        except:
            return scipy.stats.ks_2samp(P['frequency'], Q['frequency'])
    elif difference == "Wasserstein":
        try:
            return scipy.stats.wasserstein_distance(P['frequency'], Q['frequency'], u_weights=None, v_weights=None).statistic
        except:
            return scipy.stats.wasserstein_distance(P['frequency'], Q['frequency'], u_weights=None, v_weights=None)



def make_heat_map(fileids,corpora,divergence_type='KL'):
    L = []
    for p in corpora:
        l = []
        for q in corpora:
            l.append(Divergence(p,q, difference = divergence_type))
        L.append(l)
    M = np.array(L)
    fig = plt.figure()
    div = pd.DataFrame(M, columns = fileids, index = fileids)
    ax = sns.heatmap(div)
    print("Divergence Type:", divergence_type)
    plt.show()

def prep_classification_data(df,category_col,keep=[],true_cat='',holdOut=0.2):
    df['category'] = df[category_col]

    # binary race
    df = df[df['category'].isin(keep)]

    # T/F category
    df['category'] = [s == true_cat for s in df[category_col]]

    train_data_df, test_data_df = lucem_illud_2020.trainTestSplit(df, holdBackFraction=holdOut)

    TFVectorizer = sklearn.feature_extraction.text.TfidfVectorizer(max_df=100, min_df=2, stop_words='english', norm='l2')
    TFVects = TFVectorizer.fit_transform(train_data_df['text'])

    train_data_df['vect'] = [np.array(v).flatten() for v in TFVects.todense()]

    #Create vectors
    TFVects_test = TFVectorizer.transform(test_data_df['text'])
    test_data_df['vect'] = [np.array(v).flatten() for v in TFVects_test.todense()]

    return train_data_df, test_data_df

def classification(train_data_df,test_data_df,classifier):

    if classifier=="LogisticRegression":
        clf = sklearn.linear_model.LogisticRegression(penalty='l2')

    if classifier == "naiveBayes":
        clf = sklearn.naive_bayes.BernoulliNB()

    if classifier == "bag":
        clf = sklearn.ensemble.BaggingClassifier(tree, n_estimators=100, max_samples=0.8, random_state=1) 
    
    if classifier == "svm":
        clf = sklearn.svm.SVC(kernel='linear', probability = False)
    
    if classifier == "nn":
        clf = sklearn.neural_network.MLPClassifier()
    
    clf.fit(np.stack(train_data_df['vect'], axis=0), train_data_df['category'])

    print("Training Accuracy:")
    print(clf.score(np.stack(train_data_df['vect'], axis=0), train_data_df['category']))
    print("Testing Accuracy:")
    print(clf.score(np.stack(test_data_df['vect'], axis=0), test_data_df['category']))

    return clf

def evaluation(classifier, test_data_df,true_cat):
    # predict
    test_data_df['predict'] = classifier.predict(np.stack(test_data_df['vect'], axis=0))

    # precision, recall, f1 score
    
    print("Precision:")
    print(sklearn.metrics.precision_score(test_data_df['category'], test_data_df['predict']))
    print("Recall:")
    print(sklearn.metrics.recall_score(test_data_df['category'], test_data_df['predict']))
    print("F1 Score:")
    print(sklearn.metrics.f1_score(test_data_df['category'], test_data_df['predict']))

    print("True Category is:",true_cat)
    lucem_illud_2020.plotMultiROC(classifier, test_data_df)
    lucem_illud_2020.plotConfusionMatrix(classifier, test_data_df)
    print(lucem_illud_2020.evaluateClassifier(classifier, test_data_df))   

