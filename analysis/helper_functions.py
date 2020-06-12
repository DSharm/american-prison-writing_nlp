'''
Helper functions to conduct word counting, divergence, classification, and word embedding analyses
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
import gensim

def wordCounter(normalized_text):
    '''
    Takes a column of normalized texts and outputs a dataframe counting words
    '''
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
    '''
    Takes a column of normalized texts and outputs a wordcloud as well as saving the figure in directory
    '''

    wc = wordcloud.WordCloud(background_color="white", max_words=500, width= 1000, height = 1000, mode ='RGBA', scale=.5).generate(' '.join(normalized_text.sum()))
    plt.imshow(wc)
    plt.axis("off")
    plt.savefig("wordcloud.png", format = 'png')

def make_group_corpora(df,group_col,text_col='normalized_text'):
    '''
    Takes normalized text and aggregates by group column. Used to prep data
    to look at divergence in text by group, e.g. race
    '''
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
    '''
    Function carrying out KL divergence calculation between two vectors
    '''
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
    '''
    Function carrying out Chi2 divergence calculation between two vectors
    '''
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
    '''
    Given a type of divergence, corpora (with group aggregated text), and ids identifying the
    groups, this function outputs heatmaps showing divergence by group
    '''
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
    '''
    Preps data for classification by turning categories into binary (e.g. male and female),
    doing a train/test split, and creating a TF-IDF matrix. Returns train and test data
    '''

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
    '''
    Carries out classification using the specified classifier. Returns the fitted model
    '''

    if classifier=="LogisticRegression":
        clf = sklearn.linear_model.LogisticRegression(penalty='l2')

    if classifier == "naiveBayes":
        clf = sklearn.naive_bayes.BernoulliNB()

    if classifier == "bag":
        tree = sklearn.tree.DecisionTreeClassifier(max_depth=10) #Create an instance of our decision tree classifier.
        clf = sklearn.ensemble.BaggingClassifier(tree, n_estimators=100, max_samples=0.8, random_state=1) 
    
    if classifier == "SVM":
        clf = sklearn.svm.SVC(kernel='linear', probability = False)
    
    if classifier == "NeuralNet":
        clf = sklearn.neural_network.MLPClassifier()
    
    clf.fit(np.stack(train_data_df['vect'], axis=0), train_data_df['category'])
    print(classifier)
    print("Training Accuracy:")
    print(clf.score(np.stack(train_data_df['vect'], axis=0), train_data_df['category']))
    print("Testing Accuracy:")
    print(clf.score(np.stack(test_data_df['vect'], axis=0), test_data_df['category']))
    print("\n")

    return clf

def evaluation(classifier_name, classifier, test_data_df,true_cat):
    '''
    Using the fitted model, predicts on test data and computes evaluation metrics
    '''

    # predict
    test_data_df['predict'] = classifier.predict(np.stack(test_data_df['vect'], axis=0))

    # precision, recall, f1 score
    print(classifier_name)
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

def word_2_vec(df,normalized_sent_col='normalized_sents'):
    '''
    Using the gensim implementation of Word2vec, trains a w2v model and returns the output
    '''
    W2V = gensim.models.word2vec.Word2Vec(df[normalized_sent_col].sum())

    return W2V

def visualize_W2V(W2V,numWords=70,n_components=70):
    '''
    Takes a w2v model as input and outputs a visualization after carrying out PCA
    '''
    # Visualization #2 of first word2vec
    targetWords = W2V.wv.index2word[:numWords]

    wordsSubMatrix = []
    for word in targetWords:
        wordsSubMatrix.append(W2V[word])
    wordsSubMatrix = np.array(wordsSubMatrix)
    wordsSubMatrix

    pcaWords = sklearn.decomposition.PCA(n_components = n_components).fit(wordsSubMatrix)
    reducedPCA_data = pcaWords.transform(wordsSubMatrix)
    #T-SNE is theoretically better, but you should experiment
    tsneWords = sklearn.manifold.TSNE(n_components = 2).fit_transform(reducedPCA_data)

    fig = plt.figure(figsize = (10,6))
    ax = fig.add_subplot(111)
    ax.set_frame_on(False)
    plt.scatter(tsneWords[:, 0], tsneWords[:, 1], alpha = 0)#Making the points invisible 
    for i, word in enumerate(targetWords):
        ax.annotate(word, (tsneWords[:, 0][i],tsneWords[:, 1][i]), size =  20 * (numWords - i) / numWords)
    plt.xticks(())
    plt.yticks(())
    plt.savefig("w2v_viz.png", format = 'png')
    plt.show()


def most_similar_table(W2V,list_words):
    '''
    Given a W2v model and a list of words, outputs a dataframe looking at the words most similar
    to the given word
    '''
    dict_similar = {}
    for word in list_words:
        dict_similar[word] = [(x[0],round(x[1],2)) for x in W2V.most_similar(word)]
    
    dict_df = pd.DataFrame(dict([ (k,pd.Series(v)) for k,v in dict_similar.items() ]))
    
    return dict_df

def most_similar_analogy(positive,compare1,compare2,W2V):
    '''
    Given a word and two other words to compare (e.g. black and white),
    outputs a dataframe containing the words most similar to positive+compare1-compare2
    and most similar to positive+compare2-compare1
    '''
    dict_all = {}
    dict_1 = {}
    dict_1['positive']=positive + compare1
    dict_1['negative']=compare2
    d1_name = '+'.join(dict_1['positive']) + '-' + '-'.join(dict_1['negative'])
    dict_all[d1_name] = W2V.most_similar(dict_1['positive'],dict_1['negative'])

    dict_2 = {}
    dict_2['positive']=positive + compare2
    dict_2['negative']=compare1
    d2_name = '+'.join(dict_2['positive']) + '-' + '-'.join(dict_2['negative'])
    dict_all[d2_name] = W2V.most_similar(dict_2['positive'],dict_2['negative'])

    dict_df = pd.DataFrame(dict([ (k,pd.Series(v)) for k,v in dict_all.items() ]))

    return dict_df 

def d2v(df,keywords,tagged_col='TaggedTexts',norm_word_col='normalized_words',title_col='title',size=100):
    '''
    Given a list of keywords, tags texts with key words and trains a doc2vec model, returning
    the model object
    '''
    taggedDocs = []
    for index, row in df.iterrows():
        #Just doing a simple keyword assignment
        docKeywords = [s for s in keywords if s in row[norm_word_col]]
        #print(docKeywords)
        docKeywords.append(row[title_col])
        taggedDocs.append(gensim.models.doc2vec.LabeledSentence(words = row[norm_word_col], tags = docKeywords))
    df[tagged_col] = taggedDocs

    D2V = gensim.models.doc2vec.Doc2Vec(df[tagged_col], size = size) #Limiting to 100 dimensions

    return D2V 


def d2v_similar_heatmap(D2V,df,equation1,equation2,title_col='title'):
    '''
    Given a d2v model, dataframe, and equations, returns top 10 documents most similar
    to each equation and creates heat map comparing these two sets of documents.

    '''

    eq1 = D2V.docvecs.most_similar(equation1, topn=10 )
    list1 = [x[0] for x in eq1]
    #print(list1)
    eq2 = D2V.docvecs.most_similar(equation2, topn=10 )
    list2 = [x[0] for x in eq2]
    #print(list2)
    targetDocs1 = df[df[title_col].isin(list1)][title_col]
    targetDocs2 = df[df[title_col].isin(list2)][title_col]
    #print(targetDocs1)
    heatmap_doc_similar(targetDocs1,targetDocs2,D2V)
    #heatmap_doc_similar(targetDocs2,D2V)
    return targetDocs1, targetDocs2

def heatmap_doc_similar(cat1,targetDocs1,cat2,targetDocs2,d2v):
    '''
    Constructs heat map showing cosine similarity between two sets of documents for
    two categories, given a d2v model object
    '''
    heatmapMatrixD = []

    for tagOuter in targetDocs1:
        column = []
        tagVec = d2v.docvecs[tagOuter].reshape(1, -1)
        for tagInner in targetDocs2:
            column.append(sklearn.metrics.pairwise.cosine_similarity(tagVec, d2v.docvecs[tagInner].reshape(1, -1))[0][0])
        heatmapMatrixD.append(column)
    heatmapMatrixD = np.array(heatmapMatrixD)
    
    fig, ax = plt.subplots()
    hmap = ax.pcolor(heatmapMatrixD, cmap='terrain')
    cbar = plt.colorbar(hmap)

    cbar.set_label('cosine similarity', rotation=270)
    a = ax.set_xticks(np.arange(heatmapMatrixD.shape[1]) + 0.5, minor=False)
    a = ax.set_yticks(np.arange(heatmapMatrixD.shape[0]) + 0.5, minor=False)
    a = ax.set_ylabel(cat2)
    a = ax.set_xlabel(cat1)
    a = ax.set_xticklabels(targetDocs1, minor=False, rotation=270)
    a = ax.set_yticklabels(targetDocs2, minor=False)




