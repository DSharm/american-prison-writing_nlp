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

#All these packages need to be installed from pip
import requests #for http requests
import bs4 #called `beautifulsoup4`, an html parser
import docx #reading MS doc files, install as `python-docx`
import urllib.parse #For joining urls

from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager

import time

def scrape_transcribed_data(url="https://apw.dhinitiative.org/node/5",base_url="https://apw.dhinitiative.org",scraping_on = "Relationship to Prison",scraping_attribute='relation_to_prison',limit=None):
    '''
    Main function for scraping transcribed essays from APW websites. Installs chromedriver to automate
    and mimic a user clicking on individual essays and click "View Transcription"

    Calls get_essays_single_page() to get essays on a given page

    Outputs a dataframe containing url, title, text of essay, and the scraping attribute used to iterate
    over essays 
    '''
    dictionary = {scraping_attribute : [], 'url':[],'title':[],'text' : []}

    visit = requests.get(url)
    visit_soup = bs4.BeautifulSoup(visit.text, 'html.parser')
    mysection = visit_soup.body.findAll('li',{'class':'leaf'})

    for li in mysection:
        atag = li.findAll('a')
        for a in atag:
            #print(a.text)
            if a.text==scraping_on:
                relurl = a.get("href")
                attribute = requests.get(urllib.parse.urljoin(base_url,relurl)) 
                #print(base_url)
                visit_attr_page = bs4.BeautifulSoup(attribute.text, 'html.parser')
                page_options = visit_attr_page.findAll('ul',{'class':"islandora-solr-facet-pages-results"})

    #print(page_options)
    driver = webdriver.Chrome(ChromeDriverManager().install())

    for g in page_options:
        # all different options
        lists = g.findAll('li')
        for l in lists:
            atag = l.findAll('a')
            for a in atag:
                print(a.text)
                #print(len(dictionary[scraping_attribute]))
                if limit and len(dictionary[scraping_attribute]) >= limit:
                        print("Exceeding limit")
                        driver.quit()
                        return(pd.DataFrame(dictionary))

                relurl = a.get('href')
                #print(relurl)
                get_a_page = requests.get(urllib.parse.urljoin(base_url,relurl)) 
                visit_a_page = bs4.BeautifulSoup(get_a_page.text, 'html.parser')
                
                # go to last page
                # get the number of the last page
                # loop through range and call get_esssays_single_page
                try:
                    last_page = visit_a_page.find('li',{"class":"pager-last"}).find('a')
                    get_last_page = requests.get(urllib.parse.urljoin(base_url,last_page.get('href')))
                    visit_last_page = bs4.BeautifulSoup(get_last_page.text, 'html.parser')
                    page_num = visit_last_page.find('li',{'class':'pager-current'})
                    last_page = int(page_num.text)
                    print(last_page)

                    i = 0
                    while i < last_page:    
                        #print(i)
                        page_get_essay = visit_a_page.findAll('dd',{'class':"solr-value mods-titleinfo-title-t"})
                        dictionary = get_essays_single_page(a.text,page_get_essay,driver,dictionary)

                        # get next page
                        try:
                            next_page = visit_a_page.find('li',{"class":"pager-next"}).find('a')
                            get_next_page = requests.get(urllib.parse.urljoin(base_url,next_page.get('href')))
                            visit_a_page = bs4.BeautifulSoup(get_next_page.text, 'html.parser')
                            #print(i)
                        except:
                            break
                        i+=1
                except:
                    page_get_essay = visit_a_page.findAll('dd',{'class':"solr-value mods-titleinfo-title-t"})
                    dictionary = get_essays_single_page(a.text,page_get_essay,driver,dictionary,limit,base_url,scraping_attribute)
                    
    driver.quit()
    df = pd.DataFrame(dictionary)
    return df

def get_essays_single_page(category,page_get_essay,driver,dictionary,limit,base_url,scraping_attribute):
    '''
    Helper function for scrape_transcribed_data(). Gets essays for a single page and returns the dictionary
   '''
    for es in page_get_essay:
        es_a_tag = es.findAll('a')
    # start iterating through essays on a single page
        for es_a in es_a_tag:
            relurl = es_a.get('href')
            print(es_a.text)
            #print(len(dictionary[scraping_attribute]))
            #print(limit)
            if limit and len(dictionary[scraping_attribute]) >= limit:
                print('exceeding limit')
                return(dictionary)

            driver.get(urllib.parse.urljoin(base_url,relurl))
            try:
                next_btn = driver.find_element_by_id('transcript_link')
                #print(next_btn)
                next_btn.click()
                time.sleep(8)
                find_transcription = driver.find_element_by_id('webform-ajax-wrapper-transcript')
                div = find_transcription.find_element_by_tag_name('div')
                text = div.text

                dictionary[scraping_attribute].append(category)
                dictionary['url'].append(relurl)
                dictionary['title'].append(es_a.text)
                dictionary['text'].append(text)

                details_btn = driver.find_elements_by_class_name('fieldset-title')
                details_btn[0].click()
                time.sleep(8)

                get_a_page = requests.get(urllib.parse.urljoin(base_url,relurl)) 
                visit_a_page = bs4.BeautifulSoup(get_a_page.text, 'html.parser')
                find_details_title = visit_a_page.findAll('dt')
                find_details = visit_a_page.findAll('dd')
                

            except:
                continue
    return(dictionary)

def get_attributes(scraping_on,scraping_attribute,url="https://apw.dhinitiative.org/node/5",base_url="https://apw.dhinitiative.org",limit=None):
    '''
    Function to get a dataframe with all essay titles and the specified author attribute (e.g. "Ethnicity")
    The output from this function can be merged with dataframe from scrape_transcribed_data()
    '''
    parsDict = {scraping_attribute : [], 'title':[]}

    visit = requests.get(url)
    visit_soup = bs4.BeautifulSoup(visit.text, 'html.parser')
    mysection = visit_soup.body.findAll('li',{'class':'leaf'})

    for li in mysection:
        atag = li.findAll('a')
        for a in atag:
            #print(a.text)
            if a.text==scraping_on:
                relurl = a.get("href")
                attribute = requests.get(urllib.parse.urljoin(base_url,relurl)) 
                #print(base_url)
                visit_attr_page = bs4.BeautifulSoup(attribute.text, 'html.parser')
                page_options = visit_attr_page.findAll('ul',{'class':"islandora-solr-facet-pages-results"})


    #for g in rel_page_options:
    for g in page_options:
        # all different gender options
        lists = g.findAll('li')
        for l in lists:
            # each of the options (female, male, etc)
            #print(l.text)
            atag = l.findAll('a')
            for a in atag:
                print(a.text)
                if limit and len(parsDict[scraping_attribute]) >= limit:
                    print('exceeding limit')
                    df = pd.DataFrame(dict([ (k,pd.Series(v)) for k,v in parsDict.items() ]))
                    df = pd.DataFrame(df.groupby('title')['ethnicity'].sum().reset_index())
                    return(df)
                relurl = a.get('href')
                get_a_page = requests.get(urllib.parse.urljoin(base_url,relurl)) 
                visit_a_page = bs4.BeautifulSoup(get_a_page.text, 'html.parser')
                
                # go to last page
                # get the number of the last page
                # loop through range and call get_esssays_single_page
                try:
                    last_page = visit_a_page.find('li',{"class":"pager-last"}).find('a')
                    get_last_page = requests.get(urllib.parse.urljoin(base_url,last_page.get('href')))
                    visit_last_page = bs4.BeautifulSoup(get_last_page.text, 'html.parser')
                    page_num = visit_last_page.find('li',{'class':'pager-current'})
                    last_page = int(page_num.text)
                    print(last_page)

                    i = 0
                    while i < last_page:    
                        #print(i)
                        page_get_essay = visit_a_page.findAll('dd',{'class':"solr-value mods-titleinfo-title-t"})
                        #parsDict = get_essays_single_page(a.text,page_get_essay,driver,parsDict)
                        for es in page_get_essay:
                            es_a_tag = es.findAll('a')
                            # start iterating through essays on a single page
                            for es_a in es_a_tag:
                                relurl = es_a.get('href')
                                print(es_a.text)
                                parsDict[scraping_attribute].append(a.text)
                                parsDict['title'].append(es_a.text)


                        # get next page
                        try:
                            next_page = visit_a_page.find('li',{"class":"pager-next"}).find('a')
                            get_next_page = requests.get(urllib.parse.urljoin(base_url,next_page.get('href')))
                            visit_a_page = bs4.BeautifulSoup(get_next_page.text, 'html.parser')
                            #print(i)
                        except:
                            break
                        i+=1
                except:
                    page_get_essay = visit_a_page.findAll('dd',{'class':"solr-value mods-titleinfo-title-t"})
                    #parsDict = get_essays_single_page(a.text,page_get_essay,driver,parsDict)
                    for es in page_get_essay:
                            es_a_tag = es.findAll('a')
                            # start iterating through essays on a single page
                            for es_a in es_a_tag:
                                relurl = es_a.get('href')
                                print(es_a.text)
                                parsDict[scraping_attribute].append(a.text)
                                parsDict['title'].append(es_a.text)
    
    df = pd.DataFrame(dict([ (k,pd.Series(v)) for k,v in parsDict.items() ]))
    df = pd.DataFrame(df.groupby('title')['ethnicity'].sum().reset_index())
    
    return(df)


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
    '''
    Helper function to clean up group categories such as race, sex, etc
    '''
    
    df[new_var] = df[original_var]
    df.loc[~df[original_var].isin(list_of_cats),new_var]= "Other"

    return df

def norm_text(df):
    '''
    Tokenizes and normalizes text data.  Returns df with new columns "tokenized_text" and "normalized_text"
    '''
    # Tokenized and normalized texts
    df['tokenized_text'] = df['text'].apply(lambda x: lucem_illud_2020.word_tokenize(x))
    df['normalized_text'] = df['tokenized_text'].apply(lambda x: lucem_illud_2020.normalizeTokens(x))

    return df

def norm_sent(df):
    '''
    Tokenizes and normalizes sentences. Returns df with new columns "tokenized_sents" and "normalized_sents"
    '''
    # Tokenized and normalized sents
    df['tokenized_sents'] = df['text'].apply(lambda x: [lucem_illud_2020.word_tokenize(s) for s in lucem_illud_2020.sent_tokenize(x)])
    df['normalized_sents'] = df['tokenized_sents'].apply(lambda x: [lucem_illud_2020.normalizeTokens(s, lemma=False) for s in x])

    return df

def norm_words(df):
    '''
    Tokenizes and normalizes words. Differs from norm_text() because it does not lemmatize
    the words. Used in word embedding models.  Returns df with new columns "tokenized_words" and "normalized_words"
    '''
    df['tokenized_words'] = df['text'].apply(lambda x: lucem_illud_2020.word_tokenize(x))
    df['normalized_words'] = df['tokenized_words'].apply(lambda x: lucem_illud_2020.normalizeTokens(x, lemma=False))

    return df