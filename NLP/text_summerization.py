#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 10 18:22:11 2020

@author: praveen
"""

import bs4 as bs
import urllib.request
import re
import nltk
nltk.download('stopwords')
import heapq

# Gettings the data source
source = urllib.request.urlopen('https://en.wikipedia.org/wiki/Global_warming').read()

# Parsing the data/ creating BeautifulSoup object
soup = bs.BeautifulSoup(source,'lxml')

# Fetching the data
text = ""
for paragraph in soup.find_all('p'):
    text += paragraph.text
    
# Preprocessing the data
text = re.sub(r'\[[0-9]*\]',' ',text)
text = re.sub(r'\s+',' ',text)
text = re.sub(r'^\s+','',text)
clean_text = text.lower()
clean_text = re.sub(r'\W',' ',clean_text)
clean_text = re.sub(r'\d',' ',clean_text)
clean_text = re.sub(r'\s+',' ',clean_text)


sentences = nltk.sent_tokenize(text)

stop_words = nltk.corpus.stopwords.words('english')

word2count = {}

for word in nltk.word_tokenize(clean_text):
    if word not in stop_words:
        if word not in word2count.keys():
            word2count[word] = 1
        else:
            word2count[word] += 1
            
for key in word2count.keys():
    word2count[key] = word2count[key]/max(word2count.values())

#print(word2count)
# Product sentence scores    
sent2score = {}
for sentence in sentences:
    for word in nltk.word_tokenize(sentence.lower()):
        if word in word2count.keys():
            if len(sentence.split(' ')) < 25:
                if sentence not in sent2score.keys():
                    sent2score[sentence] = word2count[word]
                else:
                    sent2score[sentence] += word2count[word]
 
#print(sent2score)                   
# Gettings best 5 lines             
best_sentences = heapq.nlargest(10, sent2score, key=sent2score.get)

print('---------------------------------------------------------')
for sentence in best_sentences:
    pass
    print(sentence)

        