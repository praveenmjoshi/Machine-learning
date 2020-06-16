#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 10 10:13:00 2020

@author: praveen
"""

#Importing Libraries

import numpy as np
import pandas as pd
import re
import pickle

import nltk
from nltk.corpus import stopwords
from sklearn.datasets import load_files
nltk.download('stopwords')


#Importing Dataset

reviews = load_files('txt_sentoken/')
X , y = reviews.data , reviews.target

"""
Above way of reading files is time consuming 
So once after reading data, we can store X and y as pickle files 
and next time onwards we can read from pickle files as it is quick

"""


with open('X.pickle','wb') as f:
    pickle.dump(X,f)

with open('y.pickle','wb') as f:
    pickle.dump(y,f)  
    
    
"""
cross checking pickle files by reading X and y

"""

with open('X.pickle','rb') as f:
    _X = pickle.load(f)

with open('y.pickle','rb') as f:
    _y = pickle.load(f)  



corpus = []
for i in range(0,len(X)):
    review = re.sub(r'\W',' ',str(X[i]))
    review = review.lower()
    review = re.sub(r'\s+[a-z]\s+',' ',review) #remove single character sorrounded by one or more space
    review = re.sub(r'^[a-z]\s+','',review) # remove single characters at start of sentence
    review = re.sub(r'\s+',' ',review) #remove white space
    corpus.append(review)
    
    
from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer(max_features=2000, min_df = 3 , max_df = 0.6, stop_words = stopwords.words('english'))


"""
max_features 2000 ->  considering 2000 most appeared words
min_df 3 -> consider words if it exist in minimum 3 documents
max_df 0.6 -> exclude words which appear in 60 percent of more documents
stopwords -> Exclude stopwords

"""

X = vectorizer.fit_transform(corpus).toarray()


"""
Converting above BOW vector into TF-IDF vectore using TfidfTransformer
"""

from sklearn.feature_extraction.text import TfidfTransformer
transformer = TfidfTransformer()
X = transformer.fit_transform(X).toarray()


"""
Instead of creating count vectorizer and then transforming to tdidf vector , we can directly create tfidf vector 
"""
# Creating the Tf-Idf model directly
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(max_features = 2000, min_df = 3, max_df = 0.6, stop_words = stopwords.words('english'))
X = vectorizer.fit_transform(corpus).toarray()

"""
Splitting data into train and test data
"""
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=0)

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)


"""
Storing the model as Pickle file 
"""

with open('classifier.pickle','wb') as f:
    pickle.dump(classifier,f)
    
with open('tfidfvectorizer.pickle','wb') as f:
    pickle.dump(vectorizer,f)
    
    
"""
Importing above created pickles to classify unseen text using trained model
"""

with open('classifier.pickle','rb') as f:
   clf =  pickle.load(f)
    
with open('tfidfvectorizer.pickle','rb') as f:
   tfidf_vec =  pickle.load(f)

sampleText = ["You are a greate person, God bless you"]
sampleText = tfidf_vec.transform(sampleText).toarray()
print(clf.predict(sampleText))


sampleText2 = ["You are very bad man"]
sampleText2 = tfidf_vec.transform(sampleText2).toarray()
print(clf.predict(sampleText2))