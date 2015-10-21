import numpy as np
import re
import nltk
import string
import os

from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from operator import itemgetter
from nltk.stem.porter import PorterStemmer

categories = ['comp.graphics']
news = fetch_20newsgroups(categories = categories, remove = ('headers', 'footers', 'quotes'))

for i in range(0, len(news.data)) :
	news.data[i] = re.sub(r'[^a-zA-Z0-9\s]', '', news.data[i])

stemmer = PorterStemmer()

def stem_tokens(tokens, stemmer):
	stemmed = []
	for item in tokens:
		stemmed.append(stemmer.stem(item))
	return stemmed

def StemTokenizer(text):
	tokens = nltk.word_tokenize(text)
	stems = stem_tokens(tokens, stemmer)
	return stems
vectorizer = TfidfVectorizer(stop_words='english',tokenizer=StemTokenizer , norm=None, smooth_idf=False)
#vectorizer = TfidfVectorizer(stop_words='english' , norm=None, smooth_idf=False)

vectors = vectorizer.fit_transform(news.data[:2])

wordlength = len(vectorizer.vocabulary_)
numofdocuments = len(news.data[:2])

key_list = vectorizer.get_feature_names()
results = vectors.toarray()

print key_list
print results

biggest = 0.0
smallest = 0.0
big_row =0
big_col =0

for i in range(0, numofdocuments):
	for j in range(0, wordlength):
		if results[i][j] > biggest :
			biggest = results[i][j]
			big_col = j
			big_row = i
		elif results[i][j] < smallest :
			smallest = results[i][j]
			small_col = j
			small_row = i

print ("num of documents : %d"%numofdocuments)
print ("word length : %d" %len(key_list))
print ("biggest : %f" % biggest)
print ("big col : %d"% big_col)
print ("big row : %d"% big_row)
print ("The Biggest TF-IDF : %s"% key_list[big_col])
#print news.data[366
