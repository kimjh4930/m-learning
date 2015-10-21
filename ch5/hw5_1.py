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
#news = fetch_20newsgroups(categories = categories)

for i in range(0, len(news.data)) :
	news.data[i] = re.sub(r'[^\w\s]', '', news.data[i])

class LemmaTokenizer(object):
	def __init__(self):
		self.wnl = WordNetLemmatizer()
	def __call__(self, doc):
		return [self.wnl.lemmatize(t) for t in word_tokenize(doc)]

#token_dict = {}
stemmer = PorterStemmer()

def stem_tokens(tokens, stemmer):
	stemmed = []
	for item in tokens:
		stemmed.append(stemmer.stem(item))
	print stemmed
	return stemmed

def StemTokenizer(text):
	tokens = nltk.word_tokenize(text)
	stems = stem_tokens(tokens, stemmer)
	return stems

vectorizer = TfidfVectorizer(tokenizer = LemmaTokenizer(), stop_words='english', norm=None, smooth_idf=False)
#vectorizer = TfidfVectorizer(stop_words='english',tokenizer=StemTokenizer , norm=None, smooth_idf=False)

vectors = vectorizer.fit_transform(news.data)

wordlength = len(vectorizer.vocabulary_)
numofdocuments = len(news.data)

key_list = vectorizer.get_feature_names()
results = vectors.toarray()

print key_list

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

print ("biggest : %f" % biggest)
print ("smallest : %f" % smallest)
print ("big col : %d"% big_col)
print ("big row : %d"% big_row)
print ("bigcol key : %s"% key_list[big_col])

#add dictionary
#result_pair = dict()
#for i in range(0, wordlength):
#	result_pair[key_list[i]] = summary[i]

#print result_pair

#sorted_result = sorted(result_pair.iteritems(), key=itemgetter(1), reverse=True)

#for i in range(0,15) :
#	print sorted_result[i]

#print " "
#for i in range(wordlength-15, wordlength) :
#	print sorted_result[i]
