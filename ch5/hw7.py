import numpy as np
import re

from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from operator import itemgetter

categories = ['comp.graphics']
news = fetch_20newsgroups(categories = categories, remove=('headers', 'footers', 'quotes'))

#news.data[0].translate(None, string.punctuation)
for i in range(0, len(news.data)) :
	news.data[i] = re.sub(r'[^\w\s]', '', news.data[i])
#news = fetch_20newsgroups(categories = categories)
 
class LemmaTokenizer(object):
	def __init__(self):
		self.wnl = WordNetLemmatizer()
	def __call__(self, doc):
		return [self.wnl.lemmatize(t) for t in word_tokenize(doc)]

vectorizer = TfidfVectorizer(stop_words='english', tokenizer = LemmaTokenizer())
#vectorizer = TfidfVectorizer()
vectors = vectorizer.fit_transform(news.data)

#print vectorizer.vocabulary_

wordlength = len(vectorizer.vocabulary_)
numofdocuments = len(news.data)

key_list = vectorizer.get_feature_names()
results = vectors.toarray()

print key_list
#print len(key_list)

big_value = 0
big_col = 0

matrix = [[0 for col in range(wordlength)] for row in range(numofdocuments)]

for i in range(0, numofdocuments):
	for j in range(0, wordlength):
		if results[i][j] > big_value :
			big_value = results[i][j]
			big_col = j
			#print ("big_value : %d " % results[i][j])
	big_value = 0
	matrix[i][big_col] = 1
#	print big_col
	print results[i][big_col]

summary = []
for i in range(0,wordlength):
	summary.append(0)

for i in range(0, wordlength):
	for j in range(0, numofdocuments):
		summary[i] += matrix[j][i]

#print len(summary)

#add dictionary
result_pair = dict()
for i in range(0, wordlength):
	result_pair[key_list[i]] = summary[i]

#print result_pair

sorted_result = sorted(result_pair.iteritems(), key=itemgetter(1), reverse=True)

for i in range(0,50) :
	print sorted_result[i]

print " "
for i in range(wordlength-50, wordlength) :
	print sorted_result[i]

