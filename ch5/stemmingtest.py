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
from nltk import pos_tag

corpus = ['The dog ate a sandwich', 'The wizard transfigured a sandwich', 'I ate a sandwich']

for i in range(0, len(corpus)) :
	corpus[i] = re.sub(r'[^\w\s]', '', corpus[i])

class LemmaTokenizer(object):
	def __init__(self):
		self.wnl = WordNetLemmatizer()
	def __call__(self, doc):
		return [self.wnl.lemmatize(t) for t in word_tokenize(doc)]

'''
toekn_dict = {}
stemmer = PorterStemmer()

def stem_tokens(tokens, stemmer) : 
	stemmed = []
	for item in tokens:
		stemmed.append(stemmer.stem(item))
	return stemmed

def StemTokenizer(text):
	tokens = nltk.word_tokenize(text)
	stems = stem_tokens(tokens, stemmer)
	return stems
'''
def lemmatize(token, tag):
	if tag[0].lower() in ['n', 'v']:
		return lemmatizer.lemmatize(token, tag[0].lower())
	return token

lemmatizer = WordNetLemmatizer()
tagged_corpus = [pos_tag(word_tokenize(document)) for document in corpus]

print tagged_corpus

def lemma(token, tag):
	[[lemmatize(token, tag) for token, tag in document] for document in tagged_corpus]

#vectorizer = TfidfVectorizer(stop_words='english', tokenizer=StemTokenizer, norm=None, smooth_idf=False)
vectorizer = TfidfVectorizer(stop_words='english', tokenizer=lemma, norm=None, smooth_idf=False)

vectors = vectorizer.fit_transform(corpus)

key_list = vectorizer.get_feature_names()
result = vectors.toarray()

print key_list
print result
