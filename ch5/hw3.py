import numpy as np

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics

def show_top10(classifier, vectorizer, categories):
	feature_name = np.asarray(vectorizer.get_feature_name())
	for i, category in enumerate(categories):
		top10 = np.argsort(classifier.coef_[i])[-10:]
		print ("%s : %s" % (category, " ".join(feature_names[top10])))

categories = ['comp.graphics']
news = fetch_20newsgroups(categories = categories)

#Converting text to vectors
vectorizer = TfidfVectorizer()
vectors = vectorizer.fit_transform(news.data[:2])

vectors_test = vectorizer.transform(news.data[:2])

