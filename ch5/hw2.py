import numpy as np

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

def show_top10(classifier, vectorizer, categories):
	feature_name = np.asarray(vectorizer.get_feature_name())
	for i, category in enumerate(categories):
		top10 = np.argsort(classifier.coef_[i])[-10:]
		print ("%s : %s", %(category, " ".join(feature_names[top10])))

categories = ['comp.graphics']
news = fetch_20newsgroups(categories = categories)

corpus = news.data[:2]

vectorizer = CountVectorizer(stop_words='english')
print vectorizer.fit_transform(corpus).todense()
print vectorizer.vocabulary_
