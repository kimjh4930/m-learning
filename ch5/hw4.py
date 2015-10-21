import numpy as np

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

categories = ['comp.graphics']
news = fetch_20newsgroups(categories = categories)

index = 3

vectorizer = TfidfVectorizer(stop_words='english')
#count_vector = CountVectorizer(stop_words='english')
vectors = vectorizer.fit_transform(news.data[:index])
#print count_vector.fit_transform(news.data[:index]).todense()

#return TF-IDF weighted document-term matrix
#print vectors
print vectorizer.vocabulary_

#print(vectors[2][1])
#print(vectors[1])

results = vectors.todense()[0][0]

print results[0][0]
#print(vectors.todense()[0])
#print(vectors.todense()[1])
