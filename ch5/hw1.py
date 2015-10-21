from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

#from pprint import pprint

categories = ['comp.graphics']

newsgroups_train = fetch_20newsgroups(categories = categories)

vectorizer = CountVectorizer(stop_words='english')
swdense = vectorizer.fit_transform(newsgroups_train.data).todense()
vocabulary = vectorizer.vocabulary_

print(newsgroups_train.target_names)

#TF-IDF

tfidf_vec = TfidfVectorizer()
vectors = vectorizer.fit_transform(vocabulary)

print tfidf_vec

#access data
print newsgroups_train.data[0]


#vectorizer = TfidfVectorizer()
#vectors = vectorizer.fit_transform(newsgroups_train.data)

#print(vectors.shape)
#print(newsgroups_train.filenames.shape)
#print(newsgroups_train.target_names)
#print(newsgroups_train.target[:10])
#pprint(list(newsgroups_train.target_names))
