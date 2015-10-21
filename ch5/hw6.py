import numpy as np

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from operator import itemgetter

categories = ['comp.graphics']
news = fetch_20newsgroups(categories = categories)

#vectorizer = TfidfVectorizer(stop_words='english')
vectorizer = TfidfVectorizer()
vectors = vectorizer.fit_transform(news.data)

#print vectorizer.vocabulary_

wordlength = len(vectorizer.vocabulary_)
numofdocuments = len(news.data)

results = vectors.toarray()

name_list = vectorizer.get_feature_names()
#print name_list
#print vectorizer.vocabulary_.items()

#print sorted_voca

biggest = 0.0
smallest = 0.0
big_col = 0
small_col = 0.0


store_index = []
for i in range(0, 2):
	store_index.append(0)
for k in range(0,2) :
	for i in range(0,numofdocuments):
		for j in range(0, wordlength):
			if results[i][j] > biggest :
				biggest = results[i][j]
				if big_col > 0 and big_col == j :
					big_col = 0
					continue
				else :
					big_col = j
			elif results[i][j] < smallest :
				smallest = results[i][j]
				small_col = j
	store_index[k] = big_col

print(store_index)
#print("biggest : %f"% biggest)
#print("smallest : %f"% smallest)
print("big_col : %d"% big_col)
print("big_col key : %s"% name_list[big_col] )
#print("small_col : %d"% small_col)


