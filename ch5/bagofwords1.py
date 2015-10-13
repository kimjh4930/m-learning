from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import euclidean_distances

corpus = ['UNC played Duke in basketball',
				  'Duke lost the basketball game',
					'I ate a sandwich']
vectorizer = CountVectorizer()

counts = vectorizer.fit_transform(corpus).todense()

print counts
print vectorizer.vocabulary_

print euclidean_distances(counts[0], counts[1])
print euclidean_distances(counts[0], counts[2])
