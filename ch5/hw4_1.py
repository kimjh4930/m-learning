from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

corpus = [
			'The dog ate a sandwich',
			'The wizard transfigured a sandwich',
			'I ate a sandwich jang'
			]

print corpus
vectorizer = CountVectorizer(stop_words='english')
transformer = TfidfTransformer()
X = vectorizer.fit_transform(corpus)
print vectorizer.vocabulary_
print transformer.fit_transform(X).todense()
