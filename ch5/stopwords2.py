from sklearn.feature_extraction.text import CountVectorizer

corpus=['He ate the sandwitches',
			  'Every sandwitch was eaten by him']

vectorizer = CountVectorizer(binary=True, stop_words='english')

print vectorizer.fit_transform(corpus).todense()
print vectorizer.vocabulary_
