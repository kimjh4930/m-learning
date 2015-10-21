from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

corpus = ['The dog ate a sandwich', 'The wizard transfigured a sandwich', 'I ate a sandwich']

vectorizer = CountVectorizer(stop_words='english')
#transformer = TfidfTransformer(norm=None, smooth_idf=False)
transformer = TfidfTransformer()

print transformer

x = vectorizer.fit_transform(corpus)
print vectorizer.vocabulary_
print transformer.fit_transform(x).todense()
