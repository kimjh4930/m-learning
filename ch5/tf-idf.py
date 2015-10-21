from sklearn.feature_extraction.text import TfidfVectorizer 

corpus = ['The dog ate a sandwich', 'The wizard transfigured a sandwich', 'I ate a sandwich']
#corpus = ['The dog ate a sandwich']
vectorizer = TfidfVectorizer(stop_words='english', norm=None, smooth_idf=False)
#vectorizer = TfidfVectorizer(stop_words='english')
vectors = vectorizer.fit_transform(corpus)

key_list = vectorizer.get_feature_names()
result = vectors.todense()

print key_list
print result
