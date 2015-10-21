import nltk
import string

from nltk import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk import pos_tag
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

corpus = ['He ate the sandwiches', 'Every sandwich was eaten by him']

vectorizer = CountVectorizer(binary=True, stop_words='english')

stemmer = PorterStemmer()

def stem_tokens(tokens, stemmer):
	stemmed = []
	for item in tokens:
		stemmed.append(stemmer.stem(item))
	return stemmed

def StemTokenizer(text):
	tokens = nltk.word_tokenize(text)
	stems = stem_tokens(tokens, stemmer)
	print stems
	return stems

class LemmaTokenize(object):
	def __init__(self):
		self.wnl = WordNetLemmatizer()
	def __call__(self, doc):
		return [self.wnl.lemmatize(t) for t in word_tokenize(doc)]

#vectorizer = TfidfVectorizer(stop_words='english', tokenizer=StemTokenizer, norm=None, smooth_idf=False)
vectorizer = TfidfVectorizer(stop_words='english', tokenizer=LemmaTokenize(), norm=None, smooth_idf=False)
vectors = vectorizer.fit_transform(corpus)


key_list = vectorizer.get_feature_names()
result = vectors.toarray()


print key_list
print result


#print('Lemmatized:', [[lemmatize(token, tag) for token, tag in document] for document in tagged_corpus])
#print('Lemmatized:', [[lemmatize(token, tag) for token, tag in document] for document in tagged_corpus])
