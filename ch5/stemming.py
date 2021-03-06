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

lemmatizer = WordNetLemmatizer()
tagged_corpus = [pos_tag(word_tokenize(document)) for document in corpus]

def lemmatize(token, tag):
	if tag[0].lower() in ['n','v']:
		return lemmatizer.lemmatize(token, tag[0].lower())
	return token

token=[]
token1=[]
def LemmaTokenizer(text):
	tokens = nltk.word_tokenize(text)
	i=0

	for document in tagged_corpus:
		for token, tag in document:
			token[i] = str(lemmatize(token, tag))
			#token.append(lemmatize(token, tag))
			print token
#	print token1.append(token)
	return tokens



#vectorizer = TfidfVectorizer(stop_words='english', tokenizer=StemTokenizer, norm=None, smooth_idf=False)
vectorizer = TfidfVectorizer(stop_words='english', tokenizer=LemmaTokenizer, norm=None, smooth_idf=False)
vectors = vectorizer.fit_transform(corpus)


key_list = vectorizer.get_feature_names()
result = vectors.toarray()


print key_list
print result


#print('Lemmatized:', [[lemmatize(token, tag) for token, tag in document] for document in tagged_corpus])
#print('Lemmatized:', [[lemmatize(token, tag) for token, tag in document] for document in tagged_corpus])
