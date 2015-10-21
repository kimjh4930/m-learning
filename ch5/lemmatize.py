import nltk

from nltk import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk import pos_tag
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from nltk.corpus import stopwords

wordnet_tags = ['n', 'v']
#corpus = ['He ate the sandwiches', 'Every sandwich was eaten by him']
#corpus = ['Every sandwich was eaten by him']

corpus = ['The dog ate a sandwich', 'The wizard transfigured a sandwich', 'I ate a sandwich']

vectorizer = CountVectorizer(binary=True, stop_words='english')

lemmatizer = WordNetLemmatizer()

def lemmatize(token, tag):
	if tag[0].lower() in ['n', 'v']:
		return lemmatizer.lemmatize(token, tag[0].lower())
	return token

def LemmaTokenizer(text):
	words = nltk.word_tokenize(text)
	print words
	lem=[]
	token=[]
	tagged_corpus = [pos_tag(word_tokenize(document)) for document in words]
#	print tagged_corpus
	#tokens = [[lemmatize(token, tag) for token, tag in document] for document in tagged_corpus][0]
	
	for ducument in tagged_corpus:
		for token, tag in document:
			token = lemmatize(token, tag)
		tokens = tokens.append(token)
		print tokens
	#print lem
	
#	lem.append(tokens)
	#lem.append([[lemmatize(token, tag) for token, tag in document] for document in tagged_corpus])
	
	#print lem
	return words

	#for document in tagged_corpus:
	#	for token, tag in document:
	#		lem.append(lemmatize(token, tag))	
	#return lem

vectorizer = TfidfVectorizer(stop_words='english', tokenizer=LemmaTokenizer, norm=None, smooth_idf=False)
vectors = vectorizer.fit_transform(corpus)

key_list = vectorizer.get_feature_names()
result = vectors.toarray()


print key_list
print result


#print('Lemmatized:', [[lemmatize(token, tag) for token, tag in document] for document in tagged_corpus])
#print('Lemmatized:', [[lemmatize(token, tag) for token, tag in document] for document in tagged_corpus])
