import nltk

from nltk import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk import pos_tag
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

wordnet_tags = ['n', 'v']
corpus = ['He ate the sandwiches', 'Every sandwich was eaten by him']

vectorizer = CountVectorizer(binary=True, stop_words='english')

def lemmatize(token, tag):
	if tag[0].lower() in ['n', 'v']:
		return lemmatizer.lemmatize(token, tag[0].lower())
	return token

#lemma = WordNetLemmatizer()
lemmatizer = WordNetLemmatizer()
tagged_corpus = [pos_tag(word_tokenize(document)) for document in corpus]

#print tagged_corpus

def lemmatize(token, tag):
	#print "lemmatizeinit"
	if tag[0].lower() in ['n', 'v']:
		#print lemmatizer.lemmatize(token, tag[0].lower())
		return lemmatizer.lemmatize(token, tag[0].lower())
	return token

def LemmaTokenizer(text):
	#print text
	#print "lemmatizerinit"
	lem=[]
	tagged_corpus = [pos_tag(word_tokenize(document)) for document in corpus]
	print tagged_corpus
	for document in tagged_corpus:
		for token, tag in document:
			lem.append(lemmatize(token, tag))
	#print lem
	return lem
	#print(lem)


'''
def lemma_tokens(tokens, lemma):
	lem = []
	#print tokens
	#print lemma
	for item in tokens:
		#if tag[0].lower() in ['n', 'v']:
			#print tag[0].lower()
		#lem.append(lemma.lemmatize(item,tag[0].lower()))
		lem.append(lemma.lemmatize(item,'v'))

	#print 'lem:',[lem]
	return lem

def LemmaTokenizer(text):
	tokens = nltk.word_tokenize(text)
	print lemma
	lems = lemma_tokens(tokens, lemma)
	return lems

def LemmaTokenizer(text):
	tokens = nltk.word_tokenize(text)
	print tokens
	
	for document in tagged_corpus:
		for token, tag in document:
			lemmatize(token,tag)
'''
vectorizer = TfidfVectorizer(stop_words='english', tokenizer=LemmaTokenizer, norm=None, smooth_idf=False)
vectors = vectorizer.fit_transform(corpus)


key_list = vectorizer.get_feature_names()
result = vectors.toarray()


print key_list
print result


#print('Lemmatized:', [[lemmatize(token, tag) for token, tag in document] for document in tagged_corpus])
#print('Lemmatized:', [[lemmatize(token, tag) for token, tag in document] for document in tagged_corpus])
