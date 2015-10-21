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
		print lemmatizer.lemmatize(token, tag[0].lower())
		return lemmatizer.lemmatize(token, tag[0].lower())
	print token
	return token

def LemmaTokenizer(text):
	#print text
	#print "lemmatizerinit"
	words = nltk.word_tokenize(text)
	print ("words : ",words)
	lem=[]
	token1=[]
	tokens=[]
	tagged_corpus = [pos_tag(word_tokenize(document)) for document in corpus]
	#print tagged_corpus
	for document in tagged_corpus:
		for token, tag in document:
			token1.append(lemmatize(token, tag))
	tokens.append(token1)
	print ("tokens : ",tokens)
	return tokens



vectorizer = TfidfVectorizer(stop_words='english', tokenizer=LemmaTokenizer, norm=None, smooth_idf=False)
vectors = vectorizer.fit_transform(corpus)


key_list = vectorizer.get_feature_names()
result = vectors.toarray()


print key_list
print result
print vectorizer.vocabulary_


#print('Lemmatized:', [[lemmatize(token, tag) for token, tag in document] for document in tagged_corpus])
#print('Lemmatized:', [[lemmatize(token, tag) for token, tag in document] for document in tagged_corpus])
