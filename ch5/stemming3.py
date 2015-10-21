from nltk import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk import pos_tag
from sklearn.feature_extraction.text import CountVectorizer

wordnet_tags = ['n', 'v']
corpus = ['He ate the sandwitches', 'Every sandwich was eaten by him']
stemmer = PorterStemmer()

vectorizer = CountVectorizer(binary=True, stop_words='english')

def lemmatize(token, tag):
	#print tag
	if tag[0].lower() in ['n', 'v']:
		print lemmatizer.lemmatize(token, tag[0].lower())
		return lemmatizer.lemmatize(token, tag[0].lower())
	return token

lemmatizer = WordNetLemmatizer()
tagged_corpus = [pos_tag(word_tokenize(document)) for document in corpus]

#for document in corpus :
#	print pos_tag(word_tokenize(document))
	
print('Lemmatized:', [[lemmatize(token, tag) for token, tag in document] for document in tagged_corpus])
