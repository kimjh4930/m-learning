import nltk
from nltk.stem.wordnet import WordNetLemmatizer

nltk.download
lemmatizer = WordNetLemmatizer()

print lemmatizer.lemmatize('gathering','v')
print lemmatizer.lemmatize('gathering','n')

