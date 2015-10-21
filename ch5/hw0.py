from sklearn.datasets import fetch_20newsgroups
from pprint import pprint

newsgroups_train = fetch_20newsgroups(subset='train')
pprint(list(newsgroups_train.target_names))


