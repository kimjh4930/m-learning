from sklearn.feature_extraction import DictVectorizer

onehot_encoder = DictVectorizer()

measurements = [{'city' : 'Dubai', 'temperature' : 33}, {'city' : 'London', 'temperature' : 12}, {'city' : 'San Fransisco', 'temperature' : 18}]

print onehot_encoder
print (onehot_encoder.fit_transform(measurements).toarray())
print onehot_encoder.get_feature_names()
