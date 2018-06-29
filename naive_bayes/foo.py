import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
X = cv.fit_transform(['porridge', 'porridge', 'porridge', 'porridge pot', 'hot days']).toarray()
Y = np.array([1, 1, 2, 2, 4])

print(X)
print(cv.get_feature_names())
print(Y)
print("-"*100)

from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB()
clf.fit(X, Y)
# import code; code.interact(local=dict(globals(), **locals()))
print(clf.predict([[0, 0, 1, 1]]))
