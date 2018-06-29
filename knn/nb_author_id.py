#!/usr/bin/python

"""
    This is the code to accompany the Lesson 1 (Naive Bayes) mini-project.

    Use a Naive Bayes Classifier to identify emails by their authors

    authors and labels:
    Sara has label 0
    Chris has label 1
"""

import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()


#########################################################
### your code goes here ###
features_train = features_train[:len(features_train)]
labels_train = labels_train[:len(labels_train)]

from sklearn.metrics import accuracy_score
import numpy as np
from sklearn import neighbors, datasets

# Train model
# import code; code.interact(local=dict(globals(), **locals()))
clf = neighbors.KNeighborsClassifier(100, weights='uniform')
t0 = time()
clf.fit(features_train, labels_train)

# Get prediction
labels_test_predicted = clf.predict(features_test)

# Get accuracy
acc = accuracy_score(labels_test, labels_test_predicted)
print("Finished. Accuracy: " + str(acc))
print "training time:", round(time()-t0, 3), "s"

#########################################################
