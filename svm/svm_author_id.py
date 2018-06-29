#!/usr/bin/python

"""
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:
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
# features_train = features_train[:len(features_train)/100]
# labels_train = labels_train[:len(labels_train)/100]

from sklearn import svm
clf = svm.SVC(C=10000.0, kernel="rbf")
clf.fit(features_train, labels_train)
print("fitted")

predictions = clf.predict(features_test)
import code; code.interact(local=dict(globals(), **locals()))
print("predicted")
# from sklearn.metrics import accuracy_score
# acc = accuracy_score(labels_test, predictions)
# print("Finished. Accuracy: " + str(acc))

# # get the hyperplane# get the separating hyperplane
# import numpy as np
# w = clf.coef_[0]
# a = -w[0] / w[1]
# xx = np.linspace(0, 0.5)
# yy = a * xx - (clf.intercept_[0]) / w[1]
#
# # draw it
# import matplotlib.pyplot as plt
# plt.plot(xx, yy, 'k-')
# plt.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1],
#             s=80, facecolors='none')
# plt.scatter(features_train[:, 0], features_train[:, 1], c=labels_train, cmap=plt.cm.Paired) # RdBu_r
# plt.show()
# #########################################################
