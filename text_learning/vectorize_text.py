#!/usr/bin/python

import os
import pickle
import re
import sys

sys.path.append( "../tools/" )
from parse_out_email_text import parseOutText

"""
    Starter code to process the emails from Sara and Chris to extract
    the features and get the documents ready for classification.

    The list of all the emails from Sara are in the from_sara list
    likewise for emails from Chris (from_chris)

    The actual documents are in the Enron email dataset, which
    you downloaded/unpacked in Part 0 of the first mini-project. If you have
    not obtained the Enron email corpus, run startup.py in the tools folder.

    The data is stored in lists and packed away in pickle files at the end.
"""

## TEST
# from sklearn.feature_extraction.text import CountVectorizer
# list_of_new_documents = [
#   "hi Katie the self driving car will be late Best Sebastian",
#   "hi Sebastian the machine learning class will be great great great Best Katie",
#   "Hi Katie the machine learning class will be most excellent"
# ]
# vectorizer = CountVectorizer() # (tokenizer=tokenize)
# # fit figure out what are the words in the corpus
# # transform count them
# bag_of_words = vectorizer.fit_transform(list_of_new_documents)
# # bag_of_words = vectorizer.transform(list_of_new_documents)
# # sprint(bag_of_words)
# print(vectorizer.vocabulary_.get("great"))



# from nltk.corpus import stopwords
# import nltk
# # nltk.download()
# sw = stopwords.words("german")
# print(str(sw[0:10]))
# print(str(sw.__len__()))



# from nltk.stem.snowball import SnowballStemmer
# stemmer = SnowballStemmer("english")
# stemmer.stem("responsiveness")

# EN TEST

from_sara  = open("from_sara.txt", "r")
from_chris = open("from_chris.txt", "r")

from_data = []
word_data = []

### temp_counter is a way to speed up the development--there are
### thousands of emails from Sara and Chris, so running over all of them
### can take a long time
### temp_counter helps you only look at the first 200 emails in the list so you
### can iterate your modifications quicker
temp_counter = 0


for name, from_person in [("sara", from_sara), ("chris", from_chris)]:
    for path in from_person:
        ### only look at first 200 emails when developing
        ### once everything is working, remove this line to run over full dataset
        temp_counter += 1
        if True: # temp_counter < 200:
            path = os.path.join('..', path[:-1])
            print path
            try:
                email = open(path, "r")
            except:
                import code; code.interact(local=dict(globals(), **locals()))

            ### use parseOutText to extract the text from the opened email
            foo = parseOutText(email)

            ### use str.replace() to remove any instances of the words
            for stop_word in ["sara", "shackleton", "chris", "germani", "sshacklensf", "cgermannsf"]: # "houectect", "houect", "houston"]:
                foo = foo.replace(stop_word, "")
            foo = foo.replace("  ", " ")

            ### append the text to word_data
            word_data.append(foo)

            ### append a 0 to from_data if email is from Sara, and 1 if email is from Chris
            if name == "sara":
                from_data.append(0)
            elif name == "chris":
                from_data.append(1)
            else:
                raise "OMFG"

            email.close()

print "emails processed"
from_sara.close()
from_chris.close()

pickle.dump( word_data, open("your_word_data.pkl", "w") )
pickle.dump( from_data, open("your_email_authors.pkl", "w") )
print("-"*100)
print(word_data[152])
print("-"*100)



### in Part 4, do TfIdf vectorization here

# from nltk.corpus import stopwords
# sw = stopwords.words("english")
from sklearn.feature_extraction.text import TfidfVectorizer
tdidf_vectorizer = TfidfVectorizer(stop_words='english') # , lowercase=True) #
res = tdidf_vectorizer.fit_transform(word_data)
print(len(res.toarray())) # nb de documents
print(len(res.toarray()[0])) # nb de mots
tdidf_vectorizer.get_feature_names()[34597]

# import code; code.interact(local=dict(globals(), **locals()))
