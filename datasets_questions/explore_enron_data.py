#!/usr/bin/python

"""
    Starter code for exploring the Enron dataset (emails + finances);
    loads up the dataset (pickled dict of dicts).

    The dataset has the form:
    enron_data["LASTNAME FIRSTNAME MIDDLEINITIAL"] = { features_dict }

    {features_dict} is a dictionary of features associated with that person.
    You should explore features_dict as part of the mini-project,
    but here's an example to get you started:

    enron_data["SKILLING JEFFREY K"]["bonus"] = 5600000

"""

import pickle

enron_data = pickle.load(open("../final_project/final_project_dataset.pkl", "r"))

# 1. Center and norm shit
#########################
import numpy as np

# import code; code.interact(local=dict(globals(), **locals()))
def correct_NaN(key):
    for name in enron_data:
        if enron_data[name][key] == 'NaN':
            enron_data[name][key] = 0


correct_NaN('from_poi_to_this_person')
correct_NaN('from_this_person_to_poi')
correct_NaN('total_payments')

# def center_and_norm(key):
#     results = []
#     for name in enron_data:
#         if enron_data[name][key] == 'NaN':
#             enron_data[name][key] = 0
#         results.append(enron_data[name][key])
#
#     results = np.array(results)
#     mean    = results.mean()
#     std     = results.std()
#
#     for name in enron_data:
#         enron_data[name][key] = enron_data[name][key] - mean
#         enron_data[name][key] = enron_data[name][key] / std


# center_and_norm('from_poi_to_this_person')
# center_and_norm('from_this_person_to_poi')
# center_and_norm('total_payments')

# 2. Build train & test dataset
################################
training_dataset = {}
test_dataset     = {}
try:
    training_dataset = pickle.load(open("../final_project/training_dataset.pkl", "r"))
    test_dataset = pickle.load(open("../final_project/test_dataset.pkl", "r"))
except:
    from random import randint

    for person_k in enron_data:
        if randint(0,100) >= 80:
            training_dataset[person_k] = enron_data[person_k]
        else:
            test_dataset[person_k] = enron_data[person_k]

    # Ensure 80/20 between test- and training- dataset accross the entire people
    # Ensure 80/20 between test- and training- dataset accross the POI
    # import code; code.interact(local=dict(globals(), **locals()))
    # i = 0
    # for person_k in test_dataset:
    #     if test_dataset[person_k]['poi'] == True:
    #         i = i+1
    #
    # print("POI count: "+ str(i))

    pickle.dump(training_dataset, open("../final_project/training_dataset.pkl", "wb"))
    pickle.dump(test_dataset, open("../final_project/test_dataset.pkl", "wb"))
    print("re-created")

# 3. Model: detect PIO
# def dataset_to_value(dict, key, feature):
#     result = []
#     for people_k in dict:
#         value = dict[people_k][key]
#         if value == 'NaN':
#             value = 0
#         if feature:
#             value = [value]
#         result.append(value)
#     return result


def dataset_to_value(dict, feature):
    result = []
    for people_k in dict:
        if feature:
            val1 = dict[people_k]['from_poi_to_this_person']
            val2 = dict[people_k]['from_this_person_to_poi']
            val3 = dict[people_k]['total_payments']
            if val1 == 'NaN':
                val1 = 0
            if val2 == 'NaN':
                val2 = 0
            if val3 == 'NaN':
                val3 = 0
            value = [val1, val2, val3]
            # value = [val3]
        else:
            value = dict[people_k]['poi']
        result.append(value)
    return result

features_train = dataset_to_value(training_dataset, True)
features_test = dataset_to_value(test_dataset, True)




labels_train = dataset_to_value(training_dataset, False)
labels_test = dataset_to_value(test_dataset, False)


from sklearn.metrics import accuracy_score
import numpy as np
from sklearn import neighbors, datasets

clf = neighbors.KNeighborsClassifier(5, weights='distance')
clf.fit(features_train, labels_train)

labels_test_predicted = clf.predict(features_test)

acc = accuracy_score(labels_test, labels_test_predicted)
print("Accuracy: " + str(acc * 100))

import code; code.interact(local=dict(globals(), **locals()))




# people_random_order = enron_data.keys()
# people_random_order = shuffle()


#
# features = []
# flatten = lambda l: [item for sublist in l for item in sublist]
# # results = [t.age for t in mylist if t.person_id == 10]
#
# # for k in enron_data:
# #     features.append(enron_data[k].keys())
# # res = flatten(features)
# # res2 = reduce(lambda l, x: l if x in l else l+[x], res, [])
#
#
# # i = 0
# # for person_k in enron_data:
# #     if enron_data[person_k]['poi'] == True:
# #         i = i+1
# #     else:
# #         print(enron_data[person_k]['poi'])
#
# res = flatten(features)
# res2 = reduce(lambda l, x: l if x in l else l+[x], res, [])
#
#
#
#
#
# import re
# regex = re.compile('PRENTICE')
# results = [name for name in enron_data.keys() if regex.match(name)]
#
#
# count_email  = 0
# count_salary = 0
# for person_k in enron_data:
#     if enron_data[person_k]['email_address'] and enron_data[person_k]['email_address'] != 'NaN':
#         count_email += 1
#         print(enron_data[person_k]['email_address'])
#     if enron_data[person_k]['salary'] and enron_data[person_k]['salary'] != 'NaN':
#         count_salary += 1
#         print(enron_data[person_k]['salary'])
#     print("")
#
# print("email:" + str(count_email))
# print("salary: " + str(count_salary))
#
#
#
#
#
# count_salary = 0
# total_poi = 0
# for person_k in enron_data:
#     if True: # enron_data[person_k]['poi'] == 1:
#         total_poi = total_poi + 1
#         if enron_data[person_k]['total_payments'] == 'NaN':
#             count_salary = count_salary + 1
#
# print("")
# print("count NaN: " + str(count_salary))
# print("count POI: " + str(total_poi))
# print("percentage: " + str(float(count_salary) / total_poi * 100))
