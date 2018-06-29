#!/usr/bin/python


def outlierCleaner(predictions, ages, net_worths):
    """
        Clean away the 10% of points that have the largest
        residual errors (difference between the prediction
        and the actual net worth).

        Return a list of tuples named cleaned_data where
        each tuple is of the form (age, net_worth, error).
    """
    cleaned_data = zip(ages, net_worths, [(float(pred) - actual)**2 for pred, actual in zip(predictions, net_worths)])
    cleaned_data.sort(key = lambda tup: tup[2])

    for i in range(0, int(len(cleaned_data) * 0.1)):
        cleaned_data.pop()

    return cleaned_data
    #
    # import numpy as np
    # cleaned_data = []
    # distance2 = predictions-net_worths
    # distance2 = distance2 * distance2
    #
    #
    # # Top 10 %
    # tmp = np.sort(distance2.flatten())
    # total = tmp.__len__()
    # ten_per = total / 10
    # top_10 = tmp[total-ten_per:total]
    # tmp = None
    #
    # import code; code.interact(local=dict(globals(), **locals()))
    # # Indexes
    # index_of_10percents = []
    # for idx, e in enumerate(distance2):
    #     for top in top_10:
    #         if e[0] == top:
    #             print("it s a match: idx: " + str(idx) + "value: " + str(e[0]))
    #             index_of_10percents.append(idx)
    #
    # # Delete them
    # np.delete(predictions, index_of_10percents)
    # np.delete(ages, index_of_10percents)
    # np.delete(net_worths, index_of_10percents)
    # print("-"*100)
    # for idx, e in enumerate(ages):
    #     element = [ages[idx], net_worths[idx], predictions[idx]]
    #     cleaned_data.append(element)
    #
    #
    # return cleaned_data
