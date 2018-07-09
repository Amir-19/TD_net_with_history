import collections
import numpy as np

def create_feature_vector_of_history(a):
    a_as_one_digit = 0
    a_as_feature_vector = np.zeros(2**len(a))
    for i in range(len(a)):
       a_as_one_digit += a[len(a)-1-i]* (2 ** i)
    a_as_feature_vector[a_as_one_digit] = 1
    return a_as_feature_vector