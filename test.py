import numpy as np
from utils import *
import collections

np.zeros((2,1))
observation_history = collections.deque([1,0], 3)
print(create_feature_vector(observation_history, observation_history,np.ones((2,1))))