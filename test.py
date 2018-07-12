import numpy as np
from utils import *
import collections
from bit_to_bit_gridworld_GUI_exp import *
from bit_to_bit_gridworld_env import *
from td_net_with_history_utils import *


# n = 3
# m = 2
# zy = [[1, 2, 3]]
# zy = np.array(zy)
# c = np.asarray([[1, 1, 1]])
# x = np.asarray([[6,7]])
# W = np.full((n, m),0)
#
# yyy = np.multiply(zy,c).T
#
# print(np.dot(yyy,x))
# print(np.outer(yyy,x))
# print(zy[0])

a = [1,2]
observation_history = collections.deque(a, 2)
print(observation_history)