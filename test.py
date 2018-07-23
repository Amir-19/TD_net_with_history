import numpy as np
from utils import *
import collections
from bit_to_bit_gridworld_env import *

sum = np.zeros((4,1))
a=np.arange(4).reshape(4,1)
sum+=np.multiply(a,a)
sum+=np.multiply(a,a)
print(np.sqrt(sum))