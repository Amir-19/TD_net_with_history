import numpy as np
from utils import *
import collections
from bit_to_bit_gridworld_env import *


d = dict()

key = ((1,2),"haha")
key2 = ((1,3),"haha")
d[key] = (9,1)
d[key2] = (4,2)
if key in d:
    print(d[key])