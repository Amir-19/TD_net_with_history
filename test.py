import numpy as np
from utils import *
import collections
from bit_to_bit_gridworld_env import *

environment = BitToBitGridWorld(6, 6, [[3, 1], [3, 2], [4, 2], [3, 3], [2, 3],
                                           [4, 5], [3, 5], [2, 5], [1, 5], [0, 5]], [0, 4], Direction.West)

sequence = "FFRFFRRRFRRRRFR"
print(environment.get_n_step_observation(sequence))