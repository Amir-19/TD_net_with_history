import numpy as np
from utils import *
import collections
from bit_to_bit_gridworld_GUI_exp import *
from bit_to_bit_gridworld_env import *
from td_net_with_history_utils import *


y, history_observation, history_action, initial_position, initial_direction = experiment_file_reader(history_length=6)
m = 6
n = 6
obstacles = [[3, 1], [3, 2], [4, 2], [3, 3], [2, 3], [4, 5], [3, 5], [2, 5], [1, 5], [0, 5]]
print(y)
gui = BitToBitGridWorldGUI(m,n,obstacles,initial_position,initial_direction)

