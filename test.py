import numpy as np
from utils import *
import collections
from bit_to_bit_gridworld_GUI_exp import *
from bit_to_bit_gridworld_env import *



b = np.asarray(np.loadtxt('predictions_y.txt', dtype=float))
c = np.asarray(np.loadtxt('extra_state_setting.txt', dtype=int))
# gui creation
m = 6
n = 6
history_length = 6
obstacles = [[3, 1], [3, 2], [4, 2], [3, 3], [2, 3], [4, 5], [3, 5], [2, 5], [1, 5], [0, 5]]
history_observation = []
history_action = []
for i in range(history_length):
    history_observation.append(c[i])
    history_action.append(c[i+history_length])

initial_position = [c[12], c[13]]
initial_direction = Direction(c[14])

gui = BitToBitGridWorldGUI(m,n,obstacles,initial_position,initial_direction)

