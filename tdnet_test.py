from bit_to_bit_gridworld_env import *
from utils import *
import collections
from td_net_with_history_utils import *

# create the environment
environment = BitToBitGridWorld(6, 6, [[3, 1], [3, 2], [4, 2], [3, 3], [2, 3],
                                       [4, 5], [3, 5], [2, 5], [1, 5], [0, 5]], [0, 4], Direction.West, True)
num_actions = 2
num_observations = 2

# the td net with history attributes
history_length = 6
td_net_depth = 5
time_step = 0

# setting up the history
observation_history = collections.deque(history_length*[None], history_length)
action_history = collections.deque(history_length*[None], history_length)

last_observation = None
last_action = None

for i in range(10):
    action = np.random.choice(num_actions, 1)[0]
    if action == 0:
        environment.move_forward()
        last_observation = environment.get_observation()
        last_action = action
        observation_history.appendleft(last_observation)
        action_history.appendleft(last_action)
    elif action == 1:
        environment.turn_clockwise()
        last_observation = environment.get_observation()
        last_action = action
        observation_history.appendleft(last_observation)
        action_history.appendleft(last_action)

    print(action)
    print(action_history)
