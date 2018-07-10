from bit_to_bit_gridworld_env import *
from utils import *
import collections
from td_net_with_history_utils import *
import numpy as np


def create_feature_vector(obs_history,action_history,predictions):
    x = []

    # adding the bias unit
    x.extend([1])

    # adding history of observations


    # adding history of actions


    # adding the predictions from last time


    return x


def condition(action, n):
    if action == 1:
        base_condition = [0, 1]
    elif action == 0:
        base_condition = [1, 0]

    condition_vec = []
    condition_vec.extend(base_condition for i in range(int(int(n)/2)))
    condition_vec = np.asarray(condition_vec).flatten()
    return condition_vec


def calculate_targets(observation, prev_predictions):

    targets = np.zeros(len(prev_predictions))

    # the first two node are connected to the observation, so the target for those two is the observation
    targets[0] = observation
    targets[1] = observation

    # each node i is connected to nodes 2i and 2i + 1
    for i in range(int((len(prev_predictions)-2)/2)):
        targets[2*(i+1)] = prev_predictions[i]
        targets[2*(i+1)+1] = prev_predictions[i]

    return targets


def main():
    # create the environment
    environment = BitToBitGridWorld(6, 6, [[3, 1], [3, 2], [4, 2], [3, 3], [2, 3],
                                           [4, 5], [3, 5], [2, 5], [1, 5], [0, 5]], [0, 4], Direction.West)
    num_actions = 2
    num_observations = 2

    # the td net with history attributes
    history_length = 6
    td_net_depth = 5
    time_step = 0
    n = 62 # since we have td net with 5 layers and 2 actions 2^6  -  2  =  62
    y = np.ones(n)*0.5

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

    print(observation_history,action_history,environment.agent_direction,environment.agent_position)
    save_to_file(y,action_history,observation_history,environment.agent_direction,environment.agent_position)


if __name__ == "__main__":
    main()

