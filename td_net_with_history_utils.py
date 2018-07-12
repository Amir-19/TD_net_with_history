import numpy as np
from bit_to_bit_gridworld_env import *


def create_feature_vector_of_history(a):

    a_as_one_digit = 0
    a_as_feature_vector = np.zeros((2**len(a),1))
    for i in range(len(a)):
        a_as_one_digit += a[len(a)-1-i] * (2 ** i)
    a_as_feature_vector[a_as_one_digit] = 1

    return a_as_feature_vector


def experiment_file_reader(history_length=6):

    y = np.asarray(np.loadtxt('predictions_y.txt', dtype=float))
    c = np.asarray(np.loadtxt('extra_state_setting.txt', dtype=int))
    w = np.asarray(np.loadtxt('weights_w.txt', dtype=float))
    history_observation = []
    history_action = []
    for i in range(history_length):
        history_observation.append(c[i])
        history_action.append(c[i+history_length])

    initial_position = [c[12], c[13]]
    initial_direction = Direction(c[14])

    return w,y, history_observation, history_action, initial_position, initial_direction


def save_to_file(weights, predictions, history_action, history_observation, agent_direction, agent_position):

    np.savetxt('weights_w.txt', weights, fmt='%f')
    np.savetxt('predictions_y.txt', predictions, fmt='%f')
    extra_setting = []
    extra_setting.extend(history_observation)
    extra_setting.extend(history_action)
    extra_setting.extend(agent_position)
    extra_setting.append(int(agent_direction))
    np.savetxt('extra_state_setting.txt', extra_setting, fmt='%d')
