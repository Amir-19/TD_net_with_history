import numpy as np
from bit_to_bit_gridworld_env import *
from settings import *


def sigmoid(x):
    try:
        res = 1 / (1 + np.exp(-x))
    except OverflowError:
        res = 0.0
    return res


def condition(action, n):
    if action == 1:
        base_condition = [0, 1]
    elif action == 0:
        base_condition = [1, 0]

    condition_vec = []
    condition_vec.extend(base_condition for _ in range(int(int(n) / 2)))
    condition_vec = np.asarray(condition_vec).flatten().reshape(n, 1)
    return condition_vec


def calculate_targets(observation, prev_predictions):
    targets = np.zeros(len(prev_predictions))

    # the first two node are connected to the observation, so the target for those two is the observation
    targets[0] = observation
    targets[1] = observation

    # each node i is connected to nodes 2i and 2i + 1
    for i in range(int((len(prev_predictions) - 2) / 2)):
        targets[2 * (i + 1)] = prev_predictions[i]
        targets[2 * (i + 1) + 1] = prev_predictions[i]

    return targets.reshape(len(targets), 1)


def calculate_predictions(w, x):
    if Settings.activation_function == "sigmoid":
        return sigmoid(np.dot(w, x))
    elif Settings.activation_function == "identity":
        return np.dot(w, x)


def create_feature_vector(obs_history, action_history, predictions):
    x = []
    x = np.asarray(x)

    # adding the bias unit
    x = np.append(x, [1])

    # adding history of observations
    x = np.append(x, create_feature_vector_of_history(obs_history))

    # adding history of actions
    x = np.append(x, create_feature_vector_of_history(action_history))

    # adding the predictions from last time
    x = np.append(x, predictions)

    return x.reshape(len(x), 1)


def create_feature_vector_of_history(a):
    a_as_one_digit = 0
    a_as_feature_vector = np.zeros((2 ** len(a), 1))
    for i in range(len(a)):
        a_as_one_digit += a[len(a) - 1 - i] * (2 ** i)
    a_as_feature_vector[a_as_one_digit] = 1

    return a_as_feature_vector


def calculate_true_predictions(environment, indicator):
    true_targets = np.zeros((len(indicator), 1))
    for i in range(len(indicator)):
        true_targets[i] = environment.get_n_step_observation(indicator[i])
    return true_targets


def experiment_file_reader(history_length_action=Settings.history_length_action,
                           history_length_observation=Settings.history_length_observation):
    y = np.asarray(np.loadtxt('data/predictions_y.txt', dtype=float))
    c = np.asarray(np.loadtxt('data/extra_state_setting.txt', dtype=int))
    w = np.asarray(np.loadtxt('data/weights_w.txt', dtype=float))
    history_observation = []
    history_action = []
    for i in range(history_length_observation):
        history_observation.append(c[i])
    for i in range(history_length_observation, history_length_observation + history_length_action):
        history_action.append(c[i])
    new_index = history_length_observation + history_length_action
    initial_position = [c[new_index], c[new_index + 1]]
    initial_direction = Direction(c[new_index + 2])

    return w, y, history_observation, history_action, initial_position, initial_direction


def save_to_file(weights, predictions, history_action, history_observation, agent_direction, agent_position, rmse):
    np.savetxt('data/weights_w.txt', weights, fmt='%f')
    np.savetxt('data/predictions_y.txt', predictions, fmt='%f')
    np.savetxt('data/rmse.txt', rmse, fmt='%f')
    extra_setting = []
    extra_setting.extend(history_observation)
    extra_setting.extend(history_action)
    extra_setting.extend(agent_position)
    extra_setting.append(int(agent_direction))
    np.savetxt('data/extra_state_setting.txt', extra_setting, fmt='%d')
