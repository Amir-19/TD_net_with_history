from bit_to_bit_gridworld_env import *
from utils import *
import collections
from td_net_with_history_utils import *
import numpy as np


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

    return x.reshape(len(x),1)


def condition(action, n):
    if action == 1:
        base_condition = [0, 1]
    elif action == 0:
        base_condition = [1, 0]

    condition_vec = []
    condition_vec.extend(base_condition for _ in range(int(int(n)/2)))
    condition_vec = np.asarray(condition_vec).flatten().reshape(n,1)
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

    return targets.reshape(len(targets),1)


def calculate_predictions(W, x):
    return np.dot(W,x)
    # return sigmoid(np.dot(W,x))


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
    m = 1 + (2 * (2**history_length)) + n # bias unit + 2 history (obs and action) + previous predictions
    step_size = 0.1
    max_step = 100000
    # 1. W_{t}
    W = np.full((n, m),0)
    # 2. y_{t}
    y = np.ones((n,1))*0.5

    # setting up the history
    observation_history = collections.deque(history_length*[None], history_length)
    action_history = collections.deque(history_length*[None], history_length)

    last_observation = None
    last_action = None

    # now we take as many actions to see a full history to create the state
    # after having all the elements of both history queues not None we can start updating
    for i in range(history_length):
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
        time_step += 1

    for i in range(max_step - history_length):

        # 3. a_{t}
        a = action_history[0]

        # 4. x_{t}
        x = create_feature_vector(observation_history,action_history,y)

        # 5. y_{t} = σ(W_{t}x_{t})
        y = calculate_predictions(W,x)

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
        a = action
        # 6. c_{t}
        c = condition(a,n)

        # 7. x_{t+1}
        xtp1 = create_feature_vector(observation_history,action_history,y)

        # 8. ỹ_{t+1} =
        ỹ = calculate_predictions(W,xtp1)

        # 9. z_{t}
        z = calculate_targets(last_observation,ỹ)

        # 10. update weights W
        update = step_size*(np.outer(np.multiply(z-y,c).T,x))
        W = W + update

        # next time step
        time_step += 1

    save_to_file(y,action_history,observation_history,environment.agent_direction,environment.agent_position)


if __name__ == "__main__":
    main()

