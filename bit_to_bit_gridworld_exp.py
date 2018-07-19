from utils import *
import collections
import numpy as np
from settings import *


def main():
    # create the environment
    environment = BitToBitGridWorld(6, 6, [[3, 1], [3, 2], [4, 2], [3, 3], [2, 3],
                                           [4, 5], [3, 5], [2, 5], [1, 5], [0, 5]], [0, 4], Direction.West)
    num_actions = 2

    # the td net with history attributes
    history_length_observation = Settings.history_length_observation
    history_length_action = Settings.history_length_action
    n = 62 # since we have td net with 5 layers and 2 actions 2^6  -  2  =  62
    m = 1 + ((2**history_length_action)+ (2**history_length_observation)) + n # bias unit + 2 history (obs and action) + previous predictions
    step_size = Settings.step_size
    max_step = Settings.training_steps

    # 1. W_{1}
    W = np.full((n, m),0)
    # 1. y_{0}
    y = np.ones((n,1))*0.5

    # 2. t = 0 it is actually 1 in the algorithm but 0 here (?)
    time_step = 0
    # setting up the history
    observation_history = collections.deque(history_length_observation*[None], history_length_observation)
    action_history = collections.deque(history_length_action*[None], history_length_action)

    last_observation = None
    last_action = None

    # 3. choose a_{t-1} to observe o_{t} but since we have a history length we need to do this to fill the history
    # now we take as many actions to see a full history to create the state
    # after having all the elements of both history queues not None we can start updating
    for i in range(max(history_length_action,history_length_observation)):
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

        # go to the next time step
        time_step += 1
    for i in range(max_step - max(history_length_action,history_length_observation)):

        # 4. x_{t}
        x = create_feature_vector(observation_history, action_history, y)

        # 5. y_{t} = σ(W_{t}x_{t})
        y = calculate_predictions(W,x)

        # 6. choose a_{t} and observe o_{t+1}
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

        # 7. c_{t} = c(a_{t},y_{t})
        c = condition(a, n)

        # 8. x_{t+1} = x(a_{t},o_{t+1},y_{t})
        xtp1 = create_feature_vector(observation_history, action_history, y)

        # 9. ỹ_{t+1} = σ(W_{t}x_{t+1})
        ỹ = calculate_predictions(W, xtp1)

        # 10. z_{t}
        z = calculate_targets(last_observation, ỹ)

        # 11. update weights W
        if Settings.activation_function == "identity":
            update = step_size*(np.outer(np.multiply(z-y,c).T,x))
        elif Settings.activation_function == "sigmoid":
            part1 = np.multiply(z-y,c)
            part2 = np.multiply(part1,y)
            part3 = np.multiply(part2,1-y)
            update = step_size*(np.outer(part3.T,x))

        W = W + update

        # 12. t= t+1
        time_step += 1
        if time_step % 1000000 == 0:
            print(time_step)
    save_to_file(W, y, action_history, observation_history, environment.agent_direction, environment.agent_position)


if __name__ == "__main__":
    main()

