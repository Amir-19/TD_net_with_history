from bit_to_bit_gridworld_env import *
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
    history_length = 6
    time_step = 0
    n = 62 # since we have td net with 5 layers and 2 actions 2^6  -  2  =  62
    m = 1 + (2 * (2**history_length)) + n # bias unit + 2 history (obs and action) + previous predictions
    step_size = Settings.step_size
    max_step = Settings.training_steps

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
        if Settings.activation_function == "identity":
            update = step_size*(np.outer(np.multiply(z-y,c).T,x))
        elif Settings.activation_function == "sigmoid":
            part1 = np.multiply(z-y,c)
            part2 = np.multiply(part1,y)
            part3 = np.multiply(part2,1-y)
            update = step_size*(np.outer(part3.T,x))

        W = W + update

        # next time step
        time_step += 1
        if time_step % 1000000 == 0:
            print(time_step)
    save_to_file(W,y,action_history,observation_history,environment.agent_direction,environment.agent_position)


if __name__ == "__main__":
    main()

