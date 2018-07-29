# settings for the experiment


class Settings(object):
    step_size = 0.1
    history_length_observation = 5
    history_length_action = 5
    question_network_layer = 6
    activation_function = "sigmoid"  # the other option is "identity"
    training_steps = int(1e7)
    error_measuring_frequency = int(1e5)
