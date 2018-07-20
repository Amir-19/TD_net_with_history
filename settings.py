# settings for the experiment


class Settings(object):

    step_size = 0.1
    history_length_observation = 6
    history_length_action = 6
    question_network_layer = 5
    activation_function = "sigmoid"  # the other option is "identity"
    training_steps = 50000000
