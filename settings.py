# settings for the experiment


class Settings(object):

    step_size = 0.1
    history_length_observation = 5
    history_length_action = 5
    question_network_layer = 6
    activation_function = "sigmoid"  # the other option is "identity"
    training_steps = 500
