from g import *
import collections
from utils import *


class PredictionMode(Enum):
    Prediction = 0
    TrueValue = 1
    AbsError = 2


class BitToBitGridWorldGUI:
    def __init__(self, _m, _n, _obstacles, _agent_position, _agent_direction, _w, _y, _history_observation,
                 _history_action, _history_length_observation, _history_length_action):

        # environment attributes
        self.m = _m
        self.n = _n
        self.obstacles = []
        for obstacle in _obstacles:
            self.obstacles.append(obstacle)
        self.agent_position = _agent_position
        self.agent_direction = _agent_direction
        self.environment = BitToBitGridWorld(self.m, self.n, self.obstacles, self.agent_position, self.agent_direction)

        # interactive problem controller
        self.prediciton_show_mode = PredictionMode.Prediction
        self.W = _w
        self.history_length_observation = _history_length_observation
        self.history_length_action = _history_length_action
        self.observation_history = collections.deque(_history_observation, self.history_length_observation)
        self.action_history = collections.deque(_history_action, self.history_length_action)
        self.y = _y.reshape(len(_y), 1)
        # prediction graphic stuff

        self.agent_prediction_direction = None
        self.obs_pred = None
        self.indicator = ["" for x in range(len(_y))]
        self.indicator[0] = "F"
        self.indicator[1] = "R"

        for i in range(int((len(_y) - 2) / 2)):
            self.indicator[2 * (i + 1)] = self.indicator[i] + "F"
            self.indicator[2 * (i + 1) + 1] = self.indicator[i] + "R"
        self.graphical_indicator_prediction = [None for x in range(len(_y))]
        self.tester_list = np.ones(len(_y))
        # graphics stuff
        self.xsize = _n + 2
        self.ysize = _m + 2
        self.blocksize = 30
        # first prediction
        if self.prediciton_show_mode == PredictionMode.AbsError:
            self.y = np.absolute(calculate_true_predictions(self.environment, self.indicator)-calculate_predictions(self.W,
                                       create_feature_vector(self.observation_history, self.action_history, self.y)))
        elif self.prediciton_show_mode == PredictionMode.TrueValue:
            self.y = calculate_true_predictions(self.environment, self.indicator)
        self.window = Gwindow(gdViewport=(20, 25, 800, 600))
        self.draw_gridworld()
        self.draw_history()
        self.agent_prediction = None
        self.draw_predictions()
        self.agent = None
        self.agent = self.draw_agent()
        self.add_control_menu()



        gMakeVisible(self.window)
        gMainLoop()

    """
        draw the prediction GUI
    """

    def draw_predictions(self):
        bgcolor = gColorRGB255(True, 213, 183, 145)
        # background
        gdFillRectR(self.window, 303, 33, 13 * self.blocksize, 13 * self.blocksize, bgcolor)

        # agent in the middle with correct direction
        offset_x = 5
        offset_y = 6
        pred_dir_x = 1
        pred_dir_y = 1
        x_y_inverse = False
        if self.agent_prediction is not None:
            gDelete(self.window, self.agent_prediction)
        self.agent_prediction_direction = self.environment.agent_direction
        if self.agent_prediction_direction == Direction.East:
            self.agent_prediction = gdDrawWedge(self.window,
                                                300 + ((offset_x + 2) * self.blocksize) - (self.blocksize / 5),
                                                30 + ((offset_y) * self.blocksize) + (self.blocksize / 2), 20, 160, 40,
                                                'red')
            pred_dir_x = 1
            pred_dir_y = 1
            x_y_inverse = False
        elif self.agent_prediction_direction == Direction.South:
            self.agent_prediction = gdDrawWedge(self.window,
                                                300 + ((offset_x + 1) * self.blocksize) + (self.blocksize / 2),
                                                30 + ((offset_y + 1) * self.blocksize) - (self.blocksize / 5), 20, 70,
                                                40, 'red')
            pred_dir_x = 0
            pred_dir_y = 1
            x_y_inverse = True
        elif self.agent_prediction_direction == Direction.West:
            self.agent_prediction = gdDrawWedge(self.window,
                                                300 + ((offset_x + 1) * self.blocksize) + (self.blocksize / 5),
                                                30 + ((offset_y) * self.blocksize) + (self.blocksize / 2), 20, -20, 40,
                                                'red')
            pred_dir_x = 0
            pred_dir_y = 0
            x_y_inverse = False
        elif self.agent_prediction_direction == Direction.North:
            self.agent_prediction = gdDrawWedge(self.window,
                                                300 + ((offset_x + 1) * self.blocksize) + (self.blocksize / 2),
                                                30 + ((offset_y) * self.blocksize) + (self.blocksize / 5), 20, 250, 40,
                                                'red')
            pred_dir_x = 1
            pred_dir_y = 0
            x_y_inverse = True

        # draw the predictions and observation in the prediction GUI

        pred_x = 6
        pred_y = 6

        for i in range(len(self.indicator) - 1, -1, -1):
            if self.tester_list[i] == 1:
                prediction_value = min(self.y[i][0], 1)
                prediction_value = max(0, prediction_value)
                pred_color = gColorBW(True, prediction_value)
                a, b, c, d = self.get_the_gridcell_for_prediction(self.indicator[i])
                if self.graphical_indicator_prediction[i] is not None:
                    gDelete(self.window, self.graphical_indicator_prediction[i])
                if a == 1 and b == 0:
                    # east
                    self.graphical_indicator_prediction[i] = gdFillRectR(self.window, 300 + ((c + a) * self.blocksize),
                                                                         30 + ((d) * self.blocksize), 3, self.blocksize,
                                                                         pred_color)
                elif a == 0 and b == 1:
                    # south
                    self.graphical_indicator_prediction[i] = gdFillRectR(self.window, 300 + ((c) * self.blocksize),
                                                                         30 + ((d + b) * self.blocksize),
                                                                         self.blocksize, 3, pred_color)
                elif a == -1 and b == 0:
                    # west
                    self.graphical_indicator_prediction[i] = gdFillRectR(self.window,
                                                                         300 + ((c + a + 1) * self.blocksize),
                                                                         30 + ((d) * self.blocksize), 3, self.blocksize,
                                                                         pred_color)
                elif a == 0 and b == -1:
                    # north
                    self.graphical_indicator_prediction[i] = gdFillRectR(self.window, 300 + ((c) * self.blocksize),
                                                                         30 + ((d + b + 1) * self.blocksize),
                                                                         self.blocksize, 3, pred_color)
        obs_color = gColorBW(True, self.environment.get_observation())
        if self.obs_pred is not None:
            gDelete(self.window, self.obs_pred)
        if x_y_inverse:
            # horizantal
            self.obs_pred = gdFillRectR(self.window, 300 + ((pred_x) * self.blocksize),
                                        30 + ((pred_y + pred_dir_y) * self.blocksize), self.blocksize, 3, obs_color)
        else:
            # vertical
            self.obs_pred = gdFillRectR(self.window, 300 + ((pred_x + pred_dir_x) * self.blocksize),
                                        30 + ((pred_y) * self.blocksize), 3, self.blocksize, obs_color)

    """
        get the position of a specific prediction, y_{i}, at the prediction GUI
    """

    def get_the_gridcell_for_prediction(self, indi):
        pred_x = 6
        pred_y = 6
        if self.agent_prediction_direction == Direction.East:
            add_x = 1
            add_y = 0
        elif self.agent_prediction_direction == Direction.South:
            add_x = 0
            add_y = 1
        elif self.agent_prediction_direction == Direction.West:
            add_x = -1
            add_y = 0
        elif self.agent_prediction_direction == Direction.North:
            add_x = 0
            add_y = -1
        for i in indi:
            if i == "F":
                pred_x += add_x
                pred_y += add_y
            elif i == "R":
                if add_x == 1 and add_y == 0:
                    add_x = 0
                    add_y = 1
                elif add_x == 0 and add_y == 1:
                    add_x = -1
                    add_y = 0
                elif add_x == -1 and add_y == 0:
                    add_x = 0
                    add_y = -1
                elif add_x == 0 and add_y == -1:
                    add_x = 1
                    add_y = 0

        return add_x, add_y, pred_x, pred_y

    """
        draw the history
    """

    def draw_history(self):
        bgcolor = gColorRGB255(True, 213, 183, 145)
        gdFillRectR(self.window, 35, 380,
                    max(self.history_length_action, self.history_length_observation) * self.blocksize,
                    4 * self.blocksize - 15, bgcolor)
        gdDrawTextCentered(self.window, 'at-1', ('Helvetica', 9, 'normal'), 52, 390)
        padding = 10
        for i in range(self.history_length_action):
            if self.action_history[i] == 1:
                label = 'at' + str(-(i + 1))
                gdDrawTextCentered(self.window, label, ('Helvetica', 9, 'normal'), 40 + 12 + (i) * self.blocksize, 390)
                gdFillRectR(self.window, 40 + (i) * self.blocksize, 400, self.blocksize - padding,
                            self.blocksize - padding, 'black')
            elif self.action_history[i] == 0:
                label = 'at' + str(-(i + 1))
                gdDrawTextCentered(self.window, label, ('Helvetica', 9, 'normal'), 40 + 12 + (i) * self.blocksize, 390)
                gdFillRectR(self.window, 40 + (i) * self.blocksize, 400, self.blocksize - padding,
                            self.blocksize - padding, 'white')
        for i in range(self.history_length_observation):
            if self.observation_history[i] == 1:
                label = 'ot' + str(-(i + 1))
                gdDrawTextCentered(self.window, label, ('Helvetica', 9, 'normal'), 40 + 12 + (i) * self.blocksize, 440)
                gdFillRectR(self.window, 40 + (i) * self.blocksize, 450, self.blocksize - padding,
                            self.blocksize - padding, 'black')
            elif self.observation_history[i] == 0:
                label = 'ot' + str(-(i + 1))
                gdDrawTextCentered(self.window, label, ('Helvetica', 9, 'normal'), 40 + 12 + (i) * self.blocksize, 440)
                gdFillRectR(self.window, 40 + (i) * self.blocksize, 450, self.blocksize - padding,
                            self.blocksize - padding, 'white')

    """
        draw the gridworld
    """

    def draw_gridworld(self):
        gdFillRectR(self.window, 0, 0, self.xsize * self.blocksize - 1, self.ysize * self.blocksize, 'white')
        for i in range(1, self.xsize):
            gdDrawLine(self.window, i * self.blocksize, 30, i * self.blocksize, self.ysize * self.blocksize - 30,
                       'grey')
        for i in range(1, self.ysize):
            gdDrawLine(self.window, 30, i * self.blocksize, self.xsize * self.blocksize - 30, i * self.blocksize,
                       'grey')

        for i in range(self.m):
            for j in range(self.n):
                if [i, j] in self.obstacles:
                    gdFillRectR(self.window, (j + 1) * self.blocksize, (self.m - i) * self.blocksize,
                                self.blocksize - 1,
                                self.blocksize - 1, 'black')

    """
        draw the agent
    """

    def draw_agent(self):

        gDelete(self.window, self.agent)
        self.agent_direction = self.environment.agent_direction
        self.agent_position = self.environment.agent_position

        if self.agent_direction == Direction.East:
            self.agent = gdDrawWedge(self.window,
                                     ((self.agent_position[1] + 2) * self.blocksize) - (self.blocksize / 5),
                                     ((self.m - self.agent_position[0]) * self.blocksize) + (self.blocksize / 2), 20,
                                     160, 40, 'red')
        elif self.agent_direction == Direction.South:
            self.agent = gdDrawWedge(self.window,
                                     ((self.agent_position[1] + 1) * self.blocksize) + (self.blocksize / 2),
                                     ((self.m - self.agent_position[0] + 1) * self.blocksize) - (self.blocksize / 5),
                                     20, 70, 40, 'red')
        elif self.agent_direction == Direction.West:
            self.agent = gdDrawWedge(self.window,
                                     ((self.agent_position[1] + 1) * self.blocksize) + (self.blocksize / 5),
                                     ((self.m - self.agent_position[0]) * self.blocksize) + (self.blocksize / 2), 20,
                                     -20, 40, 'red')
        elif self.agent_direction == Direction.North:
            self.agent = gdDrawWedge(self.window,
                                     ((self.agent_position[1] + 1) * self.blocksize) + (self.blocksize / 2),
                                     ((self.m - self.agent_position[0]) * self.blocksize) + (self.blocksize / 5), 20,
                                     250, 40, 'red')

        gMakeVisible(self.window)
        return self.agent

    """
        move the agent forward in the GUI and update the predictions
    """

    def move_agent_forward_gui(self):
        self.environment.move_forward()
        self.draw_agent()
        self.action_history.appendleft(0)
        self.observation_history.appendleft(self.environment.get_observation())
        self.draw_history()
        if self.prediciton_show_mode == PredictionMode.Prediction:
            self.y = calculate_predictions(self.W,
                                       create_feature_vector(self.observation_history, self.action_history, self.y))
        elif self.prediciton_show_mode == PredictionMode.AbsError:
            self.y = np.absolute(calculate_true_predictions(self.environment, self.indicator)-calculate_predictions(self.W,
                                       create_feature_vector(self.observation_history, self.action_history, self.y)))
        elif self.prediciton_show_mode == PredictionMode.TrueValue:
            self.y = calculate_true_predictions(self.environment, self.indicator)
        self.draw_predictions()

    """
        rotate the agent in the GUI and update the predictions
    """

    def rotate_agent_gui(self):
        self.environment.turn_clockwise()
        self.draw_agent()
        self.action_history.appendleft(1)
        self.observation_history.appendleft(self.environment.get_observation())
        self.draw_history()
        if self.prediciton_show_mode == PredictionMode.Prediction:
            self.y = calculate_predictions(self.W,
                                       create_feature_vector(self.observation_history, self.action_history, self.y))
        elif self.prediciton_show_mode == PredictionMode.AbsError:
            self.y = np.absolute(calculate_true_predictions(self.environment, self.indicator)-calculate_predictions(self.W,
                                       create_feature_vector(self.observation_history, self.action_history, self.y)))
        elif self.prediciton_show_mode == PredictionMode.TrueValue:
            self.y = calculate_true_predictions(self.environment, self.indicator)
        self.draw_predictions()

    """
        draw the control menu (buttons)
    """

    def add_control_menu(self):
        back_color = gColorLightGray(self.window)
        gdAddButton(self.window, 'Step Forward', self.move_agent_forward_gui, 70, 275, back_color)
        gdAddButton(self.window, 'Turn CW', self.rotate_agent_gui, 70, 310, back_color)


def main():
    # gui creation
    history_length_observation = Settings.history_length_observation
    history_length_aciton = Settings.history_length_action
    w, y, history_observation, history_action, initial_position, initial_direction = experiment_file_reader()
    m = 6
    n = 6
    obstacles = [[3, 1], [3, 2], [4, 2], [3, 3], [2, 3], [4, 5], [3, 5], [2, 5], [1, 5], [0, 5]]
    gui = BitToBitGridWorldGUI(m, n, obstacles, initial_position, initial_direction, w, y, history_observation,
                               history_action, history_length_observation, history_length_aciton)


if __name__ == "__main__":
    main()
