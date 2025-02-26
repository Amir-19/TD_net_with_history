from enum import Enum


class Direction(Enum):
    North = 1
    East = 2
    South = 3
    West = 4

    def __int__(self):
        return self.value


class Action(Enum):
    Forward = 0
    Rotate = 1


class Observation(Enum):
    White = 0
    Black = 1


class BitToBitGridWorld:
    """
        create an m*n gridworld
        _m: rows
        _n: columns
        _obstacles: list of initial obstacles in form of [[x,y],...]
        _agent_position: the agent's current position on the grid
        _agent_direction: the agent's current direction
    """

    def __init__(self, _m, _n, _obstacles, _agent_position, _agent_direction):

        # environment attributes
        self.m = _m
        self.n = _n
        self.obstacles = []
        for obstacle in _obstacles:
            self.obstacles.append(obstacle)
        self.agent_position = _agent_position
        self.agent_direction = _agent_direction
        self.true_target_dict = dict()

    """
        check whether a position is in the grid or not. It also checks whether the position is
        on of the obstacles or not. In other words, it checks whether agent can be on the
        specific position or not.

        _position: the position to check
    """

    def is_in_grid(self, _position):

        if _position[0] < 0 or _position[0] >= self.m or _position[1] < 0 or _position[1] >= self.n:
            return False
        if _position in self.obstacles:
            return False
        return True

    """
        add an obstacle to the list of obstacles
    """

    def add_obstacle(self, _obstacle):

        self.obstacles.append(_obstacle)

    """
        moves agent one step forward in the direction it is heading
    """

    def move_forward(self):

        next_position = self.agent_position.copy()
        if self.agent_direction == Direction.North:
            next_position[0] += 1
        elif self.agent_direction == Direction.East:
            next_position[1] += 1
        elif self.agent_direction == Direction.South:
            next_position[0] -= 1
        elif self.agent_direction == Direction.West:
            next_position[1] -= 1

        if self.is_in_grid(next_position):
            self.agent_position = next_position

    """
        turn agent's direction clockwise
    """

    def turn_clockwise(self):

        if self.agent_direction == Direction.North:
            self.agent_direction = Direction.East

        elif self.agent_direction == Direction.East:
            self.agent_direction = Direction.South

        elif self.agent_direction == Direction.South:
            self.agent_direction = Direction.West

        elif self.agent_direction == Direction.West:
            self.agent_direction = Direction.North

    """
        return the agent's observation which is 1 if facing an obstacle, 0 otherwise
    """

    def get_observation(self):

        next_position = self.agent_position.copy()
        if self.agent_direction == Direction.North:
            next_position[0] += 1
        elif self.agent_direction == Direction.East:
            next_position[1] += 1
        elif self.agent_direction == Direction.South:
            next_position[0] -= 1
        elif self.agent_direction == Direction.West:
            next_position[1] -= 1

        if not self.is_in_grid(next_position):
            return 1
        else:
            return 0

    """
        return the agent's observation from a specific position and direction.
        which is 1 if facing an obstacle, 0 otherwise
    """

    def get_observation_in_pos(self, agent_pos, agent_dir):

        next_position = agent_pos.copy()
        direction = agent_dir
        if direction == Direction.North:
            next_position[0] += 1
        elif direction == Direction.East:
            next_position[1] += 1
        elif direction == Direction.South:
            next_position[0] -= 1
        elif direction == Direction.West:
            next_position[1] -= 1

        if not self.is_in_grid(next_position):
            return 1
        else:
            return 0

    """
        return the agent observation after a series of actions, this helps to calculate the td network node error in
        which this is used as the oracle value for the node.
    """

    def get_n_step_observation(self, sequence):

        direction = self.agent_direction
        position = self.agent_position.copy()

        # first check whether we have the true observation stored or not, if not we calculate it and store it.
        key = (position[0], position[1], direction, sequence)
        if key in self.true_target_dict:
            return self.true_target_dict[key]
        else:
            for action in sequence:
                if action == "R":
                    if direction == Direction.North:
                        direction = Direction.East
                    elif direction == Direction.East:
                        direction = Direction.South
                    elif direction == Direction.South:
                        direction = Direction.West
                    elif direction == Direction.West:
                        direction = Direction.North
                elif action == "F":
                    next_position = position.copy()

                    if direction == Direction.North:
                        next_position[0] += 1
                    elif direction == Direction.East:
                        next_position[1] += 1
                    elif direction == Direction.South:
                        next_position[0] -= 1
                    elif direction == Direction.West:
                        next_position[1] -= 1

                    if self.is_in_grid(next_position):
                        position = next_position.copy()

            self.true_target_dict[key] = self.get_observation_in_pos(position, direction)
            return self.true_target_dict[key]
