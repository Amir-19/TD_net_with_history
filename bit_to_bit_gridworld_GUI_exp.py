from bit_to_bit_gridworld_env import *
from g import *




class BitToBitGridWorldGUI():

    def __init__(self, _m, _n, _obstacles, _agent_position, _agent_direction):

        # environment attributes
        self.m = _m
        self.n = _n
        self.obstacles = []
        for obstacle in _obstacles:
            self.obstacles.append(obstacle)
        self.agent_position = _agent_position
        self.agent_direction = _agent_direction
        self.environment = BitToBitGridWorld(self.m, self.n, self.obstacles, self.agent_position, self.agent_direction)

        # graphics stuff
        self.xsize = _n+2
        self.ysize = _m+2
        self.blocksize = 30
        self.window = Gwindow(gdViewport = (20, 25, 800, 600))
        self.draw_gridworld()
        self.agent = None
        self.agent = self.draw_agent()
        self.add_control_menu()
        gMakeVisible(self.window)
        gMainLoop()

    """
        draw the gridworld
    """
    def draw_gridworld(self):
        gdFillRectR(self.window, 0,0, self.xsize*self.blocksize-1,self.ysize*self.blocksize,'white')
        for i in range(1,self.xsize):
            gdDrawLine(self.window, i*self.blocksize,30,i*self.blocksize, self.ysize*self.blocksize-30,'grey')
        for i in range(1,self.ysize):
            gdDrawLine(self.window, 30, i*self.blocksize, self.xsize*self.blocksize-30, i*self.blocksize, 'grey')

        for i in range(self.m):
            for j in range(self.n):
                if [i,j] in self.obstacles:
                    gdFillRectR(self.window,(j+1)*self.blocksize, (self.m - i)*self.blocksize,self.blocksize-1,
                                self.blocksize-1,'black')

    """
        draw the agent
    """
    def draw_agent(self):

        gDelete(self.window, self.agent)
        self.agent_direction = self.environment.agent_direction
        self.agent_position = self.environment.agent_position
        
        if self.agent_direction == Direction.East:
            self.agent = gdDrawWedge(self.window, ((self.agent_position[1]+2)*self.blocksize)-(self.blocksize/5),
                                  ((self.m-self.agent_position[0])*self.blocksize)+(self.blocksize/2), 20, 160, 40)
        elif self.agent_direction == Direction.South:
            self.agent = gdDrawWedge(self.window, ((self.agent_position[1]+1)*self.blocksize)+(self.blocksize/2),
                                  ((self.m-self.agent_position[0]+1)*self.blocksize)-(self.blocksize/5), 20, 70, 40)
        elif self.agent_direction == Direction.West:
            self.agent = gdDrawWedge(self.window, ((self.agent_position[1]+1)*self.blocksize)+(self.blocksize/5),
                                  ((self.m-self.agent_position[0])*self.blocksize)+(self.blocksize/2), 20, -20, 40)
        elif self.agent_direction == Direction.North:
            self.agent = gdDrawWedge(self.window, ((self.agent_position[1]+1)*self.blocksize)+(self.blocksize/2),
                                  ((self.m - self.agent_position[0])*self.blocksize)+(self.blocksize/5), 20, 250, 40)

        gMakeVisible(self.window)
        return self.agent

    def move_agent_forward_gui(self):
        self.environment.move_forward()
        self.draw_agent()

    def rotate_agent_gui(self):
        self.environment.turn_clockwise()
        self.draw_agent()
        print("haha")

    """
        draw the control menu (buttons)
    """
    def add_control_menu(self):
        back_color = gColorLightGray(self.window)
        gdAddButton(self.window, 'Step Forward', self.move_agent_forward_gui, 70, 275, back_color)
        gdAddButton(self.window, 'Turn CW', self.rotate_agent_gui, 70, 310, back_color)

def main():

    # gui creation
    m = 6
    n = 6
    obstacles = [[3, 1], [3, 2], [4, 2], [3, 3], [2, 3], [4, 5], [3, 5], [2, 5], [1, 5], [0, 5]]
    initial_position = [0, 4]
    initial_direction = Direction.West
    gui = BitToBitGridWorldGUI(m,n,obstacles,initial_position,initial_direction)

if __name__ == "__main__":
    main()