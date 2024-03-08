
class BimatrixGameSolver:
    def __init__(self, env, config, init_joint_state=None):
        self.env = env
        if init_joint_state is not None:
            # init state is list
            self.init_state = init_joint_state
        else:       
            self.init_state = [self.env.init_state['x0'], self.env.init_state['y0'], self.env.init_state['theta0'], self.env.init_state['x1'], self.env.init_state['y1'], self.env.init_state['theta1']]
        self.terminal_state = [self.env.goal_state['x0'], self.env.goal_state['y0'], self.env.goal_state['theta0'], self.env.goal_state['x1'], self.env.goal_state['y1'], self.env.goal_state['theta1']]
        
    def run_bimatrix_game(self):
        pass


