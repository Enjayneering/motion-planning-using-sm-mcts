# import modules
import os 
import sys
cwd = os.getcwd()
parent_dir = os.path.dirname(cwd)
sys.path.insert(0, parent_dir)

from utilities.common_utilities import *
from utilities.kinodynamic_utilities import *

class JointState:
    def __init__(self, joint_state):
        self.joint_state = [roundme(s) for s in joint_state]

    def get_next_state(self, joint_action, delta_t):
        # transition-function: get new state from previous state and chosen action
        next_state = []
        for i in range(len(joint_action)):
            state = self.joint_state[i]
            action = joint_action[i]
            next_state.append(kinematic_bicycle_model(state, action, delta_t))
        next_state = [mm_unicycle()]

        x0_new, y0_new, theta0_new = mm_unicycle(state_0, action_0, delta_t=delta_t)
        x1_new, y1_new, theta1_new = mm_unicycle(state_1, action_1, delta_t=delta_t)

        timestep_new = self.timestep + delta_t
        return State(x0_new, y0_new, theta0_new, x1_new, y1_new, theta1_new, timestep_new)

    def get_state(self, agent=0):
        if agent == 0:
            state_list = [self.x0, self.y0, self.theta0]
        elif agent == 1:
            state_list = [self.x1, self.y1, self.theta1]
        return state_list

    def get_state_together(self):
        state_list = [self.x0, self.y0, self.theta0, self.x1, self.y1, self.theta1]
        return state_list
    
    def get_timestep(self):
        return self.timestep