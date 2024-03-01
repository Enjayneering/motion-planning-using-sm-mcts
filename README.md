# MCTS-Based Game Theoretic Motion Planner for Mobile Agents

Code Conventions:

# Data structures
n: number of agents
s: number of state components like (x,y,theta)
t: number of timesteps of the trajectory
a: number of action component like (vel, ang_vel)
p_i: number of different components of intermediate payoffs
p_t: number of different components of terminal payoffs

Payoff weights: 2D array | (n, p_i) or (n,p_t)
Joint state: 2D array | (n, s)
Joint state trajectory: 3D array | (n, s, t)
Joint action: 2D array | (n, a)
Storing multiple joint actions: 3D array | (n, a, depth) with "depth" being the number of different actions 
Agent progress lines: Dict of Dicts with each tree and progress list{agent_ix: {KDtree: tree, Progress: list}}