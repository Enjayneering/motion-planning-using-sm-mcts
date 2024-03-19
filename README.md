# Multi-Agent Motion Planning with Simultaneous Move MCTS

Welcome to the GitHub repository of this powerful Multi-Agent Motion Planning Algorithm! This project leverages the game theoretic approach of Simultaneous Move Monte Carlo Tree Search (SM-MCTS), as introduced by Lanctot, Lisy, and Winands in 2013, to compute robust and effective trajectories for agents operating within interactive environments. The implementation focuses on investigating about convergence and applicability in mobile robotics and is able to find discretized trajectories in an environment with two robots.

## Introduction

In dynamic environments where multiple agents must make decisions simultaneously, planning optimal trajectories becomes a significant challenge. Traditional methods often fall short in addressing the intricate dynamics and the necessity for agents to anticipate others' actions. This algorithm employs Simultaneous Move MCTS, a variant of the Monte Carlo Tree Search that is specifically designed for situations where agents make decisions concurrently. This method provides a powerful framework to model and solve multi-agent motion planning problems, yielding strong and adaptive trajectories.

## Key Features

- **Simultaneous Decision Making:** Incorporates the simultaneous move concept, allowing agents to plan their moves in consideration of others' actions.
- **Adaptive Trajectories:** Generates trajectories that are not only optimal with respect to the current state but also adaptable to the unfolding dynamics of the environment, since it is designed in a receding horizon and MPC-like fashion.
- **Interactive Environment Support:** Designed to work seamlessly in interactive settings such as closing doors or dynamic environments, enabling robust planning in the presence of unpredictable elements.

## Getting Started

To get started with our Multi-Agent Motion Planning Algorithm, please follow the instructions below:

### Prerequisites

Ensure you have the following installed:

- Ubuntu 22.04
- Python 3.8 or higher
- Any other dependencies listed in `requirements.txt`

For convenience it is possible to build a docker image and run the script within a container.

## Convergence behaviour of the implementation
![image.png](https://github.com/Enjayneering/motion-planning-using-sm-mcts/blob/master/strategy_convergence.png)

Note that due to the implementation in Python, this algorithm is quite slow...

## License
This project is licensed under the MIT License.
© 2024 Jannis Boening. All rights reserved.

## Related Paper
Lanctot, Marc; Lisy, Viliam; Winands, Mark H. M. (2013). Monte Carlo Tree Search in Simultaneous Move Games with Applications to Goofspiel.

