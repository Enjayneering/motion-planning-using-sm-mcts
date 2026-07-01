"""Scenario: symmetric head-on encounter in a wide corridor.

Both agents start on the same row facing each other, each agent's goal lies
behind the other one. Neither can pass through the middle without a
collision, but the corridor leaves room above and below — the search has to
break the symmetry: one agent swerves, the other adapts its strategy.
"""

from sm_mcts_jax import MCTSParams, Planner, ascii_world
from sm_mcts_jax.viz import animate_trajectory, plot_trajectory

ENV = ascii_world("""
    ###########
    #.........#
    #b0.....1a#
    #.........#
    ###########
""")


def main(seed: int = 0):
    planner = Planner(ENV, MCTSParams(num_simulations=512), seed=seed)
    print(f"JIT compile: {planner.warmup():.1f}s")
    traj = planner.run_episode(max_steps=40, verbose=True)
    print(traj.summary())
    plot_trajectory(ENV, traj, "head_on_corridor_paths.png")
    animate_trajectory(ENV, traj, "head_on_corridor.gif")
    print("wrote head_on_corridor_paths.png and head_on_corridor.gif")
    return traj


if __name__ == "__main__":
    main()
