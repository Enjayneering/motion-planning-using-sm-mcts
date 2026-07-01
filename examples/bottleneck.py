"""Scenario: two agents crossing through the same narrow gap.

Both agents have to pass the single opening in the dividing wall in opposite
directions at the same time — the equilibrium decides who goes first and who
yields in front of the bottleneck.
"""

from sm_mcts_jax import MCTSParams, Planner, ascii_world
from sm_mcts_jax.viz import animate_trajectory, plot_trajectory

ENV = ascii_world("""
    #########
    #0..#..1#
    #...#...#
    #.......#
    #...#...#
    #b..#..a#
    #########
""")


def main(seed: int = 0):
    planner = Planner(ENV, MCTSParams(num_simulations=512), seed=seed)
    print(f"JIT compile: {planner.warmup():.1f}s")
    traj = planner.run_episode(max_steps=40, verbose=True)
    print(traj.summary())
    plot_trajectory(ENV, traj, "bottleneck_paths.png")
    animate_trajectory(ENV, traj, "bottleneck.gif")
    print("wrote bottleneck_paths.png and bottleneck.gif")
    return traj


if __name__ == "__main__":
    main()
