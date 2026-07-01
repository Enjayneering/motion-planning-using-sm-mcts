"""Four agents swapping positions through a room with a central obstacle.

Demonstrates that the JAX implementation scales beyond the two-agent case of
the original codebase: the decoupled statistics stay per-agent, only the
joint child index grows with the number of agents.
"""

from sm_mcts_jax import MCTSParams, Planner, ascii_world
from sm_mcts_jax.viz import animate_trajectory, plot_trajectory

ENV = ascii_world("""
    0.......c
    .........
    .........
    ....##...
    2...##..a
    .........
    .........
    d.......1
    .........
    b.......3
""")


def main():
    params = MCTSParams(
        num_simulations=384,
        max_depth=10,
        rollout_depth=14,
        k_rollouts=2,
        c_uct=1.2,
    )
    planner = Planner(ENV, params, seed=0)
    print(f"JIT compile: {planner.warmup():.1f}s")
    traj = planner.run_episode(max_steps=60, verbose=True)
    print(traj.summary())
    plot_trajectory(ENV, traj, "four_agents_paths.png")
    animate_trajectory(ENV, traj, "four_agents.gif")
    print("wrote four_agents_paths.png and four_agents.gif")


if __name__ == "__main__":
    main()
