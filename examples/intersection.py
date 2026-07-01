"""Two agents crossing a narrow intersection (the classic scenario of the
original SM-MCTS repository). One agent has to yield — the equilibrium found
by the simultaneous-move search."""

from sm_mcts_jax import MCTSParams, Planner, ascii_world
from sm_mcts_jax.viz import animate_trajectory, plot_trajectory

ENV = ascii_world("""
    ##1##
    ##.##
    0...a
    ##.##
    ##b##
""")


def main():
    planner = Planner(ENV, MCTSParams(num_simulations=512), seed=0)
    print(f"JIT compile: {planner.warmup():.1f}s")
    traj = planner.run_episode(max_steps=40, verbose=True)
    print(traj.summary())
    plot_trajectory(ENV, traj, "intersection_paths.png")
    animate_trajectory(ENV, traj, "intersection.gif")
    print("wrote intersection_paths.png and intersection.gif")


if __name__ == "__main__":
    main()
