"""Scenario: four agents crossing a wall with alternating gates.

All four agents must cross the dividing wall to reach the opposite side.
The wall has two gates that alternate every `frame_duration` timesteps:
first the left gate is open, then the right one. The search plans with the
occupancy grid that will be active at each future timestep, so agents time
their crossing (or wait) to match the gate schedule.
"""

from sm_mcts_jax import MCTSParams, Planner, ascii_world
from sm_mcts_jax.viz import animate_trajectory, plot_trajectory

GATE_LEFT_OPEN = """
    0.........1
    d.........c
    ...........
    ...........
    ##.########
    ...........
    ...........
    b.........a
    2.........3
"""

GATE_RIGHT_OPEN = """
    0.........1
    d.........c
    ...........
    ...........
    ########.##
    ...........
    ...........
    b.........a
    2.........3
"""

ENV = ascii_world([GATE_LEFT_OPEN, GATE_RIGHT_OPEN], frame_duration=4, cycle=True)


def main(seed: int = 0):
    params = MCTSParams(
        num_simulations=512,
        max_depth=12,
        rollout_depth=16,
        k_rollouts=2,
        c_uct=1.2,
    )
    planner = Planner(ENV, params, seed=seed)
    print(f"JIT compile: {planner.warmup():.1f}s")
    traj = planner.run_episode(max_steps=60, verbose=True)
    print(traj.summary())
    plot_trajectory(ENV, traj, "dynamic_gates_paths.png")
    animate_trajectory(ENV, traj, "dynamic_gates.gif")
    print("wrote dynamic_gates_paths.png and dynamic_gates.gif")
    return traj


if __name__ == "__main__":
    main()
