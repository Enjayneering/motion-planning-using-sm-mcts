"""Render the showcase GIFs in real-time playback.

Every animation frame is displayed for the episode's *maximum measured
planning time* on this machine, so watching a GIF gives an honest live
feeling for the CPU speed of each configuration — the worst step sets the
pace, no cherry-picking. Output goes to docs/media/showcase/.

    python examples/render_showcase.py            # all clips
    python examples/render_showcase.py gates_dec  # a single clip
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import jax

from sm_mcts_jax import DecentralizedPlanner, MCTSParams, Planner, ascii_world
from sm_mcts_jax.viz import animate_trajectory

OUT = Path(__file__).resolve().parent.parent / "docs" / "media" / "showcase"

INTERSECTION = """
    ##1##
    ##.##
    0...a
    ##.##
    ##b##
"""

HEAD_ON = """
    ###########
    #.........#
    #b0.....1a#
    #.........#
    ###########
"""

BOTTLENECK = """
    #########
    #0..#..1#
    #...#...#
    #.......#
    #...#...#
    #b..#..a#
    #########
"""

FOUR_AGENTS = """
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
"""

GATE_LEFT = """
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

GATE_RIGHT = GATE_LEFT.replace("##.########", "########.##")

P2 = MCTSParams(num_simulations=512)
P2_SAFE = MCTSParams(num_simulations=512, safety_filter=True)
P4 = MCTSParams(num_simulations=384, max_depth=10, rollout_depth=14,
                k_rollouts=2, c_uct=1.2)
PG = MCTSParams(num_simulations=512, max_depth=12, rollout_depth=16,
                k_rollouts=2, c_uct=1.2)
PG_SAFE = MCTSParams(num_simulations=512, max_depth=12, rollout_depth=16,
                     k_rollouts=2, c_uct=1.2, safety_filter=True)

# name -> (env factory, planner class, params, seed, max_steps)
CLIPS = {
    "01_intersection_central":    (lambda: ascii_world(INTERSECTION), Planner, P2, 0, 40),
    "02_head_on_central":         (lambda: ascii_world(HEAD_ON), Planner, P2, 0, 40),
    "03_head_on_dec":             (lambda: ascii_world(HEAD_ON), DecentralizedPlanner, P2, 3, 40),
    "04_head_on_dec_conflict":    (lambda: ascii_world(HEAD_ON), DecentralizedPlanner, P2, 17, 40),
    "05_head_on_dec_safe":        (lambda: ascii_world(HEAD_ON), DecentralizedPlanner, P2_SAFE, 17, 40),
    "06_bottleneck_central":      (lambda: ascii_world(BOTTLENECK), Planner, P2, 0, 40),
    "07_bottleneck_dec_safe":     (lambda: ascii_world(BOTTLENECK), DecentralizedPlanner, P2_SAFE, 0, 40),
    "08_four_agents_central":     (lambda: ascii_world(FOUR_AGENTS), Planner, P4, 0, 60),
    "09_gates_central":           (lambda: ascii_world([GATE_LEFT, GATE_RIGHT], frame_duration=4), Planner, PG, 0, 80),
    "10_gates_dec":               (lambda: ascii_world([GATE_LEFT, GATE_RIGHT], frame_duration=4), DecentralizedPlanner, PG, 0, 80),
    "11_gates_dec_safe":          (lambda: ascii_world([GATE_LEFT, GATE_RIGHT], frame_duration=4), DecentralizedPlanner, PG_SAFE, 0, 80),
}


def render(name: str) -> None:
    env_fn, cls, params, seed, max_steps = CLIPS[name]
    env = env_fn()
    planner = cls(env, params, seed=seed)
    planner.warmup()  # exclude one-time JIT compilation from step times
    # restore the RNG stream so episodes match the (warmup-free) experiments
    planner._rng = jax.random.PRNGKey(seed)
    t0 = time.perf_counter()
    traj = planner.run_episode(max_steps=max_steps)
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{name}.gif"
    animate_trajectory(env, traj, str(path), realtime=True)
    print(f"{name}: {traj.summary()}  "
          f"(episode wall time {time.perf_counter() - t0:.1f}s) -> {path.name}",
          flush=True)


def main() -> None:
    names = sys.argv[1:] or list(CLIPS)
    for name in names:
        render(name)


if __name__ == "__main__":
    main()
