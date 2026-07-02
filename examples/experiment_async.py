"""Does asynchronous replanning break coordination symmetry? (Thesis test)

Setup: the symmetric head-on corridor, decentralized planning, 256
simulations per search (deliberately noisy equilibrium selection). Four
clock configurations, identical in everything except *when* the two agents
revise their strategies:

- sync_fast   periods (1,1)            — both replan every step (baseline)
- sync_slow   periods (2,2), in phase  — both commit for 2 steps and revise
                                         at the SAME instants
- async       periods (2,2), phase 0/1 — same frequency, but interleaved:
                                         each agent revises while the other
                                         is mid-commitment
- jitter      periods (2,2) + U{0,1}   — stochastic, non-phase-locked
                                         clocks (discrete Poisson stand-in)

Hypothesis (docs/ASYNC.md): simultaneous revision is what sustains the
both-dodge-the-same-way conflict; staggered clocks let the later reviser
react to the earlier one's committed behavior, so conflicts cannot persist.
Prediction: sync_slow >> sync_fast > async ≈ jitter ≈ 0 collision steps.

    python examples/experiment_async.py run [n_seeds]
    python examples/experiment_async.py report
"""

from __future__ import annotations

import json
import sys
from math import sqrt
from pathlib import Path

from sm_mcts_jax import AsyncDecentralizedPlanner, MCTSParams, ascii_world
from experiment_decentralized import HEAD_ON, fisher_exact, wilson_interval

RESULTS = Path(__file__).with_name("experiment_async_results.json")

PARAMS = MCTSParams(num_simulations=256)
MAX_STEPS = 40

CONDITIONS = {
    "sync_fast": dict(periods=[1, 1], phases=[0, 0], jitter=0),
    "sync_slow": dict(periods=[2, 2], phases=[0, 0], jitter=0),
    "async":     dict(periods=[2, 2], phases=[0, 1], jitter=0),
    "jitter":    dict(periods=[2, 2], phases=[0, 0], jitter=1),
}


def run(n_seeds: int) -> None:
    env = ascii_world(HEAD_ON)
    data: dict = {"n_seeds": n_seeds, "conditions": {}}
    for label, kwargs in CONDITIONS.items():
        episodes = []
        for seed in range(n_seeds):
            planner = AsyncDecentralizedPlanner(env, PARAMS, seed=seed, **kwargs)
            traj = planner.run_episode(max_steps=MAX_STEPS)
            episodes.append({
                "seed": seed,
                "success": traj.all_reached,
                "collision_episode": traj.any_collision,
                "collision_steps": int(traj.collisions.sum()),
                "steps": traj.n_steps,
                "searches": int(traj.replanned.sum()),
                # steps at which ALL agents revised simultaneously (beyond
                # the mandatory joint plan at t = 0)
                "coincidences": int(
                    (traj.replanned.sum(axis=1) == traj.replanned.shape[1])
                    .sum() - 1
                ),
            })
            print(f"{label} seed={seed}: {traj.summary()}", flush=True)
        data["conditions"][label] = episodes
    RESULTS.write_text(json.dumps(data, indent=2))
    print(f"saved -> {RESULTS}")


def report() -> None:
    data = json.loads(RESULTS.read_text())
    n = data["n_seeds"]
    print(f"\n=== asynchronous replanning, head-on corridor "
          f"(n = {n} seeds per condition, decentralized, 256 sims) ===")
    print(f"{'':12s}{'coll.episodes':>14s}{'95% CI':>16s}{'coll.steps':>12s}"
          f"{'success':>9s}{'steps':>13s}{'searches':>10s}{'coincid.':>10s}")
    agg = {}
    for label, eps in data["conditions"].items():
        ce = sum(e["collision_episode"] for e in eps)
        cs = sum(e["collision_steps"] for e in eps)
        succ = sum(e["success"] for e in eps)
        steps_ok = [e["steps"] for e in eps if e["success"]]
        mean = sum(steps_ok) / len(steps_ok) if steps_ok else float("nan")
        std = (sqrt(sum((s - mean) ** 2 for s in steps_ok) / len(steps_ok))
               if steps_ok else float("nan"))
        searches = sum(e["searches"] for e in eps) / len(eps)
        coincid = sum(e.get("coincidences", 0) for e in eps) / len(eps)
        lo, hi = wilson_interval(ce, n)
        agg[label] = ce
        print(f"{label:12s}{ce:>9d}/{n:<4d}{f'[{lo:.2f}, {hi:.2f}]':>16s}"
              f"{cs:>12d}{succ:>6d}/{n:<3d}{f'{mean:.1f} ± {std:.1f}':>13s}"
              f"{searches:>10.1f}{coincid:>10.1f}")
    for a, b in (("sync_fast", "sync_slow"), ("sync_fast", "jitter"),
                 ("sync_slow", "async"), ("sync_slow", "jitter"),
                 ("sync_fast", "async")):
        p = fisher_exact(agg[a], n - agg[a], agg[b], n - agg[b])
        print(f"Fisher exact p, collision episodes {a} vs {b}: {p:.4f}")


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] == "report":
        report()
        return
    n_seeds = int(sys.argv[2]) if len(sys.argv) > 2 else 40
    run(n_seeds)


if __name__ == "__main__":
    main()
