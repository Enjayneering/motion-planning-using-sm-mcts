# The simulation ↔ planner interface (v1)

The web simulation (`sm_mcts_web/`) and the planning algorithm are
decoupled by design: the research object (the planner) must be swappable
without touching the environment, and the environment must be able to
evolve without breaking planners. This document is the contract.

## Architecture

```
Browser (canvas editor + renderer + keyboard)
   │  WebSocket JSON: start / input / stop  ← → status / tick / error
FastAPI server (sm_mcts_web/server.py)
   │
SimSession (session.py)          20 Hz continuous loop, real-time paced
   ├─ AckermannCar per agent     kinematic bicycle model (vehicles.py)
   ├─ PurePursuit per AI car     waypoint tracking + reverse recovery
   └─ PlannerAdapter  ◄──────────  THE swappable boundary (interface.py)
          │ reset(ScenarioSpec)          once, static world
          │ plan(WorldSnapshot) -> {agent_id: Route}   on its own cadence
          ▼
SMMCTSAdapter (bridge.py)        reference implementation:
   rasterize -> occupancy grid -> sm_mcts_jax.search -> metric waypoints
```

## The contract, precisely

1. **Only the interface.** The simulator calls nothing on the planner but
   `reset()` and `plan()`; the planner sees nothing but `ScenarioSpec`
   and `WorldSnapshot` (dataclasses in `sm_mcts_web/interface.py`,
   `INTERFACE_VERSION = 1`).
2. **Waypoints out, never actuators.** A planner returns metric
   `Route`s (ordered waypoints + speed cap). The vehicle model and its
   controller belong to the simulation. Swapping the vehicle (drone,
   diff-drive) therefore never touches planners, and swapping the planner
   never touches vehicles.
3. **Time budget, isolation, graceful degradation.** `plan()` runs in a
   worker thread on the cadence `PlannerConfig.replan_period_s`. A late
   planner is not an error — vehicles keep tracking their last routes. A
   *crashing* planner is contained: the exception is logged, counted in
   the stats, and the world keeps running.
4. **Read-only snapshots, no call-frequency assumptions.** Planners must
   not mutate snapshots and must tolerate any interval between calls.
5. **Versioning.** `ScenarioSpec.version` is checked on parse; breaking
   schema changes bump `INTERFACE_VERSION` and this document.

## Writing your own planner

```python
from sm_mcts_web.interface import PlannerAdapter, Route

class MyPlanner(PlannerAdapter):
    def reset(self, scenario):
        self.scenario = scenario          # rasterize / precompute here

    def plan(self, snapshot):
        return {
            agent.id: Route(waypoints=((x1, y1), (x2, y2)), speed=5.0)
            for agent in self.scenario.agents if agent.kind == "ai"
        }
```

Plug it in by instantiating your adapter in `server.py`'s websocket
handler (one line), or write a small custom entry point.

## Wire format (browser ↔ server)

```jsonc
// client -> server
{"type": "start", "scenario": {
  "version": 1, "width_m": 64, "height_m": 40,
  "obstacles": [{"kind": "house", "x": 12, "y": 32,
                  "width": 7, "height": 5.5, "rotation": 0,
                  "blocking": true}],
  "agents": [{"id": "car1", "kind": "ai", "start": [6, 24.5, 0],
               "goal": [58, 24.5], "behavior": "normal",
               "max_speed": 6, "radius": 0.95},
              {"id": "human", "kind": "human",
               "start": [6, 15, 0], "goal": null}],
  "planner": {"resolution_m": 2.0, "num_simulations": 384,
               "replan_period_s": 0.8, "commit_depth": 6}
}}
{"type": "input", "agent_id": "human", "keys": ["up", "left"]}
{"type": "stop"}

// server -> client (10 Hz)
{"type": "tick", "time_s": 7.2,
 "agents": [{"id": "car1", "x": 31.2, "y": 24.4, "theta": 0.02,
              "speed": 5.9, "reached": false}],
 "routes": {"car1": [[34, 24.5], [36, 24.5]]},
 "stats": {"plan_ms": 100, "plans": 9, "collisions_car_car": 0,
            "collisions_obstacle": 0, "planner_errors": 0,
            "all_ai_reached": false, "planner_ready": true}}
```

## Design notes of the reference adapter (bridge.py)

- **Two layers of the same principle (RQ4):** the strategic layer is the
  discrete SM-MCTS on a 2 m grid (one XLA call per replan, ~100 ms for
  3 vehicles on CPU); the tactical layer is pure pursuit on Ackermann
  kinematics at 20 Hz. Discretization mismatches surfaced and were solved
  here: turn-in-place plan steps have no Ackermann equivalent (stationary
  waypoints are skipped; a steepest-descent potential fallback covers
  all-rotation plans), and obstacle footprints are inflated by the vehicle
  radius so planned routes keep physical clearance.
- **Humans are strategic agents (RQ5 placeholder):** a human-driven car
  enters the game with a goal *estimated* by projecting its heading
  6 cells ahead; only that agent's distance field is recomputed when the
  estimate changes. Upgrading this to Bayesian goal inference is a planned
  research branch and touches only the adapter.
- **Behavior profiles** map to heterogeneous payoff weights
  (`RewardParams` accepts per-agent tuples): cautious / normal /
  aggressive differ in progress vs. proximity weighting.

## Running

```bash
pip install -e ".[web]"
python -m sm_mcts_web          # -> http://localhost:8008
```
