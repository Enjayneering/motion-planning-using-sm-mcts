"""Headless end-to-end tests of the web simulation stack (no browser):
scenario parsing -> rasterization -> SM-MCTS adapter -> Ackermann tracking.
"""

import math

import numpy as np
import pytest

from sm_mcts_web.bridge import SMMCTSAdapter, nearest_free_cell, rasterize
from sm_mcts_web.interface import scenario_from_json
from sm_mcts_web.session import SimSession

SCENARIO = {
    "version": 1,
    "width_m": 40.0,
    "height_m": 24.0,
    "obstacles": [
        {"kind": "house", "x": 20.0, "y": 12.0, "width": 6.0, "height": 5.0},
        {"kind": "tree", "x": 8.0, "y": 18.0, "radius": 1.0},
        {"kind": "road", "x": 20.0, "y": 4.0, "width": 30.0, "height": 6.0,
         "blocking": False},
    ],
    "agents": [
        {"id": "car1", "kind": "ai", "start": [4.0, 4.0, 0.0],
         "goal": [36.0, 20.0], "behavior": "normal", "max_speed": 6.0},
        {"id": "car2", "kind": "ai", "start": [36.0, 4.0, 3.1416],
         "goal": [4.0, 20.0], "behavior": "cautious", "max_speed": 5.0},
        {"id": "human", "kind": "human", "start": [4.0, 20.0, 0.0],
         "goal": None},
    ],
    "planner": {"resolution_m": 2.0, "num_simulations": 128,
                "replan_period_s": 0.6, "commit_depth": 5},
}


def test_scenario_parsing_and_validation():
    scenario = scenario_from_json(SCENARIO)
    assert scenario.width_m == 40.0
    assert len(scenario.agents) == 3
    with pytest.raises(ValueError):
        scenario_from_json({**SCENARIO, "version": 99})
    bad = {**SCENARIO, "agents": [
        {"id": "x", "kind": "ai", "start": [1, 1, 0], "goal": None}]}
    with pytest.raises(ValueError):
        scenario_from_json(bad)


def test_rasterize_blocks_house_not_road():
    scenario = scenario_from_json(SCENARIO)
    occ = rasterize(scenario, 2.0)
    assert occ.shape == (12, 20)
    assert occ[6, 10]            # house center blocked
    assert not occ[2, 10]        # road is decorative
    col, row = nearest_free_cell(occ, 10, 6)
    assert not occ[row, col]


def test_full_episode_ai_reaches_goal_and_human_drives():
    scenario = scenario_from_json(SCENARIO)
    session = SimSession(scenario, SMMCTSAdapter(seed=0))
    session.reset_planner()
    session.human_keys["human"] = {"up"}          # human drives straight

    dt = 0.05
    reached = False
    for _ in range(int(60.0 / dt)):               # up to 60 sim-seconds
        session.tick(dt)
        # force pending plans to apply promptly in the headless loop
        if session._plan_future is not None:
            session._plan_future.result(timeout=30)
        if session.stats.all_ai_reached:
            reached = True
            break
    state = session.state_json()
    session.close()

    assert reached, f"AI cars did not reach goals: {state}"
    human = next(a for a in state["agents"] if a["id"] == "human")
    assert human["x"] > 6.0, "human car should have moved forward"
    assert state["stats"]["planner_errors"] == 0
    assert state["stats"]["plans"] > 3


def test_planner_exception_degrades_gracefully():
    scenario = scenario_from_json(SCENARIO)

    class ExplodingAdapter(SMMCTSAdapter):
        def plan(self, snapshot):
            raise RuntimeError("boom")

    session = SimSession(scenario, ExplodingAdapter(seed=0))
    session.reset_planner()
    for _ in range(80):
        session.tick(0.05)
        if session._plan_future is not None:
            try:
                session._plan_future.result(timeout=30)
            except RuntimeError:
                pass
    stats = session.state_json()["stats"]
    session.close()
    assert stats["planner_errors"] >= 1           # failure was recorded
    # ... and the simulation kept running regardless
    assert math.isclose(session.stats.time_s, 80 * 0.05, rel_tol=1e-6)
