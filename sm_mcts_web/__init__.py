"""Interactive web simulation environment for sm-mcts-jax.

Continuous world + Ackermann vehicles in the browser; the discrete
real-time SM-MCTS planner runs behind the versioned PlannerAdapter
interface (see docs/SIM_INTERFACE.md) so the algorithm stays swappable.
"""

from .interface import (
    INTERFACE_VERSION,
    AgentSpec,
    AgentState,
    ObstacleSpec,
    PlannerAdapter,
    PlannerConfig,
    Route,
    ScenarioSpec,
    WorldSnapshot,
    scenario_from_json,
)

__all__ = [
    "INTERFACE_VERSION",
    "AgentSpec",
    "AgentState",
    "ObstacleSpec",
    "PlannerAdapter",
    "PlannerConfig",
    "Route",
    "ScenarioSpec",
    "WorldSnapshot",
    "scenario_from_json",
]
