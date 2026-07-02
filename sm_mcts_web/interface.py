"""The planner interface of the simulation environment (version 1).

This is the *safe boundary* between the continuous simulation and whatever
planning algorithm is under research. The simulator promises to only ever
talk to a planner through this interface, and the planner sees nothing but
the data defined here — so the algorithm can be swapped without touching
the environment, and the environment can evolve without breaking planners.

Contract (see docs/SIM_INTERFACE.md for the full specification):

- ``reset(scenario)`` is called once per run with the static world.
- ``plan(snapshot)`` is called repeatedly with the live world state and
  must return a waypoint route per AI agent, in world meters. The
  simulator's low-level controller (pure pursuit + speed control) tracks
  these routes with the vehicle model — planners never command actuators.
- ``plan`` runs in a worker thread with a soft time budget
  (``scenario.planner.replan_period_s``); if it is late, vehicles simply
  keep following their previous routes. Exceptions are caught and logged;
  the simulation keeps running on the last routes (graceful degradation).
- Planners must treat ``snapshot`` as read-only and must not assume a
  call frequency.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

INTERFACE_VERSION = 1


@dataclass(frozen=True)
class ObstacleSpec:
    """A static world item. Footprints are axis-sized rectangles rotated by
    `rotation` around their center, or circles when `radius` is set."""

    kind: str                 # "tree" | "house" | "wall" | "parked_car" | "road" | ...
    x: float                  # center, meters
    y: float
    width: float = 0.0        # rectangle footprint, meters
    height: float = 0.0
    radius: float = 0.0       # circle footprint (overrides rectangle if > 0)
    rotation: float = 0.0     # radians
    blocking: bool = True     # False = decorative only (e.g. road markings)


@dataclass(frozen=True)
class AgentSpec:
    id: str
    kind: str                 # "ai" | "human"
    start: tuple              # (x, y, theta)
    goal: tuple | None        # (x, y); None for human-driven vehicles
    max_speed: float = 6.0    # m/s, tracked by the low-level controller
    behavior: str = "normal"  # "cautious" | "normal" | "aggressive"
    radius: float = 1.2       # collision footprint radius, meters


@dataclass(frozen=True)
class PlannerConfig:
    resolution_m: float = 2.0     # grid cell size of the discrete planner
    num_simulations: int = 384    # search budget per replan
    replan_period_s: float = 0.8  # soft budget & cadence for plan() calls
    commit_depth: int = 6         # waypoints returned per plan
    mode: str = "centralized"     # reserved: "decentralized", "async", ...
    safety_filter: bool = False   # reserved for the hard-shell dial (RQ3)


@dataclass(frozen=True)
class ScenarioSpec:
    width_m: float
    height_m: float
    obstacles: tuple = ()         # tuple[ObstacleSpec]
    agents: tuple = ()            # tuple[AgentSpec]
    planner: PlannerConfig = field(default_factory=PlannerConfig)
    version: int = INTERFACE_VERSION


@dataclass(frozen=True)
class AgentState:
    id: str
    x: float
    y: float
    theta: float
    speed: float
    reached: bool


@dataclass(frozen=True)
class WorldSnapshot:
    """Live, read-only view handed to the planner on every plan() call."""

    time_s: float
    agents: tuple                 # tuple[AgentState], same order as scenario


@dataclass(frozen=True)
class Route:
    """A planner's output for one agent: waypoints in world meters. The
    low-level controller tracks them in order; `speed` caps tracking speed
    (defaults to the agent's max_speed when 0)."""

    waypoints: tuple              # tuple[(x, y)], ordered, may be empty
    speed: float = 0.0


class PlannerAdapter(ABC):
    """Implement this to plug any planning algorithm into the simulator."""

    #: bump when your adapter needs a newer scenario schema
    interface_version: int = INTERFACE_VERSION

    @abstractmethod
    def reset(self, scenario: ScenarioSpec) -> None:
        """Called once before the run with the static world."""

    @abstractmethod
    def plan(self, snapshot: WorldSnapshot) -> dict:
        """Return {agent_id: Route} for every AI agent (humans excluded).
        Missing entries mean 'keep the previous route'."""

    def debug_info(self) -> dict:
        """Optional, JSON-serializable introspection shown in the UI
        (e.g. inferred goals, beliefs). Never required for correctness."""
        return {}


def scenario_from_json(data: dict) -> ScenarioSpec:
    """Parse and validate the wire format (see docs/SIM_INTERFACE.md)."""
    if int(data.get("version", INTERFACE_VERSION)) != INTERFACE_VERSION:
        raise ValueError(
            f"scenario version {data.get('version')} != {INTERFACE_VERSION}"
        )
    obstacles = tuple(
        ObstacleSpec(
            kind=str(o["kind"]),
            x=float(o["x"]), y=float(o["y"]),
            width=float(o.get("width", 0.0)),
            height=float(o.get("height", 0.0)),
            radius=float(o.get("radius", 0.0)),
            rotation=float(o.get("rotation", 0.0)),
            blocking=bool(o.get("blocking", True)),
        )
        for o in data.get("obstacles", [])
    )
    agents = []
    seen_ids = set()
    for a in data.get("agents", []):
        if a["id"] in seen_ids:
            raise ValueError(f"duplicate agent id {a['id']!r}")
        seen_ids.add(a["id"])
        kind = str(a.get("kind", "ai"))
        if kind not in ("ai", "human"):
            raise ValueError(f"unknown agent kind {kind!r}")
        goal = a.get("goal")
        if kind == "ai" and goal is None:
            raise ValueError(f"AI agent {a['id']!r} needs a goal")
        agents.append(AgentSpec(
            id=str(a["id"]),
            kind=kind,
            start=(float(a["start"][0]), float(a["start"][1]),
                   float(a["start"][2])),
            goal=None if goal is None else (float(goal[0]), float(goal[1])),
            max_speed=float(a.get("max_speed", 6.0)),
            behavior=str(a.get("behavior", "normal")),
            radius=float(a.get("radius", 1.2)),
        ))
    p = data.get("planner", {})
    planner = PlannerConfig(
        resolution_m=float(p.get("resolution_m", 2.0)),
        num_simulations=int(p.get("num_simulations", 384)),
        replan_period_s=float(p.get("replan_period_s", 0.8)),
        commit_depth=int(p.get("commit_depth", 6)),
        mode=str(p.get("mode", "centralized")),
        safety_filter=bool(p.get("safety_filter", False)),
    )
    return ScenarioSpec(
        width_m=float(data["width_m"]),
        height_m=float(data["height_m"]),
        obstacles=obstacles,
        agents=tuple(agents),
        planner=planner,
    )
