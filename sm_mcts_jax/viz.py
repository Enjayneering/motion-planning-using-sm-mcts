"""Matplotlib visualization: static plots and animations of episodes."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from matplotlib.patches import Circle

from .environment import GridWorld
from .planner import Trajectory

_COLORS = plt.cm.tab10.colors


def _frame_at(env: GridWorld, t: int) -> np.ndarray:
    occ = np.asarray(env.occupancy, dtype=float)  # [T, H, W]
    idx = t // int(env.frame_duration)
    idx = idx % occ.shape[0] if bool(env.cycle) else min(idx, occ.shape[0] - 1)
    return occ[idx]


def _draw_map(ax, env: GridWorld, t: int = 0):
    occ = _frame_at(env, t)
    height, width = occ.shape
    image = ax.imshow(
        occ,
        cmap="gray_r",
        origin="lower",
        extent=(-0.5, width - 0.5, -0.5, height - 0.5),
        vmin=0.0,
        vmax=1.4,
    )
    ax.set_xlim(-0.5, width - 0.5)
    ax.set_ylim(-0.5, height - 0.5)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    return image


def plot_trajectory(env: GridWorld, traj: Trajectory, path: str | None = None):
    """Static overview: full paths of all agents."""
    fig, ax = plt.subplots(figsize=(7, 7))
    _draw_map(ax, env)
    goals = np.asarray(env.goals)
    for i in range(env.n_agents):
        color = _COLORS[i % len(_COLORS)]
        xy = traj.states[:, i, :2]
        ax.plot(xy[:, 0], xy[:, 1], "-o", color=color, markersize=3,
                linewidth=1.5, label=f"agent {i}")
        ax.plot(*xy[0], "s", color=color, markersize=10)
        ax.plot(*goals[i], "*", color=color, markersize=16,
                markeredgecolor="black")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_title(traj.summary(), fontsize=9)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=130)
        plt.close(fig)
    return fig


def animate_trajectory(env: GridWorld, traj: Trajectory, path: str,
                       fps: float = 4, agent_radius: float | None = None,
                       realtime: bool = False):
    """Render the episode as a GIF (or MP4 if ffmpeg is available).

    With ``realtime=True`` every frame is shown for the episode's *maximum
    measured planning time*, so watching the animation gives an honest live
    feeling for how fast the planner runs on the machine that produced the
    trajectory (worst-case step, no cherry-picking). The per-step planning
    time is displayed in the title.
    """
    max_plan_s = None
    if realtime and traj.plan_times.size:
        max_plan_s = float(traj.plan_times.max())
        fps = 1.0 / max(max_plan_s, 1e-3)
    fig, ax = plt.subplots(figsize=(7, 7))
    map_image = _draw_map(ax, env)
    goals = np.asarray(env.goals)
    radius = agent_radius or float(env.collision_radius) / 2.0

    bodies, headings, trails = [], [], []
    for i in range(env.n_agents):
        color = _COLORS[i % len(_COLORS)]
        ax.plot(*goals[i], "*", color=color, markersize=16,
                markeredgecolor="black", zorder=3)
        body = Circle(traj.states[0, i, :2], radius, color=color,
                      alpha=0.9, zorder=4)
        ax.add_patch(body)
        (heading,) = ax.plot([], [], "-", color="black", linewidth=1.5, zorder=5)
        (trail,) = ax.plot([], [], "--", color=color, linewidth=1.2,
                           alpha=0.7, zorder=2)
        bodies.append(body)
        headings.append(heading)
        trails.append(trail)
    title = ax.set_title("", fontsize=10)

    def update(t):
        map_image.set_data(_frame_at(env, t))
        for i in range(env.n_agents):
            x, y, th = traj.states[t, i]
            bodies[i].center = (x, y)
            headings[i].set_data(
                [x, x + radius * np.cos(th)], [y, y + radius * np.sin(th)]
            )
            trails[i].set_data(traj.states[: t + 1, i, 0],
                               traj.states[: t + 1, i, 1])
        label = f"t = {t} / {traj.states.shape[0] - 1}"
        if 1 <= t <= traj.plan_times.size:
            label += f"   ·   plan: {1e3 * traj.plan_times[t - 1]:.0f} ms"
        if max_plan_s is not None:
            label += (f"\nplayback = real time on CPU "
                      f"(worst step: {1e3 * max_plan_s:.0f} ms)")
        title.set_text(label)
        return bodies + headings + trails + [title]

    anim = animation.FuncAnimation(
        fig, update, frames=traj.states.shape[0], blit=False
    )
    writer = (
        animation.FFMpegWriter(fps=fps)
        if path.endswith(".mp4")
        else animation.PillowWriter(fps=fps)
    )
    anim.save(path, writer=writer)
    plt.close(fig)
    return path
