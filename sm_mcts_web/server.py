"""FastAPI server: serves the research site and runs sim sessions over
WebSocket.

    python -m sm_mcts_web            # http://localhost:8008

Protocol (docs/SIM_INTERFACE.md):
  client -> {"type": "start", "scenario": {...}}
  client -> {"type": "input", "agent_id": "...", "keys": ["up", "left"]}
  client -> {"type": "stop"}
  server -> {"type": "status", "message": "..."}
  server -> {"type": "tick", ...}        (SimSession.state_json)
  server -> {"type": "error", "message": "..."}
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .bridge import SMMCTSAdapter
from .interface import scenario_from_json
from .session import SimSession

logger = logging.getLogger("sm_mcts_web")
STATIC = Path(__file__).parent / "static"

PHYSICS_DT = 0.05      # 20 Hz integration
BROADCAST_EVERY = 2    # -> 10 Hz to the browser

app = FastAPI(title="sm-mcts-jax interactive sim")
app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.get("/")
async def index():
    return FileResponse(STATIC / "index.html")


async def _run_session(ws: WebSocket, session: SimSession) -> None:
    await ws.send_json({"type": "status",
                        "message": "Planner kompiliert (JIT) …"})
    await asyncio.to_thread(session.reset_planner)
    await ws.send_json({"type": "status", "message": "Simulation läuft"})
    tick = 0
    # real-time pacing: one physics step per PHYSICS_DT of wall time
    loop = asyncio.get_event_loop()
    next_time = loop.time()
    while True:
        session.tick(PHYSICS_DT)
        tick += 1
        if tick % BROADCAST_EVERY == 0:
            await ws.send_json(session.state_json())
        next_time += PHYSICS_DT
        delay = next_time - loop.time()
        if delay > 0:
            await asyncio.sleep(delay)
        else:  # running behind (e.g. slow machine): don't spiral
            next_time = loop.time()
            await asyncio.sleep(0)


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    session: SimSession | None = None
    runner: asyncio.Task | None = None

    async def stop_runner():
        nonlocal runner, session
        if runner is not None:
            runner.cancel()
            try:
                await runner
            except (asyncio.CancelledError, Exception):
                pass
            runner = None
        if session is not None:
            session.close()
            session = None

    try:
        while True:
            message = await ws.receive_json()
            msg_type = message.get("type")
            if msg_type == "start":
                await stop_runner()
                try:
                    scenario = scenario_from_json(message["scenario"])
                    session = SimSession(scenario, SMMCTSAdapter())
                except Exception as exc:
                    await ws.send_json({"type": "error", "message": str(exc)})
                    continue
                runner = asyncio.create_task(_run_session(ws, session))
            elif msg_type == "input" and session is not None:
                agent_id = message.get("agent_id")
                if agent_id in session.human_keys:
                    session.human_keys[agent_id] = set(message.get("keys", []))
            elif msg_type == "stop":
                await stop_runner()
                await ws.send_json({"type": "status", "message": "Gestoppt"})
    except WebSocketDisconnect:
        pass
    finally:
        await stop_runner()


def main(host: str = "127.0.0.1", port: int = 8008) -> None:
    import uvicorn
    logging.basicConfig(level=logging.INFO)
    uvicorn.run(app, host=host, port=port, log_level="warning")


if __name__ == "__main__":
    main()
