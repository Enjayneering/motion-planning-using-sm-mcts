/* SM-MCTS Live — Karteneditor + Simulations-Client.
   Welt: 64 x 40 m, y zeigt nach oben; Canvas 15 px/m. */
"use strict";

const WORLD_W = 64, WORLD_H = 40, S = 15;
const canvas = document.getElementById("world");
const ctx = canvas.getContext("2d");
const hud = document.getElementById("hud");
const statusEl = document.getElementById("status");

const CATALOG = {
  road:       { w: 6.0, h: 14.0, blocking: false },
  tree:       { r: 1.1,          blocking: true },
  house:      { w: 7.0, h: 5.5,  blocking: true },
  wall:       { w: 5.0, h: 0.7,  blocking: true },
  parked_car: { w: 4.4, h: 1.9,  blocking: true },
};
const AGENT_COLORS = ["#2b6cb0", "#c05621", "#2f855a", "#6b46c1",
                      "#b83280", "#986801"];
const HUMAN_COLOR = "#d33030";

const state = {
  mode: "build",            // build | agents | human
  item: "road",
  rotation: 0,              // radians, ghost rotation
  obstacles: [],            // {kind,x,y,width,height,radius,rotation,blocking}
  agents: [],               // {id,start:[x,y,th],goal:[x,y],behavior,max_speed}
  human: null,              // {start:[x,y,th]}
  placingAgent: null,       // {stage:"start"|"goal", start?}
  placingHuman: null,       // {stage:"pos"|"dir", pos?}
  mouse: null,              // world coords
  running: false,
  ws: null,
  tick: null,               // last tick message
  keys: new Set(),
  agentSeq: 0,
};

/* ---------- coordinates ---------- */
const toCanvas = (x, y) => [x * S, (WORLD_H - y) * S];
function toWorld(evt) {
  const r = canvas.getBoundingClientRect();
  const x = (evt.clientX - r.left) / r.width * WORLD_W;
  const y = WORLD_H - (evt.clientY - r.top) / r.height * WORLD_H;
  return [Math.min(Math.max(x, 0), WORLD_W), Math.min(Math.max(y, 0), WORLD_H)];
}

/* ---------- drawing primitives ---------- */
function drawRotated(x, y, theta, fn) {
  const [px, py] = toCanvas(x, y);
  ctx.save(); ctx.translate(px, py); ctx.rotate(-theta); fn(); ctx.restore();
}
function roundRect(x, y, w, h, r) {
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.arcTo(x + w, y, x + w, y + h, r);
  ctx.arcTo(x + w, y + h, x, y + h, r);
  ctx.arcTo(x, y + h, x, y, r);
  ctx.arcTo(x, y, x + w, y, r);
  ctx.closePath();
}

function drawObstacle(o, ghost) {
  ctx.globalAlpha = ghost ? 0.55 : 1.0;
  if (o.kind === "road") {
    drawRotated(o.x, o.y, o.rotation, () => {
      const w = o.width * S, h = o.height * S;
      ctx.fillStyle = "#6d6d6d";
      ctx.fillRect(-w / 2, -h / 2, w, h);
      ctx.strokeStyle = "#f4f0dc"; ctx.lineWidth = 2;
      ctx.setLineDash([14, 12]);
      ctx.beginPath(); ctx.moveTo(0, -h / 2 + 4); ctx.lineTo(0, h / 2 - 4);
      ctx.stroke(); ctx.setLineDash([]);
    });
  } else if (o.kind === "tree") {
    drawRotated(o.x, o.y, 0, () => {
      const r = o.radius * S;
      ctx.fillStyle = "#7a5230";
      ctx.beginPath(); ctx.arc(0, 0, r * 0.25, 0, 7); ctx.fill();
      ctx.fillStyle = "#3e7d3a";
      ctx.beginPath(); ctx.arc(-r * .25, -r * .2, r * .75, 0, 7); ctx.fill();
      ctx.fillStyle = "#4f9448";
      ctx.beginPath(); ctx.arc(r * .22, r * .18, r * .65, 0, 7); ctx.fill();
      ctx.fillStyle = "rgba(255,255,255,.14)";
      ctx.beginPath(); ctx.arc(-r * .35, -r * .35, r * .3, 0, 7); ctx.fill();
    });
  } else if (o.kind === "house") {
    drawRotated(o.x, o.y, o.rotation, () => {
      const w = o.width * S, h = o.height * S;
      ctx.fillStyle = "#b4513e";
      roundRect(-w / 2, -h / 2, w, h, 4); ctx.fill();
      ctx.fillStyle = "#93372b";
      ctx.fillRect(-w / 2 + 3, -h / 2 + 3, w - 6, h / 2 - 4);
      ctx.strokeStyle = "#7c2d22"; ctx.lineWidth = 2;
      ctx.beginPath(); ctx.moveTo(-w / 2, 0); ctx.lineTo(w / 2, 0); ctx.stroke();
    });
  } else if (o.kind === "wall") {
    drawRotated(o.x, o.y, o.rotation, () => {
      const w = o.width * S, h = o.height * S;
      ctx.fillStyle = "#9a938a"; ctx.fillRect(-w / 2, -h / 2, w, h);
      ctx.strokeStyle = "#7b756d"; ctx.lineWidth = 1;
      for (let i = 1; i < 4; i++) {
        const x = -w / 2 + (w * i) / 4;
        ctx.beginPath(); ctx.moveTo(x, -h / 2); ctx.lineTo(x, h / 2); ctx.stroke();
      }
    });
  } else if (o.kind === "parked_car") {
    drawRotated(o.x, o.y, o.rotation, () =>
      drawCarBody(o.width * S, o.height * S, "#8899a6", "#5c6b76"));
  }
  ctx.globalAlpha = 1.0;
}

function drawCarBody(len, wid, color, dark) {
  ctx.fillStyle = "rgba(0,0,0,.18)";
  roundRect(-len / 2 + 2, -wid / 2 + 2, len, wid, 5); ctx.fill();
  ctx.fillStyle = color;
  roundRect(-len / 2, -wid / 2, len, wid, 5); ctx.fill();
  ctx.fillStyle = dark;   // windshield + rear window
  roundRect(len * 0.08, -wid / 2 + 3, len * 0.22, wid - 6, 3); ctx.fill();
  roundRect(-len * 0.36, -wid / 2 + 3, len * 0.16, wid - 6, 3); ctx.fill();
  ctx.fillStyle = "#222"; // wheels
  const wy = wid / 2 - 1.5;
  [[-len * 0.3, -wy], [-len * 0.3, wy], [len * 0.3, -wy], [len * 0.3, wy]]
    .forEach(([wx, y]) => ctx.fillRect(wx - 3.5, y - 2, 7, 4));
}

function drawCar(x, y, theta, color, label, speed) {
  drawRotated(x, y, theta, () => {
    drawCarBody(4.4 * S, 1.9 * S, color, "rgba(20,25,30,.75)");
    ctx.fillStyle = "#fff"; ctx.font = "bold 11px system-ui";
    ctx.textAlign = "center"; ctx.textBaseline = "middle";
    ctx.save(); ctx.rotate(Math.PI / 2 * 0); ctx.fillText(label, -8, 0); ctx.restore();
  });
  if (speed !== undefined) {
    const [px, py] = toCanvas(x, y);
    ctx.fillStyle = "rgba(0,0,0,.55)"; ctx.font = "10px ui-monospace";
    ctx.textAlign = "center";
    ctx.fillText(speed.toFixed(1) + " m/s", px, py - 22);
  }
}

function drawFlag(x, y, color) {
  const [px, py] = toCanvas(x, y);
  ctx.strokeStyle = "#333"; ctx.lineWidth = 2;
  ctx.beginPath(); ctx.moveTo(px, py); ctx.lineTo(px, py - 18); ctx.stroke();
  ctx.fillStyle = color;
  ctx.beginPath(); ctx.moveTo(px, py - 18); ctx.lineTo(px + 13, py - 13);
  ctx.lineTo(px, py - 8); ctx.closePath(); ctx.fill();
}

/* ---------- main render ---------- */
function render() {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  // lawn texture
  ctx.fillStyle = "#a9c38a"; ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.fillStyle = "rgba(255,255,255,.05)";
  for (let i = 0; i < WORLD_W; i += 4)
    for (let j = (i / 4) % 2; j < WORLD_H; j += 4)
      ctx.fillRect(i * S, j * S, 4 * S, 4 * S);
  // planner grid (2 m), subtle
  ctx.strokeStyle = "rgba(255,255,255,.10)"; ctx.lineWidth = 1;
  for (let x = 0; x <= WORLD_W; x += 2) {
    ctx.beginPath(); ctx.moveTo(x * S, 0); ctx.lineTo(x * S, WORLD_H * S); ctx.stroke();
  }
  for (let y = 0; y <= WORLD_H; y += 2) {
    ctx.beginPath(); ctx.moveTo(0, y * S); ctx.lineTo(WORLD_W * S, y * S); ctx.stroke();
  }

  const roads = state.obstacles.filter(o => o.kind === "road");
  const solids = state.obstacles.filter(o => o.kind !== "road");
  roads.forEach(o => drawObstacle(o, false));
  solids.forEach(o => drawObstacle(o, false));

  // routes of the live planner
  if (state.running && state.tick) {
    Object.entries(state.tick.routes || {}).forEach(([id, wps], i) => {
      if (!wps.length) return;
      const agent = state.agents.find(a => a.id === id);
      const color = agent ? agent.color : "#fff";
      const car = state.tick.agents.find(a => a.id === id);
      ctx.strokeStyle = color; ctx.lineWidth = 2; ctx.setLineDash([6, 7]);
      ctx.beginPath();
      const [sx, sy] = toCanvas(car.x, car.y); ctx.moveTo(sx, sy);
      wps.forEach(([x, y]) => { const [px, py] = toCanvas(x, y); ctx.lineTo(px, py); });
      ctx.stroke(); ctx.setLineDash([]);
    });
  }

  // goals
  state.agents.forEach(a => drawFlag(a.goal[0], a.goal[1], a.color));

  // cars
  if (state.running && state.tick) {
    state.tick.agents.forEach(a => {
      const spec = state.agents.find(s => s.id === a.id);
      const color = spec ? spec.color : HUMAN_COLOR;
      const label = spec ? spec.label : "DU";
      drawCar(a.x, a.y, a.theta, color, label, a.speed);
      if (a.reached && spec) {
        const [px, py] = toCanvas(a.x, a.y);
        ctx.fillStyle = "#fff"; ctx.font = "14px system-ui";
        ctx.fillText("✅", px + 18, py - 14);
      }
    });
  } else {
    state.agents.forEach(a =>
      drawCar(a.start[0], a.start[1], a.start[2], a.color, a.label));
    if (state.human)
      drawCar(state.human.start[0], state.human.start[1],
              state.human.start[2], HUMAN_COLOR, "DU");
  }

  drawGhost();
}

function drawGhost() {
  if (state.running || !state.mouse) return;
  const [mx, my] = state.mouse;
  if (state.mode === "build") {
    const spec = CATALOG[state.item];
    drawObstacle({
      kind: state.item, x: mx, y: my, rotation: state.rotation,
      width: spec.w || 0, height: spec.h || 0, radius: spec.r || 0,
    }, true);
  } else if (state.mode === "agents" && state.placingAgent) {
    ctx.globalAlpha = 0.6;
    if (state.placingAgent.stage === "start") {
      drawCar(mx, my, 0, AGENT_COLORS[state.agentSeq % AGENT_COLORS.length], "?");
    } else {
      const s = state.placingAgent.start;
      drawCar(s[0], s[1], Math.atan2(my - s[1], mx - s[0]),
              AGENT_COLORS[state.agentSeq % AGENT_COLORS.length], "?");
      drawFlag(mx, my, AGENT_COLORS[state.agentSeq % AGENT_COLORS.length]);
    }
    ctx.globalAlpha = 1;
  } else if (state.mode === "human" && state.placingHuman) {
    ctx.globalAlpha = 0.6;
    if (state.placingHuman.stage === "pos") drawCar(mx, my, 0, HUMAN_COLOR, "DU");
    else {
      const p = state.placingHuman.pos;
      drawCar(p[0], p[1], Math.atan2(my - p[1], mx - p[0]), HUMAN_COLOR, "DU");
    }
    ctx.globalAlpha = 1;
  }
}

/* ---------- editor interactions ---------- */
canvas.addEventListener("mousemove", e => { state.mouse = toWorld(e); render(); });
canvas.addEventListener("mouseleave", () => { state.mouse = null; render(); });

canvas.addEventListener("contextmenu", e => {
  e.preventDefault();
  if (state.running || state.mode !== "build") return;
  const [mx, my] = toWorld(e);
  for (let i = state.obstacles.length - 1; i >= 0; i--) {
    const o = state.obstacles[i];
    const extent = o.radius || Math.hypot(o.width, o.height) / 2;
    if (Math.hypot(o.x - mx, o.y - my) <= extent + 0.5) {
      state.obstacles.splice(i, 1); render(); return;
    }
  }
});

canvas.addEventListener("click", e => {
  if (state.running) return;
  const [mx, my] = toWorld(e);
  if (state.mode === "build") {
    const spec = CATALOG[state.item];
    state.obstacles.push({
      kind: state.item, x: mx, y: my, rotation: state.rotation,
      width: spec.w || 0, height: spec.h || 0, radius: spec.r || 0,
      blocking: spec.blocking,
    });
  } else if (state.mode === "agents" && state.placingAgent) {
    if (state.placingAgent.stage === "start") {
      state.placingAgent = { stage: "goal", start: [mx, my] };
      setStatus("Jetzt das Ziel dieses Autos klicken.");
    } else {
      const s = state.placingAgent.start;
      const theta = Math.atan2(my - s[1], mx - s[0]);
      const id = "car" + (++state.agentSeq);
      state.agents.push({
        id, label: "A" + state.agentSeq,
        color: AGENT_COLORS[(state.agentSeq - 1) % AGENT_COLORS.length],
        start: [s[0], s[1], theta], goal: [mx, my],
        behavior: document.getElementById("sel-behavior").value,
        max_speed: parseFloat(document.getElementById("rng-speed").value),
      });
      state.placingAgent = null;
      refreshChips();
      setStatus(`KI-Auto ${id} gesetzt. Weitere hinzufügen oder Start drücken.`);
    }
  } else if (state.mode === "human" && state.placingHuman) {
    if (state.placingHuman.stage === "pos") {
      state.placingHuman = { stage: "dir", pos: [mx, my] };
      setStatus("Blickrichtung klicken.");
    } else {
      const p = state.placingHuman.pos;
      state.human = { start: [p[0], p[1], Math.atan2(my - p[1], mx - p[0])] };
      state.placingHuman = null;
      setStatus("Dein Auto steht. Start drücken und mit WASD/Pfeilen fahren.");
    }
  }
  render();
});

document.addEventListener("keydown", e => {
  if (!state.running && (e.key === "r" || e.key === "R")) {
    state.rotation = (state.rotation + Math.PI / 4) % (Math.PI * 2);
    render();
  }
  if (state.running) {
    const k = keyName(e); if (k) { e.preventDefault(); state.keys.add(k); sendKeys(); }
  }
});
document.addEventListener("keyup", e => {
  const k = keyName(e); if (k) { state.keys.delete(k); sendKeys(); }
});
const keyName = e => ({
  ArrowUp: "up", ArrowDown: "down", ArrowLeft: "left", ArrowRight: "right",
  w: "up", s: "down", a: "left", d: "right",
  W: "up", S: "down", A: "left", D: "right",
})[e.key];

/* ---------- toolbar ---------- */
document.querySelectorAll(".modes button[data-mode]").forEach(btn => {
  btn.addEventListener("click", () => {
    state.mode = btn.dataset.mode;
    document.querySelectorAll(".modes button[data-mode]")
      .forEach(b => b.classList.toggle("active", b === btn));
    document.getElementById("panel-build").classList.toggle("hidden", state.mode !== "build");
    document.getElementById("panel-agents").classList.toggle("hidden", state.mode !== "agents");
    document.getElementById("panel-human").classList.toggle("hidden", state.mode !== "human");
    if (state.mode === "human" && !state.human) state.placingHuman = { stage: "pos" };
    render();
  });
});
document.querySelectorAll("#panel-build .item").forEach(btn => {
  btn.addEventListener("click", () => {
    state.item = btn.dataset.item;
    document.querySelectorAll("#panel-build .item")
      .forEach(b => b.classList.toggle("active", b === btn));
  });
});
document.getElementById("btn-add-agent").addEventListener("click", () => {
  state.placingAgent = { stage: "start" };
  setStatus("Startposition des KI-Autos klicken.");
});
document.getElementById("btn-remove-human").addEventListener("click", () => {
  state.human = null; state.placingHuman = { stage: "pos" }; render();
});
document.getElementById("rng-speed").addEventListener("input", e => {
  document.getElementById("lbl-speed").textContent = e.target.value + " m/s";
});
document.getElementById("btn-clear").addEventListener("click", () => {
  stopSim();
  Object.assign(state, { obstacles: [], agents: [], human: null, agentSeq: 0 });
  refreshChips(); render(); setStatus("Karte geleert.");
});
document.getElementById("btn-demo").addEventListener("click", () => {
  stopSim(); loadDemo(); render();
  setStatus("Demo geladen: zwei KI-Autos kreuzen sich; platziere dich dazu und starte.");
});

function refreshChips() {
  const el = document.getElementById("agent-chips");
  el.innerHTML = "";
  state.agents.forEach((a, i) => {
    const chip = document.createElement("span");
    chip.className = "chip";
    chip.innerHTML = `<b style="background:${a.color}"></b>${a.label}
      · ${a.behavior} · ${a.max_speed} m/s <button title="löschen">✕</button>`;
    chip.querySelector("button").addEventListener("click", () => {
      state.agents.splice(i, 1); refreshChips(); render();
    });
    el.appendChild(chip);
  });
}

/* ---------- scenario / websocket ---------- */
function buildScenario() {
  const agents = state.agents.map(a => ({
    id: a.id, kind: "ai", start: a.start, goal: a.goal,
    behavior: a.behavior, max_speed: a.max_speed, radius: 0.95,
  }));
  if (state.human)
    agents.push({ id: "human", kind: "human", start: state.human.start,
                  goal: null, max_speed: 9.0, radius: 0.95 });
  return {
    version: 1, width_m: WORLD_W, height_m: WORLD_H,
    obstacles: state.obstacles,
    agents,
    planner: { resolution_m: 2.0, num_simulations: 384,
               replan_period_s: 0.8, commit_depth: 6 },
  };
}

function startSim() {
  if (!state.agents.length) { setStatus("⚠️ Mindestens ein KI-Auto setzen."); return; }
  const proto = location.protocol === "https:" ? "wss" : "ws";
  state.ws = new WebSocket(`${proto}://${location.host}/ws`);
  state.ws.onopen = () => {
    state.ws.send(JSON.stringify({ type: "start", scenario: buildScenario() }));
    state.running = true;
    document.getElementById("btn-run").textContent = "■ Stop";
    document.getElementById("btn-run").classList.add("running");
  };
  state.ws.onmessage = evt => {
    const msg = JSON.parse(evt.data);
    if (msg.type === "tick") { state.tick = msg; updateHud(msg); render(); }
    else if (msg.type === "status") setStatus(msg.message);
    else if (msg.type === "error") { setStatus("❌ " + msg.message); stopSim(); }
  };
  state.ws.onclose = () => { if (state.running) stopSim(); };
}

function stopSim() {
  if (state.ws) { try { state.ws.send(JSON.stringify({ type: "stop" })); } catch (e) {}
    state.ws.close(); state.ws = null; }
  state.running = false; state.tick = null; state.keys.clear();
  document.getElementById("btn-run").textContent = "▶ Start";
  document.getElementById("btn-run").classList.remove("running");
  hud.textContent = ""; render();
}

document.getElementById("btn-run").addEventListener("click",
  () => state.running ? stopSim() : startSim());

function sendKeys() {
  if (state.ws && state.running)
    state.ws.send(JSON.stringify({ type: "input", agent_id: "human",
                                   keys: [...state.keys] }));
}

function updateHud(msg) {
  const s = msg.stats;
  hud.textContent =
    `t          ${msg.time_s.toFixed(1)} s\n` +
    `Replan     ${s.plan_ms} ms  (#${s.plans})\n` +
    `Kollision  Auto ${s.collisions_car_car} · Objekt ${s.collisions_obstacle}\n` +
    (s.planner_errors ? `Planner-Fehler ${s.planner_errors}\n` : "") +
    (s.all_ai_reached ? "✅ alle KI-Ziele erreicht" : "");
}

const setStatus = t => { statusEl.textContent = t; };

/* ---------- demo world ---------- */
function loadDemo() {
  state.obstacles = []; state.agents = []; state.agentSeq = 0; state.human = null;
  const add = (kind, x, y, rot = 0) => {
    const s = CATALOG[kind];
    state.obstacles.push({ kind, x, y, rotation: rot, width: s.w || 0,
                           height: s.h || 0, radius: s.r || 0, blocking: s.blocking });
  };
  // crossing roads
  for (let y = 7; y <= 33; y += 13) add("road", 32, y, 0);
  for (let x = 11; x <= 53; x += 14) add("road", x, 20, Math.PI / 2);
  // houses in the quadrants
  add("house", 12, 32); add("house", 52, 32, Math.PI / 12);
  add("house", 12, 8); add("house", 52, 8);
  // trees + parked cars
  [[22, 34], [42, 34], [22, 6], [42, 6], [6, 20], [58, 20]]
    .forEach(([x, y]) => add("tree", x, y));
  add("parked_car", 24, 24.6, 0); add("parked_car", 40, 15.4, 0);
  // two AI cars crossing the intersection
  state.agents.push({
    id: "car1", label: "A1", color: AGENT_COLORS[0],
    start: [6, 24.5, 0], goal: [58, 24.5],
    behavior: "normal", max_speed: 6,
  });
  state.agents.push({
    id: "car2", label: "A2", color: AGENT_COLORS[1],
    start: [34.5, 36, -Math.PI / 2], goal: [34.5, 4],
    behavior: "normal", max_speed: 6,
  });
  state.agentSeq = 2;
  refreshChips();
}

loadDemo();
render();
setStatus("Demo geladen — bauen, Autos setzen, dich platzieren (🧑), dann ▶ Start.");
