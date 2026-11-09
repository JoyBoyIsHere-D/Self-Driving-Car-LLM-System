"""Runs one episode: the world clock, the agents and the radio.

The runner is NOT a central planner. It only (1) advances physics, (2) hands
each agent its local observation and inbox, (3) lets agents decide in
parallel (one worker thread per car), and (4) applies the actions they chose.
"""
from __future__ import annotations

import math
import os
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from .agents.vehicle_agent import VehicleAgent
from .analytics.metrics import episode_metrics
from .analytics.telemetry import Telemetry
from .comms.bus import MessageBus
from .config import RunConfig
from .llm.client import make_llm
from .memory.lessons import LessonStore
from .sim.scenarios import make_world


def run_episode(cfg: RunConfig, lessons_path: str | None = None, verbose: bool = False,
                frames: list | None = None) -> dict:
    """Run one episode, write per-node logs, return the metrics dict.
    If `frames` is a list, a snapshot of every vehicle is appended each step (for animation)."""
    sim, ag = cfg.sim, cfg.agent
    run_dir = os.path.join(cfg.out_dir, cfg.run_name)
    os.makedirs(run_dir, exist_ok=True)
    cfg.save(os.path.join(run_dir, "config.json"))
    tel = Telemetry(run_dir)

    world = make_world(sim.scenario, sim.n_agents, sim.n_humans, sim.dt, sim.seed)
    bus = MessageBus(cfg.comms.range_m, cfg.comms.latency_steps, cfg.comms.loss_prob,
                     np.random.default_rng(sim.seed + 1000), enabled=cfg.comms.enabled)
    llm = None if ag.system == "rule" else make_llm(cfg.llm.backend, cfg.llm.model, cfg.llm.host,
                                                    cfg.llm.temperature, cfg.llm.timeout_s,
                                                    cfg.llm.mock_error_rate, sim.seed)
    memory = LessonStore(lessons_path) if (ag.memory and lessons_path and ag.system != "rule") else None
    agents = {vid: VehicleAgent(vid, ag.system, llm, ag.shield, memory, ag.sensor_range_m, sim.scenario, v.v_des)
              for vid, v in world.vehicles.items() if v.kind == "agent"}
    tel.emit("world", "meta", scenario=sim.scenario, system=ag.system, n_agents=sim.n_agents,
             backend=cfg.llm.backend, model=cfg.llm.model, seed=sim.seed)

    n_steps = int(sim.duration_s / sim.dt)
    wall0 = time.perf_counter()
    pool = ThreadPoolExecutor(max_workers=max(1, len(agents)))
    try:
        for step in range(n_steps):
            live = [vid for vid in agents if world.vehicles[vid].active]
            if not live:
                break
            positions = {vid: world.vehicles[vid].x for vid in live}
            bus.deliver(step, positions)

            if step % sim.decision_every == 0:
                _decision_tick(step, world, agents, live, positions, bus, pool, tel, verbose)

            world.step()
            for ev in world.events:
                tel.emit("world", ev.pop("event"), step=step, **ev)
            for vid in live:
                v = world.vehicles[vid]
                gap, v_lead = world.leader_gap(v)
                ttc = gap / (v.v - v_lead) if (v.active and v.v > v_lead and not math.isinf(gap)) else None
                tel.emit(vid, "state", step=step, t=round(world.t, 2), x=round(v.x, 2), lane=v.lane,
                         v=round(v.v, 2), a=round(v.a, 2), ttc=None if ttc is None else round(ttc, 2),
                         crashed=v.crashed, finished=v.finished)
            if frames is not None:
                frames.append([(o.vid, o.kind, o.x, o.lane, o.v, o.crashed) for o in world.vehicles.values()
                               if o.active or o.crashed])
    finally:
        pool.shutdown(wait=True)

    wall = time.perf_counter() - wall0
    tel.emit("bus", "summary", **bus.stats)
    # Reflection: each car writes a lesson for the fleet memory.
    for vid, agent in agents.items():
        v = world.vehicles[vid]
        merged_late = v.start_lane == 0 and (v.merged_at is None or v.merged_at > 12)
        lesson = agent.reflect(v.crashed, merged_late)
        if lesson and memory is not None:
            memory.add(sim.scenario, lesson, vid, cfg.run_name)
            tel.emit(vid, "lesson", text=lesson)
        tel.emit(vid, "agent_summary", **agent.stats, crashed=v.crashed, finished=v.finished,
                 start_lane=v.start_lane, merged_at=v.merged_at, lane_changes=v.lane_changes)
    tel.emit("world", "end", t=round(world.t, 2), wall_s=round(wall, 3))
    tel.close()
    m = episode_metrics(run_dir)
    if verbose:
        print({k: m[k] for k in ("success_rate", "collisions", "mean_speed", "vetoes", "conflicts", "msgs_sent")})
    return m


def _decision_tick(step, world, agents, live, positions, bus, pool, tel, verbose):
    # 1. perceive (local view + inbox)
    obs = {vid: agents[vid].perceive(world, bus.pop_inbox(vid)) for vid in live}
    # 2. decide in parallel - each car thinks on its own
    futures = {vid: pool.submit(agents[vid].decide, obs[vid]) for vid in live}
    props = {vid: f.result() for vid, f in futures.items()}
    # 3. safety shield (local)
    shielded = {vid: agents[vid].apply_shield(props[vid]["action"], obs[vid]) for vid in live}
    # 4. share intents and negotiate (only if the radio is on)
    intents = {vid: agents[vid].make_intent(shielded[vid][0], obs[vid], step) for vid in live}
    received = bus.exchange_intents(intents, positions)
    for vid in live:
        a = agents[vid]
        safe_action, vetoed, why = shielded[vid]
        final, conflicts, winner = a.negotiate(safe_action, obs[vid], step, received.get(vid, []))
        for msg in a.outgoing_messages(obs[vid], step):
            bus.publish(msg)
        applied = a.act(world, final)
        p = props[vid]
        tel.emit(vid, "decision", step=step, t=round(world.t, 2), lane=obs[vid]["lane"],
                 x=round(obs[vid]["x"], 1), v=round(obs[vid]["v"], 2), proposed=p["action"], reason=p["reason"],
                 final=applied, vetoed=vetoed, veto_reason=why, conflicts=conflicts, yielded_to=winner,
                 latency_s=round(p["latency_s"], 4), tokens=p["tokens"], ok=p["ok"], error=p["error"],
                 n_hazards=len(obs[vid]["hazards"]), n_merge_req=len(obs[vid]["merge_requests"]))
        if verbose and (p["action"] != "IDLE" or vetoed or conflicts):
            extra = f" | VETO: {why}" if vetoed else ""
            extra += f" | yields to {winner}" if winner else ""
            print(f"t={world.t:5.1f}s {vid} lane {obs[vid]['lane']} v={obs[vid]['v']:4.1f}  "
                  f"{p['action']:<10} -> {applied:<10} ({p['reason']}){extra}")
    tel.emit("bus", "tick", step=step, **bus.stats)
