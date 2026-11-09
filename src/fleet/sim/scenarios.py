"""Two traffic scenarios, each built deterministically from a seed.

highway : 3 lanes. Slow cars in the outer lanes push agents to overtake through
          the middle lane at the same moment (a lane-change conflict), and a
          broken-down car sits in the middle lane far beyond sensor range
          (a hazard that only V2V warnings reveal early).
merge   : an on-ramp (lane 0) ends at 260 m next to a dense platoon in lane 1.
          Ramp agents must merge; main-lane agents can help by yielding.
"""
from __future__ import annotations

import numpy as np

from .world import Road, Vehicle, World


def _jit(rng: np.random.Generator, s: float) -> float:
    return float(rng.uniform(-s, s))


def make_world(scenario: str, n_agents: int, n_humans: int, dt: float, seed: int) -> World:
    rng = np.random.default_rng(seed)
    if scenario == "highway":
        return _highway(rng, n_agents, n_humans, dt)
    if scenario == "merge":
        return _merge(rng, n_agents, n_humans, dt)
    raise ValueError(f"unknown scenario {scenario!r}")


def _highway(rng, n_agents, n_humans, dt) -> World:
    road = Road(n_lanes=3, length=900.0)
    vs: dict[str, Vehicle] = {}
    # Agents come in roughly side-by-side pairs in the two outer lanes, so two
    # cars often want the same middle-lane gap at about the same time.
    for i in range(n_agents):
        lane = 0 if i % 2 == 0 else 2
        x = 70.0 - 45.0 * (i // 2) + (0.0 if lane == 0 else _jit(rng, 8))
        vs[f"A{i}"] = Vehicle(f"A{i}", x, lane, 22.0 + _jit(rng, 1), 27.0, "agent", start_lane=lane)
    # Broken-down car in the middle lane, far ahead (hazard).
    vs["S0"] = Vehicle("S0", 560.0, 1, 0.0, 0.0, "stalled", start_lane=1)
    # Humans: slow in the outer lanes, faster in the middle lane.
    plan = []
    for k in range(n_humans):
        lane = (0, 2, 1)[k % 3]
        slot = k // 3
        if lane == 1:
            x, v = -40.0 + 170.0 * slot, 26.0
        else:  # slow trucks driving side by side in the two outer lanes
            x, v = 125.0 + 80.0 * slot, 19.0
        plan.append((lane, x, v))
    for k, (lane, x, v) in enumerate(plan):
        vs[f"H{k}"] = Vehicle(f"H{k}", x + _jit(rng, 3), lane, v + _jit(rng, 1), v, "human", start_lane=lane)
    return World(road, vs, dt)


def _merge(rng, n_agents, n_humans, dt) -> World:
    road = Road(n_lanes=3, length=500.0, ramp_end=260.0)
    vs: dict[str, Vehicle] = {}
    n_ramp = (n_agents + 1) // 2
    n_main = n_agents - n_ramp
    # Congested platoon in lane 1: 20 m apart (15 m bumper gaps) at 11 m/s.
    # A merging car needs about 17 m (5 m car + 6 m front + 6 m rear), so it
    # can only get in if a main-lane car makes room. Some platoon slots are
    # main-lane agents; each ramp agent starts just ahead of one of them.
    slots = [220.0 - 20.0 * k for k in range(16)]
    agent_slots = [3 + 3 * k for k in range(n_main)]          # x = 160, 100, 40, ...
    h = 0
    for k, x in enumerate(slots):
        if k in agent_slots:
            i = n_ramp + agent_slots.index(k)
            vs[f"A{i}"] = Vehicle(f"A{i}", x + _jit(rng, 1), 1, 11.0, 11.0, "agent", start_lane=1)
        elif h < int(n_humans * 0.7) + 1:
            vs[f"H{h}"] = Vehicle(f"H{h}", x + _jit(rng, 1.5), 1, 11.0, 11.0 + _jit(rng, 0.3), "human", start_lane=1)
            h += 1
    # Ramp agents, each slightly ahead of a main-lane agent.
    for j in range(n_ramp):
        base = slots[agent_slots[j]] if j < n_main else slots[-1] - 30.0 * (j - n_main + 1)
        vs[f"A{j}"] = Vehicle(f"A{j}", base + 8.0 + _jit(rng, 3), 0, 11.0, 18.0, "agent", start_lane=0)
    # A few faster humans in the left lane.
    while h < n_humans:
        x = 20.0 + 85.0 * (h % 4)
        vs[f"H{h}"] = Vehicle(f"H{h}", x + _jit(rng, 3), 2, 27.0, 28.0, "human", start_lane=2)
        h += 1
    return World(road, vs, dt)
