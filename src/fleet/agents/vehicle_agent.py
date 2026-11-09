"""One self-driving car = one independent agent.

An agent owns its own state and memory of what it has heard. It only gets:
  * a LOCAL observation (its sensors, limited range), and
  * messages that reached it over the V2V bus.
Each decision cycle:  perceive -> decide (LLM or rule) -> shield -> share intent
-> negotiate -> act -> (after the drive) reflect.
"""
from __future__ import annotations

import time

from ..coordination.negotiation import priority, resolve, target_lane
from ..comms.bus import Message
from ..safety import shield
from ..sim.world import World


class VehicleAgent:
    def __init__(self, vid: str, system: str, llm, use_shield: bool, memory, sensor_range: float,
                 scenario: str, initial_target: float):
        self.vid, self.system, self.llm = vid, system, llm
        self.use_shield, self.memory, self.sensor_range = use_shield, memory, sensor_range
        self.scenario = scenario
        self.base_target = initial_target
        self.known_hazards: dict[tuple, dict] = {}   # (lane, x) -> info heard by radio
        self.announced: set[tuple] = set()
        self.last_intents: list[dict] = []
        self.last_merge_requests: list[dict] = []
        self.stats = {"vetoes": 0, "conflicts": 0, "yielded": 0, "decisions": 0, "llm_fail": 0}

    # ------------------------------------------------------------ perceive
    def _lane_info(self, world: World, lane: int, x: float) -> dict | None:
        if not world.road.lane_exists(lane, x):
            return None
        front, fg, rear, rg = world.neighbours_in_lane(lane, x, self.vid)
        if fg > self.sensor_range:
            front, fg = None, None
        if rg > self.sensor_range:
            rear, rg = None, None
        return {"lane": lane, "front_gap": fg, "front_v": front.v if front else None,
                "rear_gap": rg, "rear_v": rear.v if rear else None,
                "exists_ahead": world.road.lane_exists(lane, x, margin=10.0),
                "is_ramp": world.road.ramp_end is not None and lane == 0}

    def perceive(self, world: World, inbox: list[Message]) -> dict:
        me = world.vehicles[self.vid]
        road = world.road
        merge_requests = []
        for m in inbox:
            if m.mtype == "HAZARD":
                key = (m.payload["lane"], round(m.payload["x"]))
                self.known_hazards[key] = {"lane": m.payload["lane"], "x": m.payload["x"],
                                           "source": f"V2V from {m.sender}", "hops": m.payload.get("hops", 1)}
            elif m.mtype == "MERGE_REQUEST":
                merge_requests.append({"from": m.sender, "dx": m.payload["x"] - me.x, "lane": 0})
        self.last_merge_requests = merge_requests
        hazards = []
        for o in world.active():                                   # own sensors
            if o.kind == "stalled" and 0 < o.x - me.x <= self.sensor_range:
                hazards.append({"lane": o.lane, "dist": o.x - me.x, "x": o.x, "source": "own sensor"})
        for (lane, _), h in self.known_hazards.items():           # heard by radio
            d = h["x"] - me.x
            if d > 0 and not any(z["lane"] == lane and abs(z["x"] - h["x"]) < 5 for z in hazards):
                hazards.append({"lane": lane, "dist": d, "x": h["x"], "source": h["source"]})
        on_ramp = road.ramp_end is not None and me.lane == 0
        return {
            "vid": self.vid, "t": world.t, "scenario": self.scenario, "lane": me.lane, "n_lanes": road.n_lanes,
            "x": me.x, "v": me.v, "v_target": self.base_target,
            "on_ramp": on_ramp, "ramp_left_m": (road.ramp_end - me.x) if on_ramp else None,
            "lanes": {"left": self._lane_info(world, me.lane + 1, me.x),
                      "current": self._lane_info(world, me.lane, me.x),
                      "right": self._lane_info(world, me.lane - 1, me.x)},
            "hazards": hazards, "merge_requests": merge_requests,
            "intents": [i for i in self.last_intents if abs(i["dx"]) < self.sensor_range * 2],
            "lessons": self.memory.retrieve(self.scenario) if self.memory else [],
        }

    # ------------------------------------------------------------ decide
    def decide(self, obs: dict) -> dict:
        self.stats["decisions"] += 1
        if self.system == "rule":
            return rule_policy(obs)
        out = self.llm.decide(obs)
        if not out["ok"]:
            self.stats["llm_fail"] += 1
        return out

    def apply_shield(self, action: str, obs: dict) -> tuple[str, bool, str]:
        if not self.use_shield:
            return action, False, ""
        safe, vetoed, why = shield.check(action, obs)
        if vetoed:
            self.stats["vetoes"] += 1
        return safe, vetoed, why

    def make_intent(self, action: str, obs: dict, step: int) -> Message:
        payload = {"action": action, "lane": obs["lane"], "target_lane": target_lane(obs["lane"], action),
                   "x": round(obs["x"], 1), "priority": priority(obs, action)}
        return Message(self.vid, "INTENT", payload, obs["x"], step)

    def negotiate(self, action: str, obs: dict, step: int, received: list[Message]) -> tuple[str, list, str | None]:
        mine = self.make_intent(action, obs, step).payload
        others = [{**m.payload, "from": m.sender} for m in received]
        self.last_intents = [{"from": o["from"], "action": o["action"], "lane": o["lane"],
                              "dx": o["x"] - obs["x"]} for o in others]
        final, conflicts, winner = resolve(self.vid, mine, others)
        self.stats["conflicts"] += len(conflicts)
        if winner:
            self.stats["yielded"] += 1
        return final, conflicts, winner

    def outgoing_messages(self, obs: dict, step: int) -> list[Message]:
        msgs = []
        if obs["on_ramp"]:
            msgs.append(Message(self.vid, "MERGE_REQUEST", {"x": round(obs["x"], 1),
                                                            "ramp_left": round(obs["ramp_left_m"], 1)}, obs["x"], step))
        for h in obs["hazards"]:
            key = (h["lane"], round(h["x"]))
            hops = 1 if h["source"] == "own sensor" else self.known_hazards.get(key, {}).get("hops", 1) + 1
            if key not in self.announced and hops <= 2:     # announce own sightings, relay once
                self.announced.add(key)
                msgs.append(Message(self.vid, "HAZARD", {"lane": h["lane"], "x": h["x"], "hops": hops},
                                    obs["x"], step))
        return msgs

    # ------------------------------------------------------------ act
    def act(self, world: World, action: str) -> str:
        me = world.vehicles[self.vid]
        if action == "FASTER":
            self.base_target = min(30.0, self.base_target + 2.0)
        elif action == "SLOWER":
            self.base_target = max(8.0, self.base_target - 2.0)
        me.v_des = self.base_target
        if action == "YIELD":
            me.v_des = max(self.base_target - 6.0, 6.0)   # brief slow-down, never cumulative
        elif action in ("LANE_LEFT", "LANE_RIGHT"):
            tgt = target_lane(me.lane, action)
            if not world.road.lane_exists(tgt, me.x):
                return "IDLE"          # physically impossible, ignored
            world.change_lane(me, tgt)
        return action

    # ------------------------------------------------------------ reflect
    def reflect(self, crashed: bool, merged_late: bool) -> str:
        if self.system == "rule" or self.memory is None:
            return ""
        parts = []
        if crashed:
            parts.append("collision")
        if self.stats["vetoes"]:
            parts.append(f"{self.stats['vetoes']} unsafe lane-change veto(es)")
        if self.stats["conflicts"]:
            parts.append(f"{self.stats['conflicts']} lane-change conflict(s)")
        if merged_late:
            parts.append("slow merge from the ramp")
        if not parts:
            return ""
        return self.llm.reflect(self.vid, self.scenario, ", ".join(parts))


def rule_policy(obs: dict) -> dict:
    """Classical baseline (no LLM, no radio): IDM speed control + MOBIL-style lane changes."""
    t0 = time.perf_counter()

    def res(a, why):
        return {"action": a, "reason": why, "latency_s": time.perf_counter() - t0, "ok": True, "tokens": 0, "error": ""}

    def free(info, need):
        if info is None or not info.get("exists_ahead", True) or info.get("is_ramp"):
            return False
        ok, _ = shield._lane_ok(info, obs["v"])
        return ok and (info["front_gap"] is None or info["front_gap"] > need)

    L, C, R = obs["lanes"]["left"], obs["lanes"]["current"], obs["lanes"]["right"]
    if obs["on_ramp"]:
        return res("LANE_LEFT", "ramp: gap found") if free(L, 6) else res("IDLE", "ramp: wait for gap")
    blocked = any(h["lane"] == obs["lane"] for h in obs["hazards"])
    fg, fv = C["front_gap"], C["front_v"]
    slow = fg is not None and fg < 40 and fv is not None and fv < obs["v_target"] - 2
    if blocked or slow:
        need = 20 if blocked else (fg or 0) + 20
        for name, lane in (("LANE_LEFT", L), ("LANE_RIGHT", R)):
            if free(lane, need) and not any(h["lane"] == lane["lane"] for h in obs["hazards"]):
                return res(name, "MOBIL: better lane")
        return res("IDLE", "follow leader")
    if obs["v"] < obs["v_target"] - 1 and (fg is None or fg > 50):
        return res("FASTER", "free road")
    return res("IDLE", "cruise")
