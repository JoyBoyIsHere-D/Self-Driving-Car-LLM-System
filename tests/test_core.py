"""Unit + integration tests (fast: they use the built-in stand-in LLM)."""
import math

import numpy as np

from fleet.comms.bus import Message, MessageBus
from fleet.config import make_config
from fleet.coordination.negotiation import resolve
from fleet.memory.lessons import LessonStore
from fleet.runner import run_episode
from fleet.safety import shield
from fleet.sim.world import Road, Vehicle, World, idm_acc


def _lane(front_gap=None, front_v=None, rear_gap=None, rear_v=None, lane=1):
    return {"lane": lane, "front_gap": front_gap, "front_v": front_v, "rear_gap": rear_gap,
            "rear_v": rear_v, "exists_ahead": True, "is_ramp": False}


def _obs(left=None, v=20.0):
    return {"vid": "A0", "v": v, "lane": 1, "on_ramp": False, "ramp_left_m": None, "hazards": [],
            "lanes": {"left": left, "current": _lane(), "right": None}}


def test_idm_brakes_for_stopped_car():
    assert idm_acc(20.0, 25.0, 10.0, 0.0) < -4
    assert idm_acc(10.0, 25.0, math.inf, 0.0) > 0


def test_collision_detected():
    road = Road(2, 500)
    w = World(road, {"a": Vehicle("a", 10, 0, 10, 10, "agent"), "b": Vehicle("b", 12, 1, 10, 10, "agent")}, 0.25)
    w.change_lane(w.vehicles["b"], 0)
    w.step()
    assert w.vehicles["a"].crashed and w.vehicles["b"].crashed


def test_shield_vetoes_short_rear_gap():
    action, vetoed, why = shield.check("LANE_LEFT", _obs(left=_lane(rear_gap=3.0, rear_v=25.0, lane=2)))
    assert vetoed and action == "IDLE" and "rear" in why


def test_shield_allows_safe_lane_change():
    action, vetoed, _ = shield.check("LANE_LEFT", _obs(left=_lane(front_gap=50, front_v=25, rear_gap=40, rear_v=18, lane=2)))
    assert not vetoed and action == "LANE_LEFT"


def test_negotiation_lower_priority_yields():
    mine = {"action": "LANE_LEFT", "target_lane": 1, "x": 100.0, "priority": [1, -1e9, -3]}
    other = {"from": "A1", "action": "LANE_RIGHT", "target_lane": 1, "x": 104.0, "priority": [1, -1e9, -1]}
    final, conflicts, winner = resolve("A3", mine, [other])
    assert final == "IDLE" and conflicts == ["A1"] and winner == "A1"
    # and the winner keeps its action (same rule, run on the other car)
    final2, _, winner2 = resolve("A1", {**other}, [{**mine, "from": "A3"}])
    assert final2 == "LANE_RIGHT" and winner2 is None


def test_bus_range_and_loss():
    bus = MessageBus(range_m=100, latency_steps=1, loss_prob=0.0, rng=np.random.default_rng(0))
    bus.publish(Message("A0", "HAZARD", {"lane": 1, "x": 50}, 0.0, step_sent=0))
    bus.deliver(0, {"A1": 50.0, "A2": 500.0})
    assert bus.pop_inbox("A1") == []          # not yet: 1-step latency
    bus.deliver(1, {"A1": 50.0, "A2": 500.0})
    assert len(bus.pop_inbox("A1")) == 1 and bus.pop_inbox("A2") == []   # A2 out of range
    lossy = MessageBus(range_m=100, latency_steps=0, loss_prob=1.0, rng=np.random.default_rng(0))
    lossy.publish(Message("A0", "HAZARD", {}, 0.0, step_sent=0))
    lossy.deliver(0, {"A1": 10.0})
    assert lossy.pop_inbox("A1") == [] and lossy.stats["dropped"] == 1


def test_lesson_store_dedup(tmp_path):
    store = LessonStore(str(tmp_path / "l.json"))
    store.add("merge", "Yield early.", "A0", "e1")
    store.add("merge", "yield early.", "A1", "e2")
    assert store.retrieve("merge") == ["Yield early."]


def test_episode_runs_and_logs(tmp_path):
    for system in ("rule", "llm_nocomm", "llm_comm"):
        cfg = make_config(system=system, scenario="merge", seed=1, run_name=system, out_dir=str(tmp_path))
        m = run_episode(cfg, lessons_path=str(tmp_path / "lessons.json"))
        assert m["decisions"] > 0 and 0.0 <= m["success_rate"] <= 1.0
        assert (m["msgs_sent"] > 0) == (system == "llm_comm")
