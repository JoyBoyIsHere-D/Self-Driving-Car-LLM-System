"""Decentralised conflict resolution between cars (no central controller).

Every second each car broadcasts its INTENT (planned action, target lane,
position, priority). Two lane changes conflict when they aim at the same
lane within CONFLICT_DIST metres of each other. Each car runs the SAME
deterministic rule on what it received, so all cars reach the same answer
on their own: the higher-priority car goes, the other one waits.

Priority (bigger wins):  (urgency, closeness to ramp end, lower car id)
  urgency 3 = on the ramp and it ends soon (must merge)
  urgency 2 = escaping a hazard in its current lane
  urgency 1 = ordinary overtaking / lane change
If a message is lost, the cars may disagree - that is exactly the failure
mode the packet-loss experiment measures; the safety shield is the backstop.
"""
from __future__ import annotations

CONFLICT_DIST = 15.0
LANE_CHANGES = ("LANE_LEFT", "LANE_RIGHT")


def target_lane(lane: int, action: str) -> int:
    return lane + 1 if action == "LANE_LEFT" else lane - 1 if action == "LANE_RIGHT" else lane


def priority(obs: dict, action: str) -> list:
    if obs["on_ramp"] and obs["ramp_left_m"] < 150:
        urgency = 3
    elif any(h["lane"] == obs["lane"] and h["dist"] < 300 for h in obs["hazards"]):
        urgency = 2
    else:
        urgency = 1
    ramp_term = -obs["ramp_left_m"] if obs["on_ramp"] else -1e9
    return [urgency, round(ramp_term, 1), -int(obs["vid"][1:])]


def resolve(vid: str, my_intent: dict, received: list[dict]) -> tuple[str, list[str], str | None]:
    """Return (final_action, ids_of_conflicting_cars, winner_id_if_I_yield)."""
    if my_intent["action"] not in LANE_CHANGES:
        return my_intent["action"], [], None
    conflicts, best = [], None
    for other in received:
        if other["action"] not in LANE_CHANGES or other["target_lane"] != my_intent["target_lane"]:
            continue
        if abs(other["x"] - my_intent["x"]) >= CONFLICT_DIST:
            continue
        conflicts.append(other["from"])
        if other["priority"] > my_intent["priority"] and (best is None or other["priority"] > best["priority"]):
            best = other
    if best is not None:
        return "IDLE", conflicts, best["from"]
    return my_intent["action"], conflicts, None
