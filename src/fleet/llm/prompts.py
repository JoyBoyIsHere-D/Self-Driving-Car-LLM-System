"""Turns a car's local observation into a short text prompt for the LLM."""
from __future__ import annotations

ACTIONS = ["FASTER", "SLOWER", "IDLE", "LANE_LEFT", "LANE_RIGHT", "YIELD"]

SYSTEM_PROMPT = """You are the driving brain of ONE self-driving car in a fleet.
You only know what your own sensors see and what nearby cars tell you by radio (V2V).
Choose ONE tactical action for the next second.

Actions:
- FASTER: raise target speed
- SLOWER: lower target speed
- IDLE: keep lane and speed
- LANE_LEFT: move one lane to the left
- LANE_RIGHT: move one lane to the right
- YIELD: slow down briefly so another car can merge in front of you

Driving rules:
1. Only change lane when BOTH the front gap and the rear gap in that lane are large (more than 15 m).
2. If you are on the on-ramp you MUST merge left before the ramp ends.
3. If a car on the ramp next to you asks to merge, help it: YIELD, or move LANE_LEFT if that lane is free.
4. Leave a lane that has a hazard (stopped car) ahead of you, early and safely.
5. Otherwise drive close to your target speed and avoid needless lane changes.

Reply ONLY with JSON: {"action": "<ONE ACTION>", "reason": "<at most 15 words>"}"""


def _gap(g) -> str:
    if g is None:
        return "none in sensor range"
    return "right beside you (overlapping)" if g < 0 else f"{g:.0f} m"


def _lane_line(name: str, info: dict | None) -> str:
    if info is None:
        return f"- {name}: no lane"
    if info.get("is_ramp") and name != "Your":
        return f"- {name} lane {info['lane']}: ON-RAMP (entry only, never move into it)"
    fv = "" if info["front_v"] is None else f" (speed {info['front_v']:.0f} m/s)"
    rv = "" if info["rear_v"] is None else f" (speed {info['rear_v']:.0f} m/s)"
    return f"- {name} lane {info['lane']}: car ahead {_gap(info['front_gap'])}{fv}; car behind {_gap(info['rear_gap'])}{rv}"


def build_prompt(obs: dict) -> str:
    lines = [
        f"You are car {obs['vid']} at t={obs['t']:.0f} s. Scenario: {obs['scenario']}.",
        f"Your lane: {obs['lane']} of lanes 0..{obs['n_lanes'] - 1} (0 = rightmost, LEFT = higher number).",
        f"Speed {obs['v']:.1f} m/s, target {obs['v_target']:.0f} m/s.",
    ]
    if obs["on_ramp"]:
        lines.append(f"You are ON THE RAMP. It ends in {obs['ramp_left_m']:.0f} m. Merge LEFT as soon as it is safe.")
    lines.append("Lanes (from your sensors):")
    for key, name in (("left", "Left"), ("current", "Your"), ("right", "Right")):
        lines.append(_lane_line(name, obs["lanes"][key]))
    if obs["hazards"]:
        lines.append("Hazards:")
        for h in obs["hazards"]:
            lines.append(f"- stopped vehicle in lane {h['lane']}, {h['dist']:.0f} m ahead (source: {h['source']})")
    if obs["merge_requests"]:
        lines.append("Radio messages:")
        for m in obs["merge_requests"]:
            where = "ahead of" if m["dx"] > 0 else "behind"
            lines.append(f"- {m['from']} on the ramp, {abs(m['dx']):.0f} m {where} you, asks to merge into lane 1")
    if obs["intents"]:
        lines.append("Neighbour plans (last second):")
        for it in obs["intents"]:
            lines.append(f"- {it['from']} plans {it['action']} (now lane {it['lane']}, {it['dx']:+.0f} m from you)")
    if obs["lessons"]:
        lines.append("Fleet lessons from earlier drives:")
        for l in obs["lessons"]:
            lines.append(f"- {l}")
    lines.append("Choose your action now.")
    return "\n".join(lines)


REFLECT_PROMPT = """You drove car {vid} in the {scenario} scenario. What happened: {events}.
Write ONE short lesson (max 20 words) that would help other cars in the fleet drive better next time.
Reply ONLY with JSON: {{"lesson": "<text>"}}"""
