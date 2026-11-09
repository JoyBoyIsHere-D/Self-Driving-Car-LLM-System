"""Safety shield: a small rule-based check that sits under every LLM decision.

LLMs are slow and can be wrong, so the LLM only *proposes* an action. The
shield vetoes proposals that are clearly unsafe right now (short gaps, a car
closing fast, a lane that does not exist) and replaces them with a safe one.
It uses only the car's own sensor view, never other cars' plans.
"""
from __future__ import annotations

MIN_TTC = 2.5   # seconds


def _lane_ok(info: dict | None, v: float) -> tuple[bool, str]:
    if info is None:
        return False, "no lane there"
    fg, fv, rg, rv = info["front_gap"], info["front_v"], info["rear_gap"], info["rear_v"]
    if (fg is not None and fg < 0) or (rg is not None and rg < 0):
        return False, "a car is right beside me in that lane"
    if fg is not None:
        if fg < max(6.0, 0.3 * v):
            return False, f"front gap {fg:.0f} m too short"
        if fv is not None and v > fv and fg / (v - fv) < MIN_TTC:
            return False, "closing too fast on the car ahead"
    if rg is not None and rv is not None:
        if rg < max(6.0, 0.3 * rv):
            return False, f"rear gap {rg:.0f} m too short"
        if rv > v and rg / (rv - v) < MIN_TTC:
            return False, "car behind is closing too fast"
    return True, "ok"


def check(action: str, obs: dict) -> tuple[str, bool, str]:
    """Return (safe_action, vetoed, reason)."""
    if action in ("LANE_LEFT", "LANE_RIGHT"):
        key = "left" if action == "LANE_LEFT" else "right"
        info = obs["lanes"][key]
        if info is not None and not info.get("exists_ahead", True):
            return "IDLE", True, "target lane ends just ahead"
        if info is not None and info.get("is_ramp"):
            return "IDLE", True, "cannot enter the on-ramp"
        ok, why = _lane_ok(info, obs["v"])
        if not ok:
            return ("SLOWER" if obs["on_ramp"] and obs["ramp_left_m"] < 40 else "IDLE"), True, why
    if action == "FASTER":
        c = obs["lanes"]["current"]
        if c["front_gap"] is not None and c["front_v"] is not None and obs["v"] > c["front_v"] \
                and c["front_gap"] / (obs["v"] - c["front_v"]) < 4.0:
            return "IDLE", True, "no room to speed up"
    return action, False, ""
