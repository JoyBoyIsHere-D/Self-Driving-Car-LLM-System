"""The simulated road world (the "physics").

The world only moves cars and detects crashes. It never decides anything
for an agent: agents read a *local* observation from it and send back one
tactical action. Background "human" cars follow a simple rule model:
IDM (Intelligent Driver Model) for speed, plus a basic lane-change rule.

Coordinates: x = metres along the road, lane 0 = rightmost lane.
In the merge scenario lane 0 is an on-ramp that ends at `road.ramp_end`.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

CAR_LEN = 5.0      # metres
V_MAX = 32.0       # m/s (about 115 km/h)


@dataclass
class Vehicle:
    vid: str
    x: float
    lane: int
    v: float
    v_des: float                 # desired speed used by IDM
    kind: str                    # "agent" | "human" | "stalled"
    a: float = 0.0
    crashed: bool = False
    finished: bool = False
    lane_cooldown: float = 0.0   # seconds before a human may change lane again
    start_lane: int = 0
    merged_at: float | None = None
    lane_changes: int = 0

    @property
    def active(self) -> bool:
        return not self.crashed and not self.finished


@dataclass
class Road:
    n_lanes: int
    length: float                # finish line (m)
    ramp_end: float | None = None  # lane 0 ends here (merge scenario)

    def lane_exists(self, lane: int, x: float, margin: float = 0.0) -> bool:
        if lane < 0 or lane >= self.n_lanes:
            return False
        if self.ramp_end is not None and lane == 0 and x + margin > self.ramp_end:
            return False
        return True


def idm_acc(v: float, v0: float, gap: float, v_lead: float,
            a_max: float = 1.5, b: float = 2.0, T: float = 1.2, s0: float = 2.0) -> float:
    """Intelligent Driver Model: acceleration that tracks v0 while keeping a safe gap."""
    v0 = max(v0, 0.1)
    free = 1.0 - (v / v0) ** 4
    if gap is None or math.isinf(gap):
        return a_max * free
    dv = v - v_lead
    s_star = s0 + max(0.0, v * T + v * dv / (2.0 * math.sqrt(a_max * b)))
    return a_max * (free - (s_star / max(gap, 0.1)) ** 2)


@dataclass
class World:
    road: Road
    vehicles: dict[str, Vehicle]
    dt: float
    t: float = 0.0
    step_count: int = 0
    events: list[dict] = field(default_factory=list)   # filled during step(), read by the runner

    # ------------------------------------------------------------------ queries
    def active(self) -> list[Vehicle]:
        return [v for v in self.vehicles.values() if v.active]

    def neighbours_in_lane(self, lane: int, x: float, exclude: str | None = None):
        """Nearest vehicle ahead and behind position x in a lane.

        Returns (front_vehicle, front_gap, rear_vehicle, rear_gap); gaps are
        bumper-to-bumper metres, or inf when nobody is there.
        """
        front, rear = None, None
        f_gap, r_gap = math.inf, math.inf
        for o in self.vehicles.values():
            if not o.active or o.vid == exclude or o.lane != lane:
                continue
            d = o.x - x
            if d >= 0:
                g = d - CAR_LEN
                if g < f_gap:
                    front, f_gap = o, g
            else:
                g = -d - CAR_LEN
                if g < r_gap:
                    rear, r_gap = o, g
        return front, f_gap, rear, r_gap

    def leader_gap(self, veh: Vehicle) -> tuple[float, float]:
        """Gap and speed of whatever is in front (a car, or the end of the ramp)."""
        front, gap, _, _ = self.neighbours_in_lane(veh.lane, veh.x, veh.vid)
        v_lead = front.v if front else veh.v
        if self.road.ramp_end is not None and veh.lane == 0 and veh.x <= self.road.ramp_end:
            end_gap = self.road.ramp_end - veh.x
            if end_gap < gap:
                gap, v_lead = end_gap, 0.0
        return gap, v_lead

    def lane_change_is_safe(self, veh: Vehicle, target: int, min_ttc: float = 2.5) -> tuple[bool, str]:
        """Ground-truth gap check used by human drivers and by the agents' safety shield."""
        if not self.road.lane_exists(target, veh.x, margin=10.0):
            return False, "lane does not exist here"
        front, f_gap, rear, r_gap = self.neighbours_in_lane(target, veh.x, veh.vid)
        if f_gap < max(6.0, 0.3 * veh.v):
            return False, f"front gap {f_gap:.0f} m too short"
        if front is not None and veh.v > front.v and f_gap / (veh.v - front.v) < min_ttc:
            return False, "closing too fast on the car ahead"
        if rear is not None:
            if r_gap < max(6.0, 0.3 * rear.v):
                return False, f"rear gap {r_gap:.0f} m too short"
            if rear.v > veh.v and r_gap / (rear.v - veh.v) < min_ttc:
                return False, "car behind is closing too fast"
        return True, "ok"

    # ------------------------------------------------------------------ actions
    def change_lane(self, veh: Vehicle, target: int) -> None:
        veh.lane = target
        veh.lane_changes += 1
        veh.lane_cooldown = 2.0
        if veh.start_lane == 0 and self.road.ramp_end is not None and target >= 1 and veh.merged_at is None:
            veh.merged_at = self.t
            self.events.append({"event": "merge", "vid": veh.vid, "t": self.t, "x": veh.x})

    def _human_lane_change(self, veh: Vehicle) -> None:
        """Simple rule: overtake a slow leader if an adjacent lane is clearly better and safe."""
        if veh.lane_cooldown > 0:
            return
        front, gap, _, _ = self.neighbours_in_lane(veh.lane, veh.x, veh.vid)
        if front is None or gap > 35 or front.v > veh.v_des - 2:
            return
        for target in (veh.lane + 1, veh.lane - 1):
            if self.road.ramp_end is not None and target == 0:
                continue  # humans never use the on-ramp
            if not self.road.lane_exists(target, veh.x):
                continue
            _, t_gap, _, _ = self.neighbours_in_lane(target, veh.x, veh.vid)
            ok, _ = self.lane_change_is_safe(veh, target, min_ttc=3.0)
            if ok and t_gap > gap + 15:
                self.change_lane(veh, target)
                return

    # ------------------------------------------------------------------ physics
    def step(self) -> None:
        self.events = []
        movers = [v for v in self.active() if v.kind != "stalled"]
        if self.step_count % 4 == 0:
            for v in movers:
                if v.kind == "human":
                    self._human_lane_change(v)
        for v in movers:
            gap, v_lead = self.leader_gap(v)
            v.a = max(-9.0, min(3.0, idm_acc(v.v, v.v_des, gap, v_lead)))
        for v in movers:
            v.v = max(0.0, min(V_MAX, v.v + v.a * self.dt))
            v.x += v.v * self.dt
            v.lane_cooldown = max(0.0, v.lane_cooldown - self.dt)
        self._detect_collisions()
        for v in movers:
            if v.active and v.x >= self.road.length:
                v.finished = True
                self.events.append({"event": "finish", "vid": v.vid, "t": self.t, "kind": v.kind})
        self.t += self.dt
        self.step_count += 1

    def _detect_collisions(self) -> None:
        by_lane: dict[int, list[Vehicle]] = {}
        for v in self.active():
            by_lane.setdefault(v.lane, []).append(v)
        for lane, vs in by_lane.items():
            vs.sort(key=lambda o: o.x)
            for a, b in zip(vs, vs[1:]):
                if b.x - a.x < CAR_LEN and not (a.crashed or b.crashed):
                    for o in (a, b):
                        if o.kind != "stalled":
                            o.crashed = True
                            o.v = 0.0
                    self.events.append({"event": "collision", "t": self.t, "lane": lane,
                                        "x": round(a.x, 1), "a": a.vid, "b": b.vid,
                                        "kinds": [a.kind, b.kind]})
