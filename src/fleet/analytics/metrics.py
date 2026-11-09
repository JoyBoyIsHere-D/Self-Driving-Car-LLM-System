"""Analytics: collect every node's log file and compute episode metrics."""
from __future__ import annotations

import glob
import json
import os

import numpy as np
import pandas as pd


def load_events(run_dir: str) -> pd.DataFrame:
    rows = []
    for path in glob.glob(os.path.join(run_dir, "nodes", "*.jsonl")):
        with open(path, encoding="utf-8") as f:
            rows.extend(json.loads(line) for line in f if line.strip())
    return pd.DataFrame(rows)


def episode_metrics(run_dir: str) -> dict:
    ev = load_events(run_dir)
    meta = ev[ev.event == "meta"].iloc[0].to_dict()
    end = ev[ev.event == "end"].iloc[0]
    summ = ev[ev.event == "agent_summary"]
    states = ev[ev.event == "state"]
    dec = ev[ev.event == "decision"]
    col = ev[ev.event == "collision"]
    bus = ev[(ev.node == "bus") & (ev.event == "summary")].iloc[0]
    n = len(summ)
    agent_ids = set(summ.node)
    agent_cols = col[col.apply(lambda r: r["a"] in agent_ids or r["b"] in agent_ids, axis=1)] if len(col) else col
    ramp = summ[summ.start_lane == 0] if meta["scenario"] == "merge" else summ.iloc[0:0]
    merged = ramp[ramp.merged_at.notna()] if len(ramp) else ramp
    ttc = states.ttc.dropna() if "ttc" in states else pd.Series(dtype=float)
    moving = states[~states.crashed.astype(bool)] if len(states) else states
    lat = dec.latency_s.astype(float) if len(dec) else pd.Series(dtype=float)
    sim_t = float(end.t)
    return {
        "scenario": meta["scenario"], "system": meta["system"], "n_agents": int(meta["n_agents"]),
        "backend": meta["backend"], "model": meta["model"], "seed": int(meta["seed"]),
        "run": os.path.basename(run_dir),
        # safety
        "collisions": int(len(agent_cols)),
        "crash_rate": float(summ.crashed.astype(bool).mean()) if n else 0.0,
        "min_ttc": float(ttc.min()) if len(ttc) else np.nan,
        "unsafe_ttc_frac": float((ttc < 1.5).sum() / max(1, len(moving))),
        "hard_brakes": int((moving.a.astype(float) < -4.0).sum()) if len(moving) else 0,
        "vetoes": int(dec.vetoed.sum()) if len(dec) else 0,
        "veto_rate": float(dec.vetoed.mean()) if len(dec) else 0.0,
        # success & efficiency
        "success_rate": float((summ.finished.astype(bool) & ~summ.crashed.astype(bool)).mean()) if n else 0.0,
        "mean_speed": float(moving.v.astype(float).mean()) if len(moving) else 0.0,
        "lane_changes": int(summ.lane_changes.sum()),
        "merge_success": float(len(merged) / len(ramp)) if len(ramp) else np.nan,
        "merge_time": float(merged.merged_at.mean()) if len(merged) else np.nan,
        # coordination & communication
        "conflicts": int(summ.conflicts.sum()) // 2 if n else 0,   # each conflict is seen by both cars
        "yields": int(summ.yielded.sum()),
        "msgs_sent": int(bus.sent), "msgs_delivered": int(bus.delivered), "msgs_dropped": int(bus.dropped),
        "bytes_per_agent_s": float(bus.bytes) / max(1, n) / max(sim_t, 1e-6),
        # LLM cost
        "decisions": int(len(dec)),
        "llm_latency_mean": float(lat.mean()) if len(lat) else 0.0,
        "llm_latency_p95": float(lat.quantile(0.95)) if len(lat) else 0.0,
        "llm_fail_rate": float((~dec.ok.astype(bool)).mean()) if len(dec) else 0.0,
        "tokens_per_decision": float(dec.tokens.astype(float).mean()) if len(dec) else 0.0,
        "sim_time_s": sim_t, "wall_s": float(end.wall_s),
    }
