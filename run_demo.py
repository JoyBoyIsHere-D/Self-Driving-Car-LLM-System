"""Watch one episode: prints every car's decisions and saves an animation.

  python run_demo.py                                   # merge scenario, stand-in LLM
  python run_demo.py --scenario highway --backend ollama --model llama3.2:3b
  python run_demo.py --snapshots                       # scenario pictures for the report
"""
from __future__ import annotations

import argparse
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.animation import FuncAnimation, PillowWriter  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
from fleet.config import make_config  # noqa: E402
from fleet.llm.client import OllamaLLM  # noqa: E402
from fleet.runner import run_episode  # noqa: E402
from fleet.sim.scenarios import make_world  # noqa: E402

AGENT, HUMAN, STALLED, INK, INK2 = "#2a78d6", "#b9b8b3", "#e34948", "#0b0b0b", "#52514e"


def draw(ax, cars, scenario, t, x_lo, x_hi, title=None):
    ax.clear()
    lanes = 3
    ax.add_patch(Rectangle((x_lo, -0.5), x_hi - x_lo, lanes, color="#ecebe7", zorder=0))
    for k in range(1, lanes):
        ax.plot([x_lo, x_hi], [k - 0.5, k - 0.5], ls=(0, (6, 6)), color="white", lw=1.5, zorder=1)
    if scenario == "merge":
        ax.add_patch(Rectangle((260, -0.5), x_hi, 1.0, color="white", zorder=1))
        ax.text(262, -0.05, "ramp ends", fontsize=8, color=INK2, va="center")
    for vid, kind, x, lane, v, crashed in cars:
        if not (x_lo - 10 < x < x_hi + 10):
            continue
        color = AGENT if kind == "agent" else STALLED if kind == "stalled" else HUMAN
        ax.add_patch(Rectangle((x - 2.5, lane - 0.22), 5.0, 0.44, color=color, zorder=3))
        if kind == "agent":
            ax.text(x, lane + 0.33, vid, ha="center", fontsize=7.5, color=INK, zorder=4)
        if crashed:
            ax.plot([x], [lane], marker="x", color=INK, ms=10, mew=2, zorder=5)
    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(-0.6, lanes - 0.4)
    ax.set_yticks(range(lanes), [f"lane {k}" + (" (ramp)" if scenario == "merge" and k == 0 else "")
                                 for k in range(lanes)], fontsize=8)
    ax.set_xlabel("position along the road (m)", fontsize=8, color=INK2)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title(title or f"{scenario} · t = {t:.1f} s   (blue = LLM agents, grey = human drivers, red = broken-down car)",
                 loc="left", fontsize=9.5, color=INK)


def animate(frames, scenario, dt, path, every=2):
    fig, ax = plt.subplots(figsize=(10, 2.3))
    idx = list(range(0, len(frames), every))

    def update(i):
        cars = frames[idx[i]]
        agents = [c for c in cars if c[1] == "agent"]
        cx = sum(c[2] for c in agents) / max(1, len(agents)) if agents else 0
        lo = 0 if scenario == "merge" else max(-50, cx - 150)
        hi = 500 if scenario == "merge" else lo + 350
        draw(ax, cars, scenario, idx[i] * dt, lo, hi)
        fig.tight_layout()

    anim = FuncAnimation(fig, update, frames=len(idx))
    anim.save(path, writer=PillowWriter(fps=8))
    plt.close(fig)
    print("wrote", path)


def snapshots(path):
    fig, axes = plt.subplots(2, 1, figsize=(11, 4.4))
    w = make_world("highway", 4, 10, 0.25, 0)
    cars = [(o.vid, o.kind, o.x, o.lane, o.v, False) for o in w.vehicles.values()]
    draw(axes[0], cars, "highway", 0, -60, 600,
         "Highway: slow trucks side by side push agents to overtake at once; a broken-down car waits at 560 m")
    w = make_world("merge", 4, 10, 0.25, 0)
    cars = [(o.vid, o.kind, o.x, o.lane, o.v, False) for o in w.vehicles.values()]
    draw(axes[1], cars, "merge", 0, -100, 500,
         "Merge: ramp agents (lane 0) must enter a congested lane 1; main-lane agents can make room")
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote", path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="merge", choices=["highway", "merge"])
    ap.add_argument("--system", default="llm_comm", choices=["rule", "llm_nocomm", "llm_comm"])
    ap.add_argument("--backend", default="mock", choices=["mock", "ollama"])
    ap.add_argument("--model", default="llama3.2:3b")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--snapshots", action="store_true")
    a = ap.parse_args()
    os.makedirs("docs/figures", exist_ok=True)
    if a.snapshots:
        snapshots("docs/figures/scenarios.png")
        return
    if a.backend == "ollama":
        ok, why = OllamaLLM(a.model, "http://localhost:11434").available()
        if not ok:
            sys.exit(why)
    cfg = make_config(system=a.system, scenario=a.scenario, seed=a.seed, backend=a.backend, model=a.model,
                      run_name=f"demo_{a.scenario}_{a.system}_{a.backend}", out_dir="results/demo")
    frames = []
    print(f"--- {a.scenario} | {a.system} | decisions by {a.backend} {a.model if a.backend == 'ollama' else ''}")
    m = run_episode(cfg, lessons_path="results/demo/lessons.json", verbose=True, frames=frames)
    print("\nResult:", {k: m[k] for k in ("success_rate", "collisions", "merge_time", "vetoes", "conflicts",
                                          "msgs_sent", "llm_latency_mean")})
    animate(frames, a.scenario, cfg.sim.dt, f"docs/figures/demo_{a.scenario}_{a.system}.gif")


if __name__ == "__main__":
    main()
