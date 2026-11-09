"""Turn experiment CSVs into the figures used in the report and slides.

  python make_figures.py                 # figures from results/mock
  python make_figures.py --tag ollama_llama3.2-3b
Figures go to docs/figures/<tag>/ ; diagrams go to docs/figures/.
"""
from __future__ import annotations

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

# Colours (validated categorical slots 1-3, light mode) and text inks.
OURS, NOCOMM, RULE = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#ffffff"
SYSTEMS = [("rule", "Rule-based (no LLM)", RULE), ("llm_nocomm", "LLM, no V2V", NOCOMM),
           ("llm_comm", "LLM + V2V (ours)", OURS)]

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 0.8,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.dpi": 200, "legend.frameon": False,
})


def ci(x: pd.Series) -> float:
    x = x.dropna()
    return 1.96 * x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else 0.0


def finish(fig, title: str, subtitle: str, path: str, legend_ax=None) -> None:
    fig.text(0.01, 0.985, title, ha="left", va="top", fontsize=13, fontweight="bold", color=INK)
    fig.text(0.01, 0.90, subtitle, ha="left", va="top", fontsize=9.5, color=INK2)
    bottom = 0.0
    if legend_ax is not None:
        h, l = legend_ax.get_legend_handles_labels()
        fig.legend(h, l, loc="lower center", ncol=len(l), fontsize=9.5, bbox_to_anchor=(0.5, 0.0))
        bottom = 0.09
    fig.tight_layout(rect=(0, bottom, 1, 0.83))
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print("wrote", path)


def bar_panel(ax, df, metric, scenarios, ylabel, fmt="{:.2f}"):
    width = 0.26
    xs = np.arange(len(scenarios))
    for k, (sys_, label, color) in enumerate(SYSTEMS):
        means, errs = [], []
        for sc in scenarios:
            d = df[(df.system == sys_) & (df.scenario == sc)][metric]
            means.append(d.mean())
            errs.append(ci(d))
        pos = xs + (k - 1) * width
        ax.bar(pos, means, width - 0.03, yerr=errs, color=color, label=label, capsize=3,
               error_kw={"elinewidth": 1, "ecolor": INK2})
        for p, m, e in zip(pos, means, errs):
            if not np.isnan(m):
                ax.annotate(fmt.format(m), (p, m + e), ha="center", va="bottom", fontsize=8, color=INK2,
                            xytext=(0, 3), textcoords="offset points")
    ax.set_xticks(xs, [s.capitalize() for s in scenarios])
    ax.set_ylabel(ylabel)
    ax.set_ylim(bottom=0)


def subtitle_for(tag: str, df: pd.DataFrame) -> str:
    seeds = df.seed.nunique()
    who = "built-in stand-in decision model (not an LLM)" if tag == "mock" else f"local LLM {df.model.iloc[0]}"
    return f"Decisions by {who} · mean of {seeds} seeds per bar/point · error bars = 95% confidence interval"


def fig_e1(root, out, tag):
    df = pd.read_csv(os.path.join(root, "E1.csv"))
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    bar_panel(axes[0], df, "collisions", ["highway", "merge"], "Collisions per episode")
    axes[0].set_title("Safety", loc="left", color=INK, fontsize=11)
    bar_panel(axes[1], df, "success_rate", ["highway", "merge"], "Cars reaching the end safely (share)")
    axes[1].set_title("Success", loc="left", color=INK, fontsize=11)
    bar_panel(axes[2], df, "merge_time", ["merge"], "Time for ramp cars to merge (s)", "{:.1f}")
    axes[2].set_title("Merge efficiency", loc="left", color=INK, fontsize=11)
    finish(fig, "Talking cars avoid lane-change crashes and merge faster",
           subtitle_for(tag, df), os.path.join(out, "fig_e1_comparison.png"), legend_ax=axes[0])


def line_panel(ax, df, x, metric, color, ylabel, xlabel, fmt="{:.0f}"):
    g = df.groupby(x)[metric]
    m, e = g.mean(), g.apply(ci)
    ax.errorbar(m.index, m.values, yerr=e.values, color=color, lw=2, marker="o", ms=6, capsize=3,
                ecolor=INK2, elinewidth=1)
    for xi, yi, ei in zip(m.index, m.values, e.values):
        ax.annotate(fmt.format(yi), (xi, yi + ei), xytext=(0, 5), textcoords="offset points", ha="center",
                    fontsize=8, color=INK2)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_ylim(bottom=0)


def fig_e2(root, out, tag):
    df = pd.read_csv(os.path.join(root, "E2.csv"))
    df["delivered_per_agent_s"] = df.msgs_delivered / df.n_agents / df.sim_time_s
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    line_panel(axes[0], df, "n_agents", "delivered_per_agent_s", OURS, "Messages received per car per second",
               "Number of LLM cars", "{:.1f}")
    line_panel(axes[1], df, "n_agents", "bytes_per_agent_s", OURS, "Radio load per car (bytes/s)", "Number of LLM cars")
    line_panel(axes[2], df, "n_agents", "success_rate", OURS, "Cars reaching the end safely (share)",
               "Number of LLM cars", "{:.2f}")
    axes[2].set_ylim(0, 1.1)
    for ax in axes:
        ax.set_xticks([2, 4, 6, 8])
    finish(fig, "Radio load per car grows with fleet size, while safety holds",
           subtitle_for(tag, df) + " · highway scenario", os.path.join(out, "fig_e2_scalability.png"))


def fig_e3(root, out, tag):
    df = pd.read_csv(os.path.join(root, "E3.csv"))
    df["loss_pct"] = (df.cfg_loss_prob * 100).round().astype(int)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    line_panel(axes[0], df[df.scenario == "highway"], "loss_pct", "collisions", OURS,
               "Collisions per episode (highway)", "Messages lost (%)", "{:.2f}")
    line_panel(axes[1], df[df.scenario == "merge"], "loss_pct", "merge_time", OURS,
               "Time for ramp cars to merge (s)", "Messages lost (%)", "{:.1f}")
    for ax in axes:
        ax.set_xticks([0, 20, 40, 60])
    finish(fig, "Lost messages slowly erode the benefit of talking - no sudden failure",
           subtitle_for(tag, df), os.path.join(out, "fig_e3_packet_loss.png"))


def fig_e4(root, out, tag):
    df = pd.read_csv(os.path.join(root, "E4.csv"))
    variants = [((True, True), "Full system", OURS), ((False, True), "Without safety shield", NOCOMM),
                ((True, False), "Without fleet memory", RULE)]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for ax, metric, ylabel in ((axes[0], "collisions", "Collisions per episode"),
                               (axes[1], "success_rate", "Cars reaching the end safely (share)")):
        xs = np.arange(2)
        for k, ((sh, mem), label, color) in enumerate(variants):
            means, errs = [], []
            for sc in ("highway", "merge"):
                d = df[(df.scenario == sc) & (df.cfg_shield == sh) & (df.cfg_memory == mem)][metric]
                means.append(d.mean())
                errs.append(ci(d))
            pos = xs + (k - 1) * 0.26
            ax.bar(pos, means, 0.23, yerr=errs, color=color, label=label, capsize=3,
                   error_kw={"elinewidth": 1, "ecolor": INK2})
            for p, m, e in zip(pos, means, errs):
                ax.annotate(f"{m:.2f}", (p, m + e), xytext=(0, 3), textcoords="offset points", ha="center",
                            fontsize=8, color=INK2)
        ax.set_xticks(xs, ["Highway", "Merge"])
        ax.set_ylabel(ylabel)
        ax.set_ylim(bottom=0)
    axes[1].set_ylim(0, 1.2)
    finish(fig, "The safety shield is what keeps imperfect decisions from becoming crashes",
           subtitle_for(tag, df), os.path.join(out, "fig_e4_ablation.png"), legend_ax=axes[0])


def fig_e5(root_models, out):
    path = os.path.join(root_models, "E5.csv")
    if not os.path.exists(path):
        return
    df = pd.read_csv(path)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    models = list(df.model.unique())
    colors = [OURS, NOCOMM, RULE][:len(models)]
    for ax, metric, ylabel in ((axes[0], "llm_latency_mean", "Seconds per decision"),
                               (axes[1], "success_rate", "Cars reaching the end safely (share)")):
        xs = np.arange(2)
        for k, (m, c) in enumerate(zip(models, colors)):
            vals = [df[(df.model == m) & (df.scenario == sc)][metric] for sc in ("highway", "merge")]
            pos = xs + (k - (len(models) - 1) / 2) * 0.3
            ax.bar(pos, [v.mean() for v in vals], 0.27, yerr=[ci(v) for v in vals], color=c, label=m,
                   capsize=3, error_kw={"elinewidth": 1, "ecolor": INK2})
        ax.set_xticks(xs, ["Highway", "Merge"])
        ax.set_ylabel(ylabel)
    finish(fig, "Bigger local model: slower decisions - compare against the success it buys",
           f"Local LLMs via Ollama · {df.seed.nunique()} seeds per bar · error bars = 95% CI",
           os.path.join(out, "fig_e5_models.png"), legend_ax=axes[0])


# ----------------------------------------------------------------- diagrams
def _box(ax, x, y, w, h, title, lines=(), accent=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.12",
                                fc="#e8f1fc" if accent else SURFACE, ec=OURS if accent else "#a9a8a3",
                                lw=2 if accent else 1.2))
    ax.text(x + 0.15, y + h - 0.32, title, fontsize=10.5, fontweight="bold", color=INK, va="top")
    for i, line in enumerate(lines):
        ax.text(x + 0.15, y + h - 0.72 - 0.34 * i, line, fontsize=8.8, color=INK2, va="top")


def _arrow(ax, p, q, label="", both=False, lx=0.1):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="<|-|>" if both else "-|>", mutation_scale=12,
                                 color="#7d7c78", lw=1.2))
    if label:
        ax.text((p[0] + q[0]) / 2 + lx, (p[1] + q[1]) / 2, label, fontsize=8.5, color=INK2, va="center")


def fig_architecture(out):
    fig, ax = plt.subplots(figsize=(12, 8.2))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 8.2)
    ax.axis("off")
    ax.text(0.1, 8.15, "Each car decides on its own; cars share only intentions, warnings and lessons",
            fontsize=13, fontweight="bold", color=INK, va="top")
    _box(ax, 0.1, 6.5, 11.8, 1.0, "Simulated road world (Python, IDM physics)",
         ["Moves all cars every 0.25 s, detects crashes. Gives each car ONLY its local sensor view (60 m)."])
    ax.add_patch(FancyBboxPatch((0.1, 3.0), 11.8, 2.95, boxstyle="round,pad=0,rounding_size=0.12",
                                fc="#f4f3f0", ec="#d0cfca", lw=1))
    ax.text(0.25, 5.8, "Distributed agent layer: one independent worker per car", fontsize=10.5,
            fontweight="bold", color=INK, va="top")
    loop = ["1 Perceive (sensors + inbox)", "2 Decide: LLM picks action", "3 Safety shield check",
            "4 Share intent, negotiate", "5 Act, broadcast, reflect"]
    other = {2: ["same 5-step loop", "own LLM call", "own memory of", "what it heard"],
             3: ["any number of cars", "(2 to 8 tested)"]}
    for i, name in enumerate(["Car A0", "Car A1", "Car A2", "Car A3"]):
        _box(ax, 0.3 + i * 2.95, 3.15, 2.75, 2.3, name, loop if i < 2 else other[i])
    _box(ax, 0.1, 1.05, 3.8, 1.35, "V2V radio (message bus)",
         ["Range 150 m, delay, packet loss", "INTENT · MERGE_REQUEST · HAZARD"], accent=True)
    _box(ax, 4.1, 1.05, 3.8, 1.35, "Local LLM (Ollama)", ["llama3.2:3b default, llama3.1:8b", "JSON answer: action + reason"])
    _box(ax, 8.1, 1.05, 3.8, 1.35, "Fleet memory (lessons.json)", ["Cars write 1-line lessons after", "a drive; others read them"])
    _arrow(ax, (4.0, 6.5), (4.0, 5.97), "local observation")
    _arrow(ax, (8.0, 5.97), (8.0, 6.5), "chosen action")
    _arrow(ax, (2.0, 3.0), (2.0, 2.42), "messages", both=True)
    _arrow(ax, (6.0, 3.0), (6.0, 2.42), "prompt / answer", both=True)
    _arrow(ax, (10.0, 3.0), (10.0, 2.42), "lessons", both=True)
    _box(ax, 0.1, 0.0, 11.8, 0.8, "Distributed analytics: every node writes its own log  ->  merge  ->  metrics  ->  charts + dashboard", [])
    _arrow(ax, (2.0, 1.05), (2.0, 0.82))
    fig.savefig(os.path.join(out, "architecture.png"), bbox_inches="tight")
    plt.close(fig)
    print("wrote", os.path.join(out, "architecture.png"))


def fig_decision_loop(out):
    fig, ax = plt.subplots(figsize=(12, 2.9))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 2.9)
    ax.axis("off")
    ax.text(0.05, 2.85, "One decision cycle per car, every simulated second", fontsize=13,
            fontweight="bold", color=INK, va="top")
    steps = [("1. Perceive", ["own sensors (60 m)", "+ radio inbox"]),
             ("2. Decide", ["LLM reads a text", "scene, picks action"]),
             ("3. Shield", ["veto if a gap or", "collision time is unsafe"]),
             ("4. Negotiate", ["share intent; if two", "cars want one gap,", "priority decides"]),
             ("5. Act", ["apply the action,", "send radio messages"])]
    w, gap = 2.1, 0.32
    for i, (t, lines) in enumerate(steps):
        x = 0.05 + i * (w + gap)
        _box(ax, x, 0.35, w, 1.75, t, lines, accent=(i == 1))
        if i < len(steps) - 1:
            _arrow(ax, (x + w + 0.02, 1.2), (x + w + gap - 0.02, 1.2))
    ax.text(0.05, 0.08, "After the drive: 6. Reflect - the car writes a one-line lesson into fleet memory.",
            fontsize=9, color=INK2)
    fig.savefig(os.path.join(out, "decision_loop.png"), bbox_inches="tight")
    plt.close(fig)
    print("wrote", os.path.join(out, "decision_loop.png"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="mock")
    ap.add_argument("--results", default="results")
    a = ap.parse_args()
    root = os.path.join(a.results, a.tag)
    base = os.path.join("docs", "figures")
    out = os.path.join(base, a.tag)
    os.makedirs(out, exist_ok=True)
    fig_architecture(base)
    fig_decision_loop(base)
    for name, fn in (("E1", fig_e1), ("E2", fig_e2), ("E3", fig_e3), ("E4", fig_e4)):
        if os.path.exists(os.path.join(root, f"{name}.csv")):
            fn(root, out, a.tag)
    fig_e5(os.path.join(a.results, "ollama_models"), base)


if __name__ == "__main__":
    main()
