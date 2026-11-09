"""Build a one-file interactive analytics dashboard (open it in any browser).

  python make_dashboard.py               # uses results/mock and results/demo
  python make_dashboard.py --tag ollama_llama3.2-3b
Output: results/<tag>/dashboard.html
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
from fleet.analytics.metrics import load_events  # noqa: E402

COLORS = {"rule": "#1baf7a", "llm_nocomm": "#eb6834", "llm_comm": "#2a78d6"}
NAMES = {"rule": "Rule-based (no LLM)", "llm_nocomm": "LLM, no V2V", "llm_comm": "LLM + V2V (ours)"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="mock")
    a = ap.parse_args()
    root = os.path.join("results", a.tag)
    e1 = pd.read_csv(os.path.join(root, "E1.csv"))

    fig = make_subplots(rows=3, cols=2, vertical_spacing=0.12, horizontal_spacing=0.1, subplot_titles=(
        "Collisions per episode", "Ramp merge time (s)",
        "Mean speed of agent cars (m/s)", "Safety-shield vetoes per episode",
        "One episode: messages on the radio over time", "One episode: decisions taken, by action"))
    for sys_ in ("rule", "llm_nocomm", "llm_comm"):
        d = e1[e1.system == sys_]
        g = d.groupby("scenario")
        for (r, c, metric, scen) in ((1, 1, "collisions", ["highway", "merge"]), (1, 2, "merge_time", ["merge"]),
                                     (2, 1, "mean_speed", ["highway", "merge"]), (2, 2, "vetoes", ["highway", "merge"])):
            m = g[metric].mean().reindex(scen)
            fig.add_bar(x=scen, y=m.values, name=NAMES[sys_], marker_color=COLORS[sys_], row=r, col=c,
                        showlegend=(r, c) == (1, 1), legendgroup=sys_,
                        hovertemplate=f"{NAMES[sys_]}<br>%{{x}}: %{{y:.2f}}<extra></extra>")

    demo = sorted(glob.glob(os.path.join("results", "demo", "demo_merge_llm_comm*")))
    if demo:
        ev = load_events(demo[0])
        ticks = ev[(ev.node == "bus") & (ev.event == "tick")].sort_values("step")
        for col, name, color in (("delivered", "delivered", "#2a78d6"), ("dropped", "lost", "#e34948")):
            fig.add_scatter(x=ticks.step * 0.25, y=ticks[col], mode="lines", name=f"messages {name} (cumulative)",
                            line={"color": color, "width": 2}, row=3, col=1)
        dec = ev[ev.event == "decision"]
        counts = dec.groupby(["node", "final"]).size().unstack(fill_value=0)
        palette = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#6250d6"]
        for k, action in enumerate(counts.columns):
            fig.add_bar(x=counts.index, y=counts[action], name=action, marker_color=palette[k % 6], row=3, col=2)
        fig.update_xaxes(title_text="simulated time (s)", row=3, col=1)
    fig.update_layout(barmode="group", height=1050, template="plotly_white", font={"family": "Arial", "size": 12},
                      title={"text": "Fleet analytics dashboard - distributed LLM driving<br><sup>Source: "
                                     f"results/{a.tag} (experiment E1) and results/demo (one merge episode). "
                                     "Hover any bar or line for values.</sup>"},
                      legend={"orientation": "h", "y": -0.06})
    fig.update_layout(barmode="group")
    out = os.path.join(root, "dashboard.html")
    fig.write_html(out, include_plotlyjs=True)
    print("wrote", out)


if __name__ == "__main__":
    main()
