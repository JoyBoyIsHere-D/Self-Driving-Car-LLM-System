"""Build docs/Report.docx (the single, complete explanation) and
docs/Presentation.pptx from the experiment results and figures.

  python make_report.py                          # numbers from results/mock
  python make_report.py --tag ollama_llama3.2-3b # numbers from the real-LLM run
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
from fleet.agents.vehicle_agent import VehicleAgent  # noqa: E402
from fleet.llm.prompts import SYSTEM_PROMPT, build_prompt  # noqa: E402
from fleet.reporting.docx_writer import Docx  # noqa: E402
from fleet.sim.scenarios import make_world  # noqa: E402

TITLE = "Distributed Automated Driving with LLMs"
SUBTITLE = "A network of self-driving cars that think with a local LLM and coordinate over V2V radio"
COURSE = "Distribution Analytics - Course Project"


# ---------------------------------------------------------------------------- numbers
class Results:
    def __init__(self, tag: str):
        self.tag = tag
        root = os.path.join("results", tag)
        self.e = {k: pd.read_csv(os.path.join(root, f"{k}.csv")) for k in ("E1", "E2", "E3", "E4")
                  if os.path.exists(os.path.join(root, f"{k}.csv"))}
        e5 = os.path.join("results", "ollama_models", "E5.csv")
        self.e5 = pd.read_csv(e5) if os.path.exists(e5) else None
        self.fig = os.path.join("docs", "figures", tag)
        self.mock = tag == "mock"
        e1 = self.e["E1"]
        self.seeds = int(e1.seed.nunique())
        self.who = ("the built-in stand-in decision model (a rule-based imitation of an LLM, used because Ollama "
                    "was not installed when these results were produced)") if self.mock else f"the local LLM {e1.model.iloc[0]}"

    def m(self, exp, metric, **where) -> float:
        d = self.e[exp]
        for k, v in where.items():
            d = d[d[k] == v]
        return float(d[metric].mean())


def f2(x):
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.2f}"


def f1(x):
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.1f}"


def example_prompt() -> str:
    w = make_world("merge", 4, 10, 0.25, 0)
    agent = VehicleAgent("A0", "llm_comm", None, True, None, 60.0, "merge", 18.0)
    obs = agent.perceive(w, [])
    obs["merge_requests"] = []
    obs["intents"] = [{"from": "A2", "action": "YIELD", "lane": 1, "dx": -8.0}]
    return build_prompt(obs)


def decision_log_excerpt(n=8) -> list[list[str]]:
    runs = sorted(glob.glob(os.path.join("results", "demo", "demo_merge_llm_comm*")))
    if not runs:
        return []
    rows = []
    for path in sorted(glob.glob(os.path.join(runs[0], "nodes", "A*.jsonl"))):
        for line in open(path, encoding="utf-8"):
            e = json.loads(line)
            if e["event"] == "decision" and (e["proposed"] != "IDLE" or e["vetoed"]):
                rows.append(e)
    rows.sort(key=lambda e: (e["t"], e["node"]))
    out = [["Time", "Car", "Lane", "Proposed", "Applied", "Reason / shield note"]]
    for e in rows[:n]:
        note = e["reason"] + (f"  [VETO: {e['veto_reason']}]" if e["vetoed"] else "")
        out.append([f"{e['t']:.0f} s", e["node"], str(e["lane"]), e["proposed"], e["final"], note])
    return out


# ---------------------------------------------------------------------------- report
def build_report(R: Results, path: str) -> None:
    d = Docx()
    E1 = "E1"
    hw = dict(scenario="highway")
    mg = dict(scenario="merge")
    col = {s: R.m(E1, "collisions", system=s, **hw) for s in ("rule", "llm_nocomm", "llm_comm")}
    mt = {s: R.m(E1, "merge_time", system=s, **mg) for s in ("rule", "llm_nocomm", "llm_comm")}
    suc = {(s, sc): R.m(E1, "success_rate", system=s, scenario=sc) for s in ("rule", "llm_nocomm", "llm_comm")
           for sc in ("highway", "merge")}
    conf = R.m(E1, "conflicts", system="llm_comm", **hw)
    msgs = R.m(E1, "msgs_sent", system="llm_comm", **hw)
    bps = R.m(E1, "bytes_per_agent_s", system="llm_comm", **hw)
    lat = R.m(E1, "llm_latency_mean", system="llm_comm", **hw)

    # ---- title page
    d.title(TITLE, SUBTITLE)
    d.para(COURSE, size=13, color="333333")
    d.para("Prepared by: ______________________      Roll no.: ____________", size=12, color="333333")
    d.para("November 2026", size=12, color="333333")
    d.para("")
    d.callout("This one document explains everything: the idea, every part of the system, how it was built and "
              "tested, the results, how to run it, and how to present it. Sections 2-4 explain the concepts in plain "
              "language; Sections 5-8 describe the design; Sections 9-10 show the results; Sections 11-13 are "
              "practical (running the code, the code map, likely viva questions).", "How to read this report.")
    d.page_break()
    d.toc()
    d.page_break()

    # ---- 1 summary
    d.heading("1. Summary")
    d.para(f"We built a **distributed automated driving system**: a small fleet of simulated self-driving cars in "
           f"which **every car is its own decision-maker**. Each car looks only at its own surroundings, asks a "
           f"**local Large Language Model (LLM)** which manoeuvre to make next, checks the answer with a **safety "
           f"shield**, and tells nearby cars what it plans to do over a simulated **vehicle-to-vehicle (V2V) "
           f"radio**. When two cars want the same gap, they settle it among themselves with a simple shared "
           f"priority rule. There is **no central controller**. After each drive, cars write short lessons into a "
           f"shared fleet memory. Every car writes its own log file, and an analytics pipeline merges these logs "
           f"into metrics, charts and a dashboard.")
    d.para(f"We compared three ways of driving on two traffic scenarios (a highway with an overtaking conflict and a "
           f"broken-down car, and a congested on-ramp merge), with {R.seeds} random seeds each. Decisions in the "
           f"results below were made by {R.who}.")
    d.bullets([
        f"**Safety:** on the highway, cars that talk had **{f2(col['llm_comm'])} collisions per episode**, against "
        f"{f2(col['llm_nocomm'])} for the same decision-maker without the radio and {f2(col['rule'])} for a classic "
        f"rule-based driver. The talking cars detected and resolved {f1(conf)} lane-change conflicts per episode.",
        f"**Efficiency:** ramp cars merged in **{f1(mt['llm_comm'])} s** with V2V, against {f1(mt['llm_nocomm'])} s "
        f"without it and {f1(mt['rule'])} s for the rule-based driver, because main-lane cars made room when asked.",
        f"**Cost of talking:** about {bps:.0f} bytes per car per second ({msgs:.0f} messages per episode with 4 cars), "
        f"which is tiny for modern V2V radios.",
        "**Robustness:** losing messages slowly reduces the benefit; it does not cause a sudden failure, because the "
        "safety shield still checks every action.",
    ])
    if R.mock:
        d.callout("These results come from the stand-in decision model, not from an LLM. They prove that the "
                  "distributed machinery (radio, negotiation, shield, logs, analytics) works. To produce the LLM "
                  "results, install Ollama and run the commands in Section 11; the report and slides then rebuild "
                  "themselves with the new numbers.", "Important.")

    # ---- 2 problem
    d.heading("2. The problem in simple words")
    d.para("A self-driving car normally drives alone: it only knows what its own cameras and sensors see. That causes "
           "three problems that people solve every day by communicating:")
    d.bullets([
        "**It cannot see far.** A broken-down car 500 m ahead is invisible until the car is close and must brake hard.",
        "**It cannot know what others plan.** Two cars may decide to move into the same empty space at the same moment.",
        "**It cannot ask for help.** A car on an on-ramp in heavy traffic may never find a gap unless someone lets it in.",
    ])
    d.para("One obvious fix is a central computer that controls all cars. That is a **single point of failure** "
           "(if it crashes, every car is affected) and a **bottleneck** (it must think for everyone). Our goal is the "
           "opposite: cars that **think for themselves** but **share just enough information** to cooperate. That is "
           "what makes the system **distributed**.")
    d.para("We use an **LLM** as each car's tactical brain because it can read a situation written in plain words "
           "(\"a car on the ramp next to you asks to merge\"), follow written traffic rules, explain its choice in a "
           "sentence, and use advice written by other cars. We run the LLM **locally** (on the car's own computer, "
           "here via Ollama) because a real car cannot depend on an internet connection to make driving decisions.")

    # ---- 3 key ideas
    d.heading("3. Key ideas explained")
    d.table([
        ["Term", "Simple explanation", "Where it is in the project"],
        ["Agent", "A program that observes, decides and acts on its own. Here: one car.", "agents/vehicle_agent.py"],
        ["Distributed system", "Many independent computers working together with no single boss; they only exchange messages.", "Every car decides separately; runner.py runs them in parallel"],
        ["LLM", "Large Language Model: an AI that reads and writes text. We give it a text description of the road and it answers with an action.", "llm/client.py, llm/prompts.py"],
        ["Local LLM / Ollama", "Ollama runs open LLMs (Llama 3.2 3B, Llama 3.1 8B) on your own PC/GPU - no internet, no cost.", "OllamaLLM class"],
        ["Stand-in model", "A fast rule-based imitation of the LLM with the same input/output, used for testing and big experiment sweeps.", "MockLLM class"],
        ["V2V", "Vehicle-to-vehicle radio. Cars broadcast short messages to cars within range (150 m here).", "comms/bus.py"],
        ["Latency / packet loss", "Delay before a message arrives / chance a message never arrives. Real radios have both.", "MessageBus(latency_steps, loss_prob)"],
        ["Intent", "A short \"I am about to do X\" message each car sends every second.", "INTENT messages"],
        ["Negotiation / consensus", "How cars agree who goes first when two want the same gap - without asking a boss.", "coordination/negotiation.py"],
        ["Safety shield", "Simple, strict rules that can veto an unsafe LLM decision (e.g. a gap that is too short).", "safety/shield.py"],
        ["IDM", "Intelligent Driver Model: a standard formula that sets a car's acceleration to keep a safe distance.", "sim/world.py idm_acc()"],
        ["TTC", "Time-to-collision: seconds until two cars would touch if nothing changes. Small TTC = dangerous.", "shield and metrics"],
        ["Fleet memory", "Short lessons cars write after a drive (\"yield early for ramp cars\") that other cars read next time.", "memory/lessons.py"],
        ["Telemetry", "Every car writes its own event log; the analytics step collects and merges them.", "analytics/telemetry.py, metrics.py"],
    ], widths=[1.3, 3.3, 1.9], caption="Glossary of the ideas used in this project")

    # ---- 4 references
    d.heading("4. How this project relates to the reference projects")
    d.para("Two earlier projects were used as references for scope and deliverables: **(1) distributed path planning "
           "for multi-UAV (drone) systems using an LLM**, and **(2) a distributed automated driving system for "
           "self-driving cars using Flower** (a federated-learning framework). We did not copy either; Table 2 shows "
           "what we borrowed and what we did differently.")
    d.table([
        ["Aspect", "Ref 1: Multi-UAV + LLM", "Ref 2: Self-driving + Flower", "This project"],
        ["What is distributed", "Execution: each drone flies its own path", "Training: each car trains a model locally; a server averages the weights", "Decisions, communication, negotiation and memory"],
        ["Role of AI", "LLM turns a mission into waypoints/tasks", "Neural network driving policy", "A local LLM per car picks tactical actions and explains them"],
        ["Coordination", "Often one central LLM planner", "A central aggregation server", "Peer-to-peer intents + a shared priority rule; no server"],
        ["What is shared", "Plans / waypoints", "Model weights (never raw data)", "Intents, warnings, merge requests, text lessons"],
        ["Safety", "Collision-avoidance layer under the LLM", "Learned policy", "Rule-based shield under every LLM action"],
    ], widths=[1.25, 1.75, 1.75, 1.75], caption="Reference projects compared with this project")
    d.para("**Borrowed:** an AI planner sitting above a safe low-level controller (from the UAV project); learning "
           "across many vehicles without collecting their data centrally (from the Flower project - here as shared "
           "text lessons instead of model weights); and the deliverable pattern of architecture, simulation, "
           "baselines, experiments, analytics and presentation.")

    # ---- 5 design
    d.page_break()
    d.heading("5. System design")
    d.para("The system has four layers (Figure 1). The **road world** moves cars and detects crashes. The **agent "
           "layer** holds one independent agent per self-driving car. The **shared services** are the V2V radio, the "
           "local LLM server and the fleet memory. The **analytics layer** turns every car's logs into results.")
    d.image("docs/figures/architecture.png", 6.4, "System architecture: four layers, one independent agent per car")
    d.table([
        ["Component", "File", "What it does"],
        ["Road world", "sim/world.py", "Cars, lanes, IDM physics (0.25 s steps), lane changes, crash detection, finish line"],
        ["Scenarios", "sim/scenarios.py", "Builds the highway and merge situations from a random seed"],
        ["Vehicle agent", "agents/vehicle_agent.py", "One car's brain: perceive, decide, shield, negotiate, act, reflect"],
        ["LLM client", "llm/client.py", "Talks to Ollama (JSON answer); retry once; fall back to IDLE on timeout or bad output"],
        ["Prompt builder", "llm/prompts.py", "Turns the car's observation into a short text scene + the system prompt with rules"],
        ["V2V radio", "comms/bus.py", "Delivers messages within 150 m, after a delay, with optional random loss"],
        ["Negotiation", "coordination/negotiation.py", "Detects two cars aiming at the same gap and applies the priority rule"],
        ["Safety shield", "safety/shield.py", "Vetoes lane changes with short gaps or fast-closing cars; blocks impossible moves"],
        ["Fleet memory", "memory/lessons.py", "Stores and returns short lessons per scenario (JSON file)"],
        ["Runner", "runner.py", "Clock: delivers messages, asks agents to decide in parallel, applies actions, logs"],
        ["Analytics", "analytics/*.py, make_*.py", "Per-car logs -> merged table -> metrics -> CSV, charts, dashboard, report"],
    ], widths=[1.3, 1.9, 3.3], caption="Components of the system")

    d.heading("5.1 One decision, step by step", 2)
    d.image("docs/figures/decision_loop.png", 6.4, "The decision cycle every car runs once per simulated second")
    d.bullets([
        "**Perceive.** The car reads its own sensors (cars within 60 m in its lane and the lanes beside it) and its radio inbox (warnings, merge requests, neighbours' plans).",
        "**Decide.** The observation is written as a short text and sent to the LLM with the driving rules. The LLM replies with JSON such as `{\"action\": \"YIELD\", \"reason\": \"let A0 merge in front\"}`.",
        "**Shield.** The safety shield checks the proposed action against hard limits (gaps of at least 6 m or 0.3 s x speed, time-to-collision of at least 2.5 s). An unsafe action is replaced by a safe one and the veto is logged.",
        "**Negotiate.** The car broadcasts its intent. If two cars aim at the same lane within 15 m of each other, both apply the same priority rule and the lower-priority car waits.",
        "**Act.** The action is applied (speed target or lane change), and radio messages are sent: MERGE_REQUEST from ramp cars, HAZARD when a stopped car is seen.",
        "**Reflect** (after the drive). If something went wrong - a veto, a conflict, a slow merge - the car writes a one-line lesson into the fleet memory.",
    ], numbered=True)
    d.heading("5.2 What the LLM actually sees", 2)
    d.para("This is a real prompt produced by the code for the ramp car A0 at the start of the merge scenario "
           "(the system prompt with the rules is sent alongside it - see Section 12):")
    d.code(example_prompt())
    d.para("The LLM must answer with one JSON object. If the answer is not valid JSON or names an unknown action, the "
           "client asks once more and otherwise falls back to IDLE (keep lane and speed), so a confused LLM can never "
           "freeze the car.")
    log = decision_log_excerpt()
    if log:
        d.heading("5.3 A short extract of real decisions", 2)
        d.para(f"Decisions logged during one merge episode (decisions by {'the stand-in model' if R.mock else 'the LLM'}). "
               "Notice the shield catching unsafe merges and main-lane cars yielding to ramp cars.")
        d.table(log, widths=[0.55, 0.45, 0.45, 1.0, 0.95, 3.1], caption="Extract of the decision log (results/demo)")

    # ---- 6 distributed parts
    d.page_break()
    d.heading("6. The distributed parts in detail")
    d.heading("6.1 Independent agents running in parallel", 2)
    d.para("Each car is an object with its own state (target speed, hazards it has heard about, its statistics) and "
           "its own LLM calls. The runner submits all cars' decisions to a thread pool at the same time, so the cars "
           "really think in parallel, and the LLM server receives several requests at once - just as each car would "
           "use its own on-board computer. The runner never decides anything for a car: it only advances the clock, "
           "hands each car its own observation and inbox, and applies the actions the cars chose. A car never reads "
           "another car's memory; information moves only through radio messages.")
    d.heading("6.2 The V2V radio", 2)
    d.table([
        ["Message", "Sent by", "When", "Content", "Used for"],
        ["INTENT", "every car", "every second", "planned action, target lane, position, priority", "detecting and resolving conflicts"],
        ["MERGE_REQUEST", "ramp cars", "every second while on the ramp", "position, metres left on the ramp", "main-lane cars can yield or move left"],
        ["HAZARD", "a car that sees a stopped car", "once per hazard; relayed once", "lane and position of the hazard", "cars far behind leave that lane early"],
    ], widths=[1.15, 1.1, 1.3, 1.6, 1.35], caption="Radio messages")
    d.para("The radio has three realistic limits we can tune: **range** (150 m), **latency** (1 step = 0.25 s) and "
           "**packet loss** (0-60 % in experiment E3). A HAZARD warning is relayed once by the cars that hear it "
           "(multi-hop), so it can travel further than one radio range.")
    d.heading("6.3 Decentralised negotiation", 2)
    d.para("Each car's intent carries a priority made of three parts, compared in order:")
    d.bullets(["**Urgency:** 3 = on the ramp and it ends soon; 2 = escaping a hazard in its own lane; 1 = ordinary lane change.",
               "**Closeness to the ramp end** (closer goes first).",
               "**Car number** (lower number wins) - a final tie-breaker so there is always exactly one winner."])
    d.para("Because every car runs **the same deterministic rule on the same messages**, all cars reach the same "
           "answer without a coordinator - a simple form of distributed consensus. Example: A0 (lane 0) and A1 "
           "(lane 2) both plan to move into lane 1 at x = 140 m. Both have urgency 1, so the car number decides: A0 "
           "goes, A1 waits one second. If a message is lost, the cars may disagree; that is exactly what experiment "
           "E3 measures, and the safety shield is the last line of defence.")
    d.heading("6.4 Safety shield", 2)
    d.para("The LLM is slow and can be wrong, so it only *proposes*. The shield vetoes a lane change if the gap in "
           "front or behind in the target lane is shorter than max(6 m, 0.3 s x speed), if the time-to-collision with "
           "either car is under 2.5 s, if the lane does not exist or ends within 10 m, or if the target is the on-ramp. "
           "It also blocks speeding up when the car ahead is less than 4 s away. The shield uses only the car's own "
           "sensors, never other cars' plans - that is negotiation's job.")
    d.heading("6.5 Fleet memory", 2)
    d.para("After each drive, a car that had trouble asks its LLM for one short lesson (for example \"Check the rear "
           "gap before changing lanes; fast cars behind close the gap quickly\"). Lessons are stored per scenario in "
           "a JSON file (duplicates removed, latest 50 kept) and the three newest are added to every car's prompt "
           "next time. This is our lightweight alternative to federated learning: knowledge is shared as readable "
           "text instead of model weights. (The stand-in model ignores lessons, so their effect can only be measured "
           "with the real LLM.)")
    d.heading("6.6 Distributed logging and analytics", 2)
    d.para("Every node writes **its own** log file - results/.../nodes/A0.jsonl, A1.jsonl, bus.jsonl, world.jsonl - "
           "the same pattern real distributed systems use (no shared database to become a bottleneck). The analytics "
           "step then collects all files, merges them into one table with pandas, computes the metrics in Table 6 for "
           "each episode, and aggregates them over seeds (mean and 95 % confidence interval). Outputs: CSV files, the "
           "charts in this report, and an interactive dashboard (results/<tag>/dashboard.html).")

    # ---- 7 simulation
    d.heading("7. Simulation and scenarios")
    d.para("We wrote a small traffic simulator in Python instead of using a heavy simulator such as CARLA, so the "
           "project runs on a normal laptop and every line is explainable. Cars move along lanes; their speed follows "
           "the IDM formula (keep a safe distance, approach a desired speed). Background \"human\" cars use IDM and a "
           "simple overtaking rule; they never send messages. Lane changes are instant, so two cars moving into the "
           "same spot at the same moment really collide.")
    d.image("docs/figures/scenarios.png", 6.5, "The two scenarios at the start of an episode (blue = LLM agents)")
    d.table([
        ["Setting", "Highway", "Merge"],
        ["Lanes", "3", "2 main lanes + an on-ramp that ends at 260 m"],
        ["Road length", "900 m", "500 m"],
        ["Self-driving (agent) cars", "4 (2-8 in E2), in pairs side by side", "2 on the ramp + 2 in the congested lane"],
        ["Human cars", "10: slow trucks side by side in the outer lanes, faster cars in the middle", "10: dense queue (15 m gaps, 11 m/s) + a few fast cars"],
        ["Challenge", "Two cars want the same middle-lane gap at once; a broken-down car at 560 m", "A merging car needs ~17 m but gaps are 15 m: someone must make room"],
        ["Time", "physics every 0.25 s, decisions every 1 s, at most 60 s", "same"],
    ], widths=[1.6, 2.45, 2.45], caption="Scenario settings")

    # ---- 8 experiments
    d.heading("8. Experiments and metrics")
    d.table([
        ["ID", "Question", "What changes", "Main metrics"],
        ["E1", "Does talking help?", "Rule-based vs LLM without V2V vs LLM with V2V (ours); both scenarios", "collisions, success, merge time"],
        ["E2", "Does it scale?", "2, 4, 6, 8 self-driving cars (highway)", "messages and bytes per car, success"],
        ["E3", "What if the radio is bad?", "0, 20, 40, 60 % of messages lost", "collisions, merge time"],
        ["E4", "Which parts matter?", "Safety shield off; fleet memory off", "collisions, success"],
        ["E5", "Does a bigger LLM help?", "llama3.2:3b vs llama3.1:8b (needs Ollama)", "decision time, success"],
    ], widths=[0.45, 1.5, 2.85, 1.7], caption="Experiments")
    d.table([
        ["Metric", "Meaning", "Why it matters"],
        ["Collisions per episode", "crashes involving a self-driving car", "the most important safety number"],
        ["Success rate", "share of self-driving cars that reach the end without crashing", "overall task completion"],
        ["Merge time", "seconds until a ramp car is in the main lane", "efficiency and cooperation"],
        ["Shield vetoes", "LLM proposals blocked as unsafe", "how often the LLM was wrong"],
        ["Conflicts / yields", "pairs of cars that wanted the same gap / cars that waited", "how much coordination was needed"],
        ["Messages, bytes per car per second", "radio traffic", "cost of communication, scalability"],
        ["Decision time (s)", "time for one LLM answer", "can the LLM keep up with real time?"],
        ["Min TTC, hard brakes", "closest call; decelerations stronger than 4 m/s^2", "comfort and near-misses"],
    ], widths=[1.8, 2.6, 2.1], caption="Metrics computed from the logs")
    d.para(f"Each setting was run with {R.seeds} different random seeds (slightly different starting positions and "
           "speeds). Charts show the mean; error bars show a 95 % confidence interval (1.96 x standard error).")

    # ---- 9 results
    d.page_break()
    d.heading("9. Results")
    if R.mock:
        d.callout("All numbers in this section were produced by the stand-in decision model. They show that the "
                  "distributed system behaves as designed; they are not a measurement of LLM driving skill. Re-run "
                  "with Ollama (Section 11) to replace them with LLM results.", "Read this first.")
    d.heading("9.1 E1 - Does talking help?", 2)
    d.image(os.path.join(R.fig, "fig_e1_comparison.png"), 6.5, "Main comparison of the three systems")
    d.bullets([
        f"**Highway safety.** Without the radio, two cars sometimes moved into the same middle-lane gap at the same moment "
        f"({f2(col['llm_nocomm'])} collisions per episode for the LLM-style driver, {f2(col['rule'])} for the rule-based "
        f"driver). With intents and the priority rule this dropped to {f2(col['llm_comm'])}: the conflict was seen "
        f"and one car waited ({f1(conf)} conflicts resolved per episode).",
        f"**Merging.** Ramp cars needed {f1(mt['rule'])} s (rule-based) and {f1(mt['llm_nocomm'])} s (no radio) to get in, "
        f"because gaps in the queue were too small. With MERGE_REQUEST messages, main-lane cars yielded and the "
        f"merge took {f1(mt['llm_comm'])} s.",
        f"**Success** (reaching the end without a crash): highway {f2(suc[('rule', 'highway')])} / "
        f"{f2(suc[('llm_nocomm', 'highway')])} / {f2(suc[('llm_comm', 'highway')])}, merge "
        f"{f2(suc[('rule', 'merge')])} / {f2(suc[('llm_nocomm', 'merge')])} / {f2(suc[('llm_comm', 'merge')])} "
        "(rule-based / no radio / ours).",
    ])
    if "E2" in R.e:
        e2 = R.e["E2"].copy()
        e2["rx"] = e2.msgs_delivered / e2.n_agents / e2.sim_time_s
        g = e2.groupby("n_agents")
        d.heading("9.2 E2 - Does it scale?", 2)
        d.image(os.path.join(R.fig, "fig_e2_scalability.png"), 6.5, "Scalability with the number of self-driving cars")
        d.para(f"With 2 cars each car received {g.rx.mean().iloc[0]:.1f} messages per second; with 8 cars, "
               f"{g.rx.mean().iloc[-1]:.1f}. The radio load per car grew from {g.bytes_per_agent_s.mean().iloc[0]:.0f} to "
               f"{g.bytes_per_agent_s.mean().iloc[-1]:.0f} bytes per second - roughly linear in the number of neighbours, "
               "because every car hears every neighbour in range. Even at 8 cars this is under 1 kB/s, far below what "
               f"V2V radios carry. Success stayed between {g.success_rate.mean().min():.2f} and "
               f"{g.success_rate.mean().max():.2f}. With a real LLM the limiting factor is decision time, not the radio: "
               "all cars call the LLM at the same moment.")
    if "E3" in R.e:
        e3 = R.e["E3"]
        h = e3[e3.scenario == "highway"].groupby("cfg_loss_prob").collisions.mean()
        mm = e3[e3.scenario == "merge"].groupby("cfg_loss_prob").merge_time.mean()
        d.heading("9.3 E3 - What if messages are lost?", 2)
        d.image(os.path.join(R.fig, "fig_e3_packet_loss.png"), 6.0, "Effect of radio packet loss")
        d.para(f"As loss rose from 0 % to 60 %, highway collisions rose from {h.iloc[0]:.2f} to {h.iloc[-1]:.2f} per "
               f"episode and merge time from {mm.iloc[0]:.1f} s to {mm.iloc[-1]:.1f} s. When an intent is lost, two cars "
               "can both believe they have the gap; when a merge request is lost, the main-lane car does not yield. "
               "The decline is gradual - the system degrades gracefully instead of failing.")
    if "E4" in R.e:
        e4 = R.e["E4"]
        ns = e4[e4.cfg_shield == False].groupby("scenario").collisions.mean()  # noqa: E712
        nss = e4[e4.cfg_shield == False].groupby("scenario").success_rate.mean()  # noqa: E712
        d.heading("9.4 E4 - Which parts matter? (ablation)", 2)
        d.image(os.path.join(R.fig, "fig_e4_ablation.png"), 6.0, "Removing the safety shield or the fleet memory")
        d.para(f"Without the safety shield, collisions rose to {ns.get('highway', np.nan):.2f} (highway) and "
               f"{ns.get('merge', np.nan):.2f} (merge) per episode, and success fell to "
               f"{nss.get('highway', np.nan):.2f} and {nss.get('merge', np.nan):.2f}. The decision-maker proposes "
               "lane changes with short rear gaps, and the shield is what stops those from becoming crashes. "
               + ("Turning the fleet memory off changed nothing here because the stand-in model does not read "
                  "lessons; this comparison is meaningful only with the real LLM." if R.mock else
                  "Turning the fleet memory off shows how much the shared lessons helped the LLM."))
    if R.e5 is not None:
        d.heading("9.5 E5 - Does a bigger LLM help?", 2)
        d.image(os.path.join("docs", "figures", "fig_e5_models.png"), 6.0, "Local model size: decision time and success")
        g5 = R.e5.groupby("model")[["llm_latency_mean", "success_rate", "collisions", "vetoes"]].mean()
        rows = [["Model", "Seconds per decision", "Success", "Collisions/episode", "Shield vetoes/episode"]]
        for mdl, r in g5.iterrows():
            rows.append([mdl, f"{r.llm_latency_mean:.2f}", f"{r.success_rate:.2f}", f"{r.collisions:.2f}", f"{r.vetoes:.1f}"])
        d.table(rows, widths=[1.6, 1.3, 1.0, 1.3, 1.3], caption="Local LLM comparison (E5)")
    else:
        d.heading("9.5 E5 - Does a bigger LLM help?", 2)
        d.para("Not run yet: it needs Ollama with llama3.2:3b and llama3.1:8b. Command: "
               "`python run_experiments.py --exp E5 --models llama3.2:3b llama3.1:8b --seeds 3`. Expectation: the 8B "
               "model follows the rules better (fewer shield vetoes) but is 2-4x slower per decision on a 4 GB GPU, "
               "because part of it runs on the CPU.")

    # ---- 10 discussion
    d.heading("10. Discussion, limitations and future work")
    d.heading("What the results mean", 2)
    d.bullets(["Sharing **intentions**, not just positions, is what prevents two cars from taking the same gap.",
               "**Asking for help** (merge requests) turns an impossible merge into an easy one - cooperation beats individual skill.",
               "An LLM should **propose, not command**: a simple rule-based shield makes an imperfect decision-maker safe.",
               "Communication is **cheap** (under 1 kB/s per car) and degrades **gracefully** when messages are lost."])
    d.heading("Limitations (honest list)", 2)
    d.bullets(["The road world is simplified: instant lane changes, no curves, no traffic lights, perfect sensors within 60 m.",
               "Cars run as parallel threads on one PC, not on separate machines; the design (no shared state, message-only communication) would allow that.",
               f"{'Results in this version come from the stand-in decision model; the LLM results must be generated with Ollama.' if R.mock else 'LLM decisions take ~0.5-2 s, so the simulation waits for them; a real car would need a faster model or asynchronous decisions.'}",
               "The priority rule is fixed; a real system would need security (signed messages) against cars that lie."])
    d.heading("Future work", 2)
    d.bullets(["Run cars as separate processes or machines (e.g. with Ray) and a real message broker (MQTT/Redis).",
               "Asynchronous decisions so the simulation never waits for the LLM.",
               "Federated fine-tuning of the LLM with Flower (a bridge to reference project 2).",
               "Intersections and roundabouts; a city-scale simulator such as SUMO."])

    # ---- 11 run
    d.page_break()
    d.heading("11. How to run the project")
    d.para("Requirements: Python 3.10+ with numpy, pandas, matplotlib, requests, plotly, python-pptx, pytest "
           "(all already installed on the development PC). Open a terminal in the project folder.")
    d.table([
        ["Step", "Command", "What you get"],
        ["1. Test", "python -m pytest", "8 tests pass"],
        ["2. Watch a drive", "python run_demo.py --scenario merge", "decision log in the terminal + docs/figures/demo_merge_llm_comm.gif"],
        ["3. Experiments", "python run_experiments.py", "results/mock/E1-E4.csv (stand-in, ~1 minute)"],
        ["4. Charts", "python make_figures.py", "docs/figures/mock/*.png"],
        ["5. Dashboard", "python make_dashboard.py", "results/mock/dashboard.html"],
        ["6. Report + slides", "python make_report.py", "docs/Report.docx, docs/Presentation.pptx"],
    ], widths=[1.2, 2.6, 2.7], caption="Running with the stand-in model")
    d.para("**With the real local LLM:**")
    d.bullets([
        "Install Ollama from https://ollama.com/download (or `winget install Ollama.Ollama`) and start it.",
        "Download the models: `ollama pull llama3.2:3b` and `ollama pull llama3.1:8b`.",
        "Demo: `python run_demo.py --scenario merge --backend ollama --model llama3.2:3b`",
        "Experiments (about 1-3 minutes per episode): `python run_experiments.py --backend ollama --model llama3.2:3b --seeds 3`",
        "Model comparison: `python run_experiments.py --exp E5 --models llama3.2:3b llama3.1:8b --seeds 3`",
        "Rebuild everything with the LLM numbers: `python make_figures.py --tag ollama_llama3.2-3b`, "
        "`python make_dashboard.py --tag ollama_llama3.2-3b`, `python make_report.py --tag ollama_llama3.2-3b`",
    ], numbered=True)
    d.para("Hardware note: on an RTX 3050 laptop GPU (4 GB), the 3B model fits on the GPU; the 8B model (~4.9 GB) "
           "runs partly on the CPU and is noticeably slower.")

    # ---- 12 code map
    d.heading("12. Code walkthrough")
    d.para("The project is about 1,200 lines of Python. The most important pieces to understand:")
    d.table([
        ["File", "Key function / class", "Read it to understand"],
        ["sim/world.py", "World.step(), idm_acc(), lane_change_is_safe()", "how cars move and crash"],
        ["agents/vehicle_agent.py", "VehicleAgent.perceive/decide/negotiate/act", "the life of one car"],
        ["llm/prompts.py", "SYSTEM_PROMPT, build_prompt()", "exactly what the LLM is told"],
        ["llm/client.py", "OllamaLLM.decide(), MockLLM.decide()", "LLM call, JSON parsing, fallback"],
        ["coordination/negotiation.py", "priority(), resolve()", "the decentralised consensus rule"],
        ["safety/shield.py", "check()", "the hard safety limits"],
        ["comms/bus.py", "MessageBus.publish/deliver/exchange_intents", "range, delay and loss of messages"],
        ["runner.py", "run_episode(), _decision_tick()", "how everything is wired together each second"],
        ["analytics/metrics.py", "episode_metrics()", "how logs become numbers"],
    ], widths=[1.9, 2.5, 2.1], caption="Where to look in the code")
    d.para("The system prompt given to the LLM:")
    d.code(SYSTEM_PROMPT)

    # ---- 13 viva
    d.heading("13. Presentation and viva preparation")
    qa = [
        ("Why is this system distributed?", "Every car decides on its own with only local information; cars exchange messages but there is no central controller, no shared memory between cars, and logs are written per node."),
        ("Why use an LLM instead of normal rules?", "An LLM can read the situation in words, follow written rules, explain its choice and use lessons from other cars without retraining. Rules are still used where they are best - the safety shield."),
        ("Why a local LLM?", "A car cannot depend on the internet to drive; local inference is private, free and works offline. Ollama makes this easy."),
        ("What if the LLM gives a wrong answer?", "Invalid output -> retry once, then IDLE. Unsafe action -> the shield vetoes it. Timeouts -> IDLE. Experiment E4 shows how important the shield is."),
        ("How do cars agree without a leader?", "They broadcast intents and all apply the same deterministic priority rule (urgency, closeness to ramp end, car number), so they reach the same decision independently."),
        ("What happens if messages are lost?", "Cars may disagree about who goes first; E3 shows performance degrades gradually, and the shield still prevents most crashes."),
        ("How is this different from the Flower project?", "Flower shares model weights in federated training through a server. We share intents, warnings and text lessons directly between cars at driving time."),
        ("How is it different from the UAV project?", "The UAV project usually plans centrally with an LLM for all drones; here each car's LLM plans only for itself."),
        ("What does 'distributed analytics' mean here?", "Each node logs its own events; logs are collected, merged and aggregated into metrics, charts and a dashboard - the same pattern as log shipping in real systems."),
        ("What would you do next?", "Separate processes/machines, asynchronous LLM calls, federated fine-tuning with Flower, intersections and a city-scale simulator."),
    ]
    for q, a in qa:
        d.para(f"**Q: {q}**", space_after=40)
        d.para(f"A: {a}")

    # ---- 14 conclusion
    d.heading("14. Conclusion")
    d.para("We built and tested a distributed automated driving system in which every car runs its own local LLM, "
           "checks it with a safety shield, and coordinates with neighbours over a V2V radio using a leaderless "
           "priority rule. The experiments show the value of each distributed piece: sharing intentions prevents "
           "conflicting lane changes, merge requests make cooperation possible, communication is cheap and degrades "
           "gracefully, and the shield keeps imperfect decisions safe. The whole pipeline - simulation, agents, "
           "radio, logs, analytics, figures, report and slides - is automated and reproducible from one folder.")
    d.heading("References", 2)
    d.bullets([
        "Hu, S. et al. AgentsCoDriver: LLM-empowered collaborative driving with lifelong learning. arXiv:2404.06345 (2024).",
        "Fang, S. et al. Towards Interactive and Learnable Cooperative Driving Automation: a Large Language Model-Driven Decision-Making Framework (CoDrivingLLM). arXiv:2409.12812 (2024).",
        "ConvoyLLM: Dynamic Multi-Lane Convoy Control Using LLMs. arXiv:2502.17529 (2025).",
        "LEVIOSA: natural-language based UAV trajectory generation with a distributed safety filter. github.com/sesem738/Leviosa.",
        "Beutel, D. et al. Flower: A Friendly Federated Learning Framework. github.com/adap/flower.",
        "Treiber, M., Hennecke, A., Helbing, D. Congested traffic states in empirical observations and microscopic simulations (IDM). Physical Review E 62 (2000).",
        "Kesting, A., Treiber, M., Helbing, D. General lane-changing model MOBIL for car-following models. Transportation Research Record (2007).",
        "Ollama - run large language models locally. ollama.com.",
    ])
    d.save(path, title=TITLE, author="")
    print("wrote", path)


# ---------------------------------------------------------------------------- slides
def build_slides(R: Results, path: str) -> None:
    from pptx import Presentation
    from pptx.dml.color import RGBColor
    from pptx.enum.shapes import MSO_SHAPE
    from pptx.util import Emu, Inches, Pt

    INK, INK2, ACC, SOFT = RGBColor(0x0B, 0x0B, 0x0B), RGBColor(0x52, 0x51, 0x4E), RGBColor(0x2A, 0x78, 0xD6), RGBColor(0xEE, 0xF4, 0xFC)
    prs = Presentation()
    prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)
    blank = prs.slide_layouts[6]
    E1 = "E1"
    col = {s: R.m(E1, "collisions", system=s, scenario="highway") for s in ("rule", "llm_nocomm", "llm_comm")}
    mt = {s: R.m(E1, "merge_time", system=s, scenario="merge") for s in ("rule", "llm_nocomm", "llm_comm")}
    n = [0]

    def text(slide, x, y, w, h, s, size=18, color=INK, bold=False, align=None):
        tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
        tf = tb.text_frame
        tf.word_wrap = True
        lines = s if isinstance(s, list) else [s]
        for i, line in enumerate(lines):
            p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
            bullet = line.startswith("- ")
            r = p.add_run()
            r.text = ("•  " + line[2:]) if bullet else line
            r.font.size, r.font.bold, r.font.color.rgb, r.font.name = Pt(size), bold, color, "Calibri"
            p.space_after = Pt(8)
            if align:
                p.alignment = align
        return tb

    def slide(title, notes="", kicker=None):
        s = prs.slides.add_slide(blank)
        n[0] += 1
        bar = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.5), Inches(0.45), Inches(0.09), Inches(0.62))
        bar.fill.solid()
        bar.fill.fore_color.rgb = ACC
        bar.line.fill.background()
        if kicker:
            text(s, 0.75, 0.2, 11, 0.35, kicker.upper(), 11, ACC, True)
        text(s, 0.75, 0.42, 12, 0.8, title, 28, INK, True)
        text(s, 12.2, 7.0, 0.9, 0.35, str(n[0]), 11, INK2)
        if R.mock and kicker == "Results":
            text(s, 0.75, 7.0, 10, 0.35, "Numbers from the built-in stand-in decision model (not an LLM) - rerun with Ollama for LLM results",
                 10, INK2)
        if notes:
            s.notes_slide.notes_text_frame.text = notes
        return s

    def picture(s, path, x, y, w=None, h=None):
        if os.path.exists(path):
            s.shapes.add_picture(path, Inches(x), Inches(y), width=Inches(w) if w else None,
                                 height=Inches(h) if h else None)

    def card(s, x, y, w, h, head, body, accent=False, size=17):
        box = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
        box.adjustments[0] = 0.06
        box.shadow.inherit = False
        box.fill.solid()
        box.fill.fore_color.rgb = SOFT if accent else RGBColor(0xFF, 0xFF, 0xFF)
        box.line.color.rgb = ACC if accent else RGBColor(0xC9, 0xC8, 0xC3)
        box.line.width = Emu(19050 if accent else 12700)
        text(s, x + 0.2, y + 0.12, w - 0.4, 0.5, head, 19, INK, True)
        text(s, x + 0.2, y + 0.68, w - 0.4, h - 0.8, body.split("\n"), size, INK2)

    def stat(s, x, y, w, big, small):
        text(s, x, y, w, 0.7, big, 30, ACC, True)
        text(s, x, y + 0.7, w, 0.8, small, 14, INK2)

    # 1 title
    s = prs.slides.add_slide(blank)
    n[0] += 1
    band = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, prs.slide_width, Inches(0.25))
    band.fill.solid()
    band.fill.fore_color.rgb = ACC
    band.line.fill.background()
    text(s, 0.8, 2.0, 11.5, 1.2, TITLE, 42, INK, True)
    text(s, 0.8, 3.3, 11.5, 1.0, SUBTITLE, 22, INK2)
    text(s, 0.8, 5.2, 11.5, 0.5, COURSE, 16, INK2)
    text(s, 0.8, 5.7, 11.5, 0.5, "Presented by: ____________________", 16, INK2)
    s.notes_slide.notes_text_frame.text = ("Introduce the project in one sentence: we built a fleet of simulated self-driving "
                                           "cars where each car has its own local LLM brain and the cars coordinate by radio, "
                                           "with no central controller.")

    # 2 problem
    s = slide("Self-driving cars that drive alone can't see far, can't read minds, can't ask for help",
              "Explain the three everyday problems. Then: a central controller would fix them but is a single point of "
              "failure and a bottleneck. We want cars that think for themselves but share just enough.", "The problem")
    card(s, 0.75, 1.7, 3.8, 2.1, "Can't see far", "Sensors reach ~60 m. A broken-down car 500 m ahead is a surprise.")
    card(s, 4.75, 1.7, 3.8, 2.1, "Can't read minds", "Two cars can pick the same empty gap at the same moment.")
    card(s, 8.75, 1.7, 3.8, 2.1, "Can't ask for help", "A car on a busy on-ramp may never find a gap alone.")
    text(s, 0.75, 4.3, 11.8, 1.6, ["- A central computer could control every car - but it is a single point of failure and a bottleneck.",
                                   "- Our answer: every car decides for itself, and cars share only intentions, warnings and lessons."], 18, INK)

    # 3 architecture
    s = slide("Our system: one LLM brain per car, a radio between them, no boss",
              "Walk top to bottom: the world gives each car only its local view; each car is an independent worker; "
              "they share the radio, a local LLM server (Ollama) and the fleet memory; every node logs for analytics.",
              "Architecture")
    picture(s, "docs/figures/architecture.png", 2.3, 1.35, h=5.6)

    # 4 decision loop
    s = slide("Every second, each car runs the same five steps",
              "Go through the steps. Emphasise: the LLM only proposes; the shield can veto; negotiation is between "
              "the cars themselves. After the drive the car writes a lesson.", "How a car decides")
    picture(s, "docs/figures/decision_loop.png", 0.75, 1.5, w=11.8)
    text(s, 0.75, 4.75, 11.8, 2.0, ['- The LLM (Ollama, llama3.2:3b) reads a short text scene and answers JSON: {"action": "YIELD", "reason": "let A0 merge in front"}',
                                    "- Bad or slow answers fall back to IDLE, so the car never freezes",
                                    "- Actions: FASTER, SLOWER, IDLE, LANE_LEFT, LANE_RIGHT, YIELD"], 16)

    # 5 talking
    s = slide("Cars talk with three short radio messages and agree with one shared rule",
              "Explain the messages, then the priority rule. Key point: because every car applies the same rule to "
              "the same messages, they all reach the same decision - consensus without a leader.", "Communication")
    card(s, 0.75, 1.6, 3.8, 2.0, "INTENT", "Every second: \"I plan LANE_LEFT into lane 1 at x = 140 m, priority ...\"")
    card(s, 4.75, 1.6, 3.8, 2.0, "MERGE_REQUEST", "Ramp cars: \"I need to get into lane 1, 80 m of ramp left\"")
    card(s, 8.75, 1.6, 3.8, 2.0, "HAZARD", "\"Stopped car in lane 1 at 560 m\" - relayed once so it travels further")
    card(s, 0.75, 3.95, 11.8, 2.6, "Conflict? Two cars aim at the same lane within 15 m -> priority decides",
         "- Priority = (urgency, closeness to ramp end, lower car number)\n- Urgency 3 = must merge now, 2 = escaping a hazard, 1 = normal lane change\n"
         "- Every car applies the same rule to the same messages, so all agree; the loser waits one second\n- Radio: 150 m range, 0.25 s delay, adjustable packet loss", accent=True)

    # 6 shield + memory
    s = slide("The LLM proposes, the safety shield decides what is allowed",
              "LLMs are slow and can be wrong. The shield is a few strict rules. The fleet memory is our light-weight "
              "alternative to federated learning: lessons are shared as text.", "Safety and learning")
    card(s, 0.75, 1.6, 5.8, 3.9, "Safety shield (rules)",
         "- Gap in front and behind >= max(6 m, 0.3 s x speed)\n- Time-to-collision >= 2.5 s\n- Lane must exist; never into the on-ramp\n"
         "- No speeding up if the car ahead is < 4 s away\n- Unsafe -> replaced by a safe action, logged as a veto", accent=True)
    card(s, 6.85, 1.6, 5.7, 3.9, "Fleet memory (shared lessons)",
         "- After a drive with trouble, the car's LLM writes one lesson\n- e.g. \"Check the rear gap before changing lanes.\"\n"
         "- Saved in lessons.json; the 3 newest go into every car's prompt\n- Knowledge is shared as text, not model weights")

    # 7 scenarios
    s = slide("Two scenarios designed so that cooperation matters",
              "Highway: trucks side by side make two agents want the middle lane at once; a stopped car at 560 m is "
              "beyond sensor range. Merge: the queue has 15 m gaps but a merging car needs about 17 m.", "Simulation")
    picture(s, "docs/figures/scenarios.png", 0.9, 1.4, w=11.5)

    # 8 references
    s = slide("Inspired by two reference projects - but distributed in a different way",
              "Reference 1 plans centrally with an LLM for drones. Reference 2 shares model weights through a Flower "
              "server. We distribute the decisions themselves and share intents and text lessons peer-to-peer.", "Positioning")
    card(s, 0.75, 1.6, 3.8, 3.4, "Ref 1: Multi-UAV + LLM", "- LLM turns a mission into waypoints\n- Often one central planner\n- Safety layer under the LLM\n- We borrow: LLM above a safe controller")
    card(s, 4.75, 1.6, 3.8, 3.4, "Ref 2: Cars + Flower", "- Each car trains locally\n- A server averages the weights\n- Data never leaves the car\n- We borrow: fleet learning without central data")
    card(s, 8.75, 1.6, 3.8, 3.4, "This project", "- One local LLM per car\n- Peer-to-peer intents\n- Leaderless priority rule\n- Shared text lessons\n- Per-node logs -> analytics", accent=True)

    # 9 experiments
    s = slide("Five experiments, each answering one question",
              f"Each setting ran with {R.seeds} random seeds; charts show means with 95% confidence intervals.", "Method")
    rows = [["ID", "Question", "What changes"], ["E1", "Does talking help?", "Rule-based vs LLM no radio vs LLM + radio"],
            ["E2", "Does it scale?", "2, 4, 6, 8 cars"], ["E3", "Bad radio?", "0-60 % of messages lost"],
            ["E4", "Which parts matter?", "Shield off, memory off"], ["E5", "Bigger LLM?", "llama3.2:3b vs llama3.1:8b"]]
    tbl = s.shapes.add_table(len(rows), 3, Inches(0.75), Inches(1.6), Inches(11.8), Inches(4.2)).table
    for c, w in enumerate((1.2, 4.6, 6.0)):
        tbl.columns[c].width = Inches(w)
    for r, row in enumerate(rows):
        for c, val in enumerate(row):
            cell = tbl.cell(r, c)
            cell.text = val
            para = cell.text_frame.paragraphs[0]
            para.runs[0].font.size = Pt(18)
            para.runs[0].font.bold = r == 0
            para.runs[0].font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF) if r == 0 else INK
            cell.fill.solid()
            cell.fill.fore_color.rgb = ACC if r == 0 else (SOFT if r % 2 else RGBColor(0xFF, 0xFF, 0xFF))
    text(s, 0.75, 6.1, 11.8, 0.6, "Metrics: collisions, success rate, merge time, shield vetoes, conflicts, messages and bytes per car, decision time", 15, INK2)

    # 10-13 results
    s = slide("Talking cars avoid lane-change crashes and merge faster",
              f"Highway collisions per episode: rule {col['rule']:.2f}, no radio {col['llm_nocomm']:.2f}, ours {col['llm_comm']:.2f}. "
              f"Merge time: {mt['rule']:.1f} s, {mt['llm_nocomm']:.1f} s, {mt['llm_comm']:.1f} s.", "Results")
    picture(s, os.path.join(R.fig, "fig_e1_comparison.png"), 0.75, 1.3, w=11.8)
    stat(s, 0.9, 5.65, 3.8, f"{col['llm_comm']:.2f} vs {col['llm_nocomm']:.2f}", "highway collisions per episode, with vs without V2V")
    stat(s, 4.9, 5.65, 3.8, f"{mt['llm_comm']:.1f} s vs {mt['llm_nocomm']:.1f} s", "time for ramp cars to merge, with vs without V2V")
    stat(s, 8.9, 5.65, 3.8, f"{R.m(E1, 'bytes_per_agent_s', system='llm_comm', scenario='highway'):.0f} B/s", "radio traffic per car - the cost of talking")
    if "E2" in R.e:
        s = slide("Radio load per car grows with fleet size, while safety holds",
                  "Each car hears every neighbour in range, so load grows roughly linearly - but stays under 1 kB/s.", "Results")
        picture(s, os.path.join(R.fig, "fig_e2_scalability.png"), 0.75, 1.5, w=11.8)
    if "E3" in R.e:
        s = slide("Lost messages slowly erode the benefit - no sudden failure",
                  "A lost intent can make two cars think they both own the gap; a lost merge request means no yield. "
                  "The shield still catches most problems.", "Results")
        picture(s, os.path.join(R.fig, "fig_e3_packet_loss.png"), 1.3, 1.5, w=10.7)
    if "E4" in R.e:
        s = slide("Without the safety shield, imperfect decisions become crashes",
                  "Ablation: removing the shield causes collisions; removing memory has no effect with the stand-in "
                  "(it ignores lessons) - measure it with the real LLM.", "Results")
        picture(s, os.path.join(R.fig, "fig_e4_ablation.png"), 1.3, 1.5, w=10.7)
    if R.e5 is not None:
        s = slide("Bigger local model: slower decisions - is the extra quality worth it?", "", "Results")
        picture(s, os.path.join("docs", "figures", "fig_e5_models.png"), 1.3, 1.5, w=10.7)

    # 14 analytics
    s = slide("Distributed analytics: every node logs for itself, then we merge",
              "Like log shipping in real distributed systems: no shared database. Show the dashboard live if possible "
              "(results/<tag>/dashboard.html).", "Analytics")
    steps = [("1. Log", "each node writes its own file: A0.jsonl, A1.jsonl, bus.jsonl"), ("2. Merge", "pandas collects and joins all node files"),
             ("3. Metrics", "per episode: safety, success, radio, LLM time"), ("4. Aggregate", "mean + 95% CI over seeds"),
             ("5. Present", "charts, HTML dashboard, this deck")]
    for i, (h, b) in enumerate(steps):
        card(s, 0.75 + i * 2.4, 1.8, 2.2, 2.6, h, b, accent=(i == 0), size=15)
    text(s, 0.75, 5.1, 11.8, 1.2, ["- Everything is one command each: run_experiments.py -> make_figures.py -> make_dashboard.py -> make_report.py",
                                   "- Rerun with the real LLM and the report and these slides rebuild with the new numbers"], 16)

    # 15 limitations
    s = slide("Limitations and next steps", "Be honest about simplifications; show you know the path to a real system.", "Discussion")
    card(s, 0.75, 1.6, 5.8, 3.9, "Limitations",
         "- Simplified road: instant lane changes, perfect 60 m sensors\n- Cars run as threads on one PC\n- Simulation waits for the LLM (0.5-2 s per decision)\n- No message security"
         + ("\n- Current numbers come from the stand-in model" if R.mock else ""))
    card(s, 6.85, 1.6, 5.7, 3.9, "Next steps",
         "- Cars as separate processes/machines (Ray, MQTT)\n- Asynchronous LLM decisions\n- Federated fine-tuning with Flower\n- Intersections, SUMO city-scale traffic", accent=True)

    # 16 conclusion
    s = slide("Conclusion", "End with the four take-aways, then invite questions.", "Wrap-up")
    text(s, 0.75, 1.6, 11.8, 4.5, [
        "- Every car thinks for itself with a local LLM - no central controller",
        "- Sharing intentions prevents two cars from taking the same gap",
        "- Asking for help (merge requests) makes hard merges easy",
        "- The LLM proposes; a simple safety shield keeps it safe",
        "- Talking is cheap (< 1 kB/s per car) and degrades gracefully",
    ], 22, INK)
    text(s, 0.75, 6.0, 11.8, 0.7, "Thank you - questions?", 26, ACC, True)
    prs.save(path)
    print("wrote", path)


def finish_with_word(docx_path: str) -> None:
    """If Microsoft Word is installed, fill in the table of contents and save a PDF copy."""
    try:
        import win32com.client
    except ImportError:
        print("(Word automation not available - open the .docx and accept 'update fields')")
        return
    word = None
    try:
        word = win32com.client.DispatchEx("Word.Application")
        word.Visible, word.DisplayAlerts = False, 0
        doc = word.Documents.Open(os.path.abspath(docx_path))
        for toc in doc.TablesOfContents:
            toc.Update()
        doc.Save()
        doc.ExportAsFixedFormat(os.path.abspath(docx_path[:-5] + ".pdf"), 17)
        doc.Close()
        print("updated table of contents and wrote", docx_path[:-5] + ".pdf")
    except Exception as e:  # noqa: BLE001
        print("(Word automation failed:", e, ")")
    finally:
        if word is not None:
            word.Quit()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="mock")
    ap.add_argument("--no-word", action="store_true", help="skip the Word TOC/PDF step")
    a = ap.parse_args()
    R = Results(a.tag)
    report = os.path.join("docs", "Report.docx")
    build_report(R, report)
    build_slides(R, os.path.join("docs", "Presentation.pptx"))
    if not a.no_word:
        finish_with_word(report)


if __name__ == "__main__":
    main()
