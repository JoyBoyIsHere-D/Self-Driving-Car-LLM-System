"""Run the project's experiments and save one CSV per experiment.

Examples
  python run_experiments.py                       # all experiments, stand-in LLM (fast)
  python run_experiments.py --backend ollama --model llama3.2:3b --seeds 3
  python run_experiments.py --exp E5 --backend ollama --models llama3.2:3b llama3.1:8b --seeds 3

Experiments
  E1  Main comparison: rule-based vs LLM without radio vs LLM with radio (ours), both scenarios
  E2  Scalability: 2, 4, 6, 8 LLM cars
  E3  Network robustness: packet loss 0-60 %
  E4  Ablation: safety shield on/off, fleet memory on/off
  E5  Model size: compare local LLMs (Ollama only)
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from fleet.config import make_config  # noqa: E402
from fleet.llm.client import OllamaLLM  # noqa: E402
from fleet.runner import run_episode  # noqa: E402


def tag_for(backend: str, model: str) -> str:
    return "mock" if backend == "mock" else f"ollama_{model.replace(':', '-').replace('/', '-')}"


def run_grid(name: str, settings: list[dict], seeds: int, backend: str, model: str, root: str) -> pd.DataFrame:
    rows = []
    total = len(settings) * seeds
    t0 = time.time()
    for k, s in enumerate(settings):
        for seed in range(seeds):
            label = "_".join(f"{v}" for v in s.values())
            cfg = make_config(seed=seed, backend=s.get("backend", backend), model=s.get("model", model),
                              run_name=f"{name}_{label}_s{seed}".replace(":", "-"), out_dir=os.path.join(root, "runs"),
                              **{k2: v for k2, v in s.items() if k2 not in ("backend", "model")})
            lessons = os.path.join(root, "memory", f"{name}_{s.get('system', 'llm_comm')}_{s.get('scenario', 'highway')}"
                                                  f"_{s.get('model', model).replace(':', '-')}.json")
            m = run_episode(cfg, lessons_path=lessons)
            m.update({f"cfg_{a}": b for a, b in s.items()})
            rows.append(m)
            done = k * seeds + seed + 1
            print(f"  [{name}] {done}/{total} {label} seed={seed}  success={m['success_rate']:.2f} "
                  f"collisions={m['collisions']}  ({time.time() - t0:.0f}s elapsed)", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(root, f"{name}.csv"), index=False)
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", nargs="+", default=["E1", "E2", "E3", "E4"])
    ap.add_argument("--backend", default="mock", choices=["mock", "ollama"])
    ap.add_argument("--model", default="llama3.2:3b")
    ap.add_argument("--models", nargs="+", default=["llama3.2:3b", "llama3.1:8b"], help="models for E5")
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--out", default="results")
    a = ap.parse_args()

    if a.backend == "ollama" or "E5" in a.exp:
        for m in ([a.model] if a.backend == "ollama" else []) + (a.models if "E5" in a.exp else []):
            ok, why = OllamaLLM(m, "http://localhost:11434").available()
            if not ok:
                sys.exit(f"Cannot use Ollama: {why}")

    root = os.path.join(a.out, tag_for(a.backend, a.model))
    os.makedirs(root, exist_ok=True)
    print(f"Writing results to {root}")
    scen = ["highway", "merge"]
    if "E1" in a.exp:
        run_grid("E1", [{"system": s, "scenario": sc} for sc in scen for s in ("rule", "llm_nocomm", "llm_comm")],
                 a.seeds, a.backend, a.model, root)
    if "E2" in a.exp:
        run_grid("E2", [{"system": "llm_comm", "scenario": "highway", "n_agents": n} for n in (2, 4, 6, 8)],
                 a.seeds, a.backend, a.model, root)
    if "E3" in a.exp:
        run_grid("E3", [{"system": "llm_comm", "scenario": sc, "loss_prob": p}
                        for sc in scen for p in (0.0, 0.2, 0.4, 0.6)], a.seeds, a.backend, a.model, root)
    if "E4" in a.exp:
        run_grid("E4", [{"system": "llm_comm", "scenario": sc, "shield": sh, "memory": mem}
                        for sc in scen for sh, mem in ((True, True), (False, True), (True, False))],
                 a.seeds, a.backend, a.model, root)
    if "E5" in a.exp:
        settings = [{"system": "llm_comm", "scenario": sc, "backend": "ollama", "model": m}
                    for sc in scen for m in a.models]
        run_grid("E5", settings, a.seeds, "ollama", a.model, os.path.join(a.out, "ollama_models"))
    print("done")


if __name__ == "__main__":
    main()
