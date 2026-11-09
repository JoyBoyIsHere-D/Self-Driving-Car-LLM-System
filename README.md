# Distributed LLM Driving: a network of self-driving cars that think with a local LLM and talk over V2V

Each car is an independent agent. It sees only its own surroundings (60 m sensors), asks a **local LLM**
(Ollama, e.g. `llama3.2:3b` or `llama3.1:8b`) what to do next, passes that through a **safety shield**, tells
nearby cars its plan over a simulated **V2V radio**, and resolves conflicts with them **without any central
controller**. Every car writes its own log, and an **analytics pipeline** merges the logs into metrics, charts and
a dashboard.

The full explanation (concepts, design, results, how to present it) is in **`docs/Report.docx`**.
The slides are in **`docs/Presentation.pptx`**.

## Quick start

```bash
python -m pytest
```

```bash
python run_demo.py --scenario merge
```

```bash
python run_experiments.py
```

```bash
python make_figures.py
```

```bash
python make_dashboard.py
```

```bash
python make_report.py
```

Everything above runs with the built-in **stand-in decision model** (`--backend mock`), so it works without a GPU
or Ollama. Results from the stand-in are labelled as such everywhere. They test the system, not the LLM.

## Using the real local LLM (Ollama)

1. Install Ollama from https://ollama.com/download (Windows installer), or run `winget install Ollama.Ollama`.
2. Pull the models:

```bash
ollama pull llama3.2:3b
```

```bash
ollama pull llama3.1:8b
```

3. Run with the LLM. This is slower: about 0.5 to 2 s per decision on an RTX 3050, and the 8B model partly runs on the CPU because it is bigger than 4 GB of VRAM:

```bash
python run_demo.py --scenario merge --backend ollama --model llama3.2:3b
```

```bash
python run_experiments.py --backend ollama --model llama3.2:3b --seeds 3
```

```bash
python run_experiments.py --exp E5 --models llama3.2:3b llama3.1:8b --seeds 3
```

```bash
python make_figures.py --tag ollama_llama3.2-3b
```

```bash
python make_report.py --tag ollama_llama3.2-3b
```

## Project layout

```text
src/fleet/
  config.py                 all settings (dataclasses)
  sim/world.py              road, cars, IDM physics, crash detection
  sim/scenarios.py          highway (overtake conflict + hazard) and merge (congested on-ramp)
  agents/vehicle_agent.py   one car = perceive -> decide -> shield -> negotiate -> act -> reflect
  llm/client.py             OllamaLLM (real) and MockLLM (stand-in)
  llm/prompts.py            system prompt + scene-to-text
  comms/bus.py              V2V radio: range, latency, packet loss
  coordination/negotiation.py   decentralised conflict resolution (priority rule)
  safety/shield.py          rule-based veto of unsafe actions
  memory/lessons.py         shared fleet memory (lessons.json)
  analytics/telemetry.py    per-node event logs (JSONL)
  analytics/metrics.py      merge logs -> episode metrics
  runner.py                 runs one episode (cars decide in parallel threads)
run_demo.py                 watch one episode, save a GIF
run_experiments.py          experiments E1-E5 -> CSV
make_figures.py             CSV -> charts and diagrams (docs/figures)
make_dashboard.py           interactive HTML dashboard
make_report.py              builds docs/Report.docx and docs/Presentation.pptx
tests/test_core.py          unit and integration tests
```

## Experiments

| ID | Question | What changes |
| --- | --- | --- |
| E1 | Does talking help? | rule-based vs LLM without V2V vs LLM with V2V, in both scenarios |
| E2 | Does it scale? | 2, 4, 6, 8 LLM cars |
| E3 | What if the radio is bad? | 0, 20, 40, 60 % of messages lost |
| E4 | Which parts matter? | safety shield off, fleet memory off |
| E5 | Bigger model = better? | llama3.2:3b vs llama3.1:8b (needs Ollama) |
