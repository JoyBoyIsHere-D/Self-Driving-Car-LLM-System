"""All tunable settings in one place.

Every experiment is just a different combination of these dataclasses,
so a run can always be reproduced from its saved config.json.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field


@dataclass
class SimConfig:
    scenario: str = "highway"      # "highway" or "merge"
    n_agents: int = 4              # cars driven by LLM agents
    n_humans: int = 10             # background cars driven by a simple rule model (IDM)
    duration_s: float = 60.0       # maximum episode length in simulated seconds
    dt: float = 0.25               # physics step (s)
    decision_every: int = 4        # agents decide every 4 steps = once per simulated second
    seed: int = 0


@dataclass
class CommsConfig:
    enabled: bool = True           # V2V radio on/off
    range_m: float = 150.0         # radio range
    latency_steps: int = 1         # delay before a message arrives (1 step = 0.25 s)
    loss_prob: float = 0.0         # probability that a message to one receiver is lost


@dataclass
class AgentConfig:
    system: str = "llm_comm"       # "rule" | "llm_nocomm" | "llm_comm"
    shield: bool = True            # rule-based safety check under every LLM action
    memory: bool = True            # shared lesson memory
    sensor_range_m: float = 60.0   # how far a car can "see" with its own sensors


@dataclass
class LLMConfig:
    backend: str = "mock"          # "mock" (built-in stand-in) or "ollama" (real local LLM)
    model: str = "llama3.2:3b"
    host: str = "http://localhost:11434"
    temperature: float = 0.2
    timeout_s: float = 30.0
    mock_error_rate: float = 0.05  # stand-in only: chance of a random (possibly bad) action


@dataclass
class RunConfig:
    sim: SimConfig = field(default_factory=SimConfig)
    comms: CommsConfig = field(default_factory=CommsConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    out_dir: str = "results/runs"
    run_name: str = "run"

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)


def make_config(system: str = "llm_comm", scenario: str = "highway", seed: int = 0,
                n_agents: int = 4, backend: str = "mock", model: str = "llama3.2:3b",
                loss_prob: float = 0.0, latency_steps: int = 1, shield: bool = True,
                memory: bool = True, run_name: str = "run", out_dir: str = "results/runs") -> RunConfig:
    """Convenience builder used by the scripts and experiments."""
    cfg = RunConfig(out_dir=out_dir, run_name=run_name)
    cfg.sim.scenario = scenario
    cfg.sim.seed = seed
    cfg.sim.n_agents = n_agents
    cfg.agent.system = system
    cfg.agent.shield = shield
    cfg.agent.memory = memory and system != "rule"
    cfg.comms.enabled = system == "llm_comm"
    cfg.comms.loss_prob = loss_prob
    cfg.comms.latency_steps = latency_steps
    cfg.llm.backend = backend
    cfg.llm.model = model
    return cfg
