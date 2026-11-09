"""LLM back-ends.

OllamaLLM : a real local LLM served by Ollama (http://localhost:11434).
MockLLM   : a fast rule-based STAND-IN with the same interface. It reads the
            same observation and makes plausible decisions, including some
            deliberately bad ones (mock_error_rate), so the whole pipeline -
            messages, negotiation, shield, analytics - can be tested without
            a GPU. Results produced with it are NOT LLM results.
"""
from __future__ import annotations

import json
import random
import time

import requests

from .prompts import ACTIONS, REFLECT_PROMPT, SYSTEM_PROMPT, build_prompt


def _result(action: str, reason: str, t0: float, ok: bool = True, tokens: int = 0, error: str = "") -> dict:
    return {"action": action, "reason": reason, "latency_s": time.perf_counter() - t0,
            "ok": ok, "tokens": tokens, "error": error}


class OllamaLLM:
    def __init__(self, model: str, host: str, temperature: float = 0.2, timeout_s: float = 30.0):
        self.model, self.host, self.temperature, self.timeout_s = model, host.rstrip("/"), temperature, timeout_s

    def available(self) -> tuple[bool, str]:
        try:
            r = requests.get(f"{self.host}/api/tags", timeout=3)
            names = [m["name"] for m in r.json().get("models", [])]
        except Exception as e:  # noqa: BLE001 - any failure means "not reachable"
            return False, f"Ollama not reachable at {self.host}: {e}"
        if not any(n == self.model or n.split(":")[0] == self.model for n in names):
            return False, f"model {self.model} not pulled (have: {names}); run: ollama pull {self.model}"
        return True, "ok"

    def _chat(self, system: str, user: str, max_tokens: int = 80) -> tuple[dict, int]:
        body = {"model": self.model, "stream": False, "format": "json", "keep_alive": "15m",
                "options": {"temperature": self.temperature, "num_predict": max_tokens},
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
        r = requests.post(f"{self.host}/api/chat", json=body, timeout=self.timeout_s)
        r.raise_for_status()
        data = r.json()
        tokens = int(data.get("prompt_eval_count", 0)) + int(data.get("eval_count", 0))
        return json.loads(data["message"]["content"]), tokens

    def decide(self, obs: dict) -> dict:
        t0 = time.perf_counter()
        prompt = build_prompt(obs)
        last_err, tokens = "", 0
        for _ in range(2):                       # one retry on a bad answer
            try:
                out, tok = self._chat(SYSTEM_PROMPT, prompt)
                tokens += tok
                action = str(out.get("action", "")).upper().strip()
                if action in ACTIONS:
                    return _result(action, str(out.get("reason", ""))[:120], t0, tokens=tokens)
                last_err = f"invalid action {action!r}"
            except requests.Timeout:
                return _result("IDLE", "LLM timeout - keep lane", t0, ok=False, tokens=tokens, error="timeout")
            except Exception as e:  # noqa: BLE001
                last_err = f"{type(e).__name__}: {e}"
        return _result("IDLE", "fallback after bad LLM output", t0, ok=False, tokens=tokens, error=last_err)

    def reflect(self, vid: str, scenario: str, events: str) -> str:
        try:
            out, _ = self._chat("You are a concise driving coach.",
                                REFLECT_PROMPT.format(vid=vid, scenario=scenario, events=events), max_tokens=60)
            return str(out.get("lesson", "")).strip()[:160]
        except Exception:  # noqa: BLE001
            return ""


class MockLLM:
    """Heuristic stand-in. It checks FRONT gaps but is careless about REAR gaps,
    like an imperfect LLM - the safety shield is there to catch that."""

    def __init__(self, error_rate: float = 0.05, seed: int = 0):
        self.error_rate = error_rate
        self.rng = random.Random(seed)

    @staticmethod
    def _front_ok(lane: dict | None, need: float = 18.0) -> bool:
        return lane is not None and not lane.get("is_ramp") and (lane["front_gap"] is None or lane["front_gap"] > need)

    def decide(self, obs: dict) -> dict:
        t0 = time.perf_counter()
        build_prompt(obs)  # same work as the real path, keeps timing comparable
        if self.rng.random() < self.error_rate:
            return _result(self.rng.choice(ACTIONS), "(stand-in random choice)", t0)
        L, C, R = obs["lanes"]["left"], obs["lanes"]["current"], obs["lanes"]["right"]
        if obs["on_ramp"]:
            if self._front_ok(L, 8):
                return _result("LANE_LEFT", "on ramp, merging into the main lane", t0)
            return _result("IDLE" if obs["ramp_left_m"] > 60 else "SLOWER", "on ramp, waiting for a gap", t0)
        for m in obs["merge_requests"]:
            if obs["lane"] == 1 and -10 < m["dx"] < 35:
                if self._front_ok(L, 25):
                    return _result("LANE_LEFT", f"make room for merging {m['from']}", t0)
                return _result("YIELD", f"let {m['from']} merge in front", t0)
        for h in obs["hazards"]:
            if h["lane"] == obs["lane"] and h["dist"] < 300:
                for name, lane in (("LANE_LEFT", L), ("LANE_RIGHT", R)):
                    if self._front_ok(lane, 25) and not any(z["lane"] == lane["lane"] for z in obs["hazards"]):
                        return _result(name, "leave the lane with a stopped car ahead", t0)
                return _result("SLOWER", "stopped car ahead, no free lane yet", t0)
        fg, fv = C["front_gap"], C["front_v"]
        if fg is not None and fg < 40 and fv is not None and fv < obs["v_target"] - 2:
            for name, lane in (("LANE_LEFT", L), ("LANE_RIGHT", R)):
                if self._front_ok(lane, fg + 20) and not any(z["lane"] == lane["lane"] for z in obs["hazards"]):
                    return _result(name, "overtake the slower car ahead", t0)
            return _result("IDLE", "slow car ahead, adjacent lanes busy", t0)
        if obs["v"] < obs["v_target"] - 1 and (fg is None or fg > 50):
            return _result("FASTER", "road ahead is clear", t0)
        return _result("IDLE", "keep lane and speed", t0)

    def reflect(self, vid: str, scenario: str, events: str) -> str:
        if "veto" in events:
            return "Check the rear gap before changing lanes; fast cars behind close the gap quickly."
        if "conflict" in events:
            return "Announce lane changes early; when two cars target one gap, the lower-priority car waits."
        if "merge" in events:
            return "Main-lane cars should yield early when a ramp car asks to merge."
        return ""


def make_llm(backend: str, model: str, host: str, temperature: float, timeout_s: float,
             mock_error_rate: float, seed: int):
    if backend == "ollama":
        return OllamaLLM(model, host, temperature, timeout_s)
    return MockLLM(mock_error_rate, seed)
