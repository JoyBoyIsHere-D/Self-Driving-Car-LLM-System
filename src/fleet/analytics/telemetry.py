"""Distributed logging: every node (car, bus, world) writes its OWN event file.

results/runs/<run>/nodes/A0.jsonl, A1.jsonl, ..., bus.jsonl, world.jsonl
The analytics step later collects all node files and merges them - the same
pattern as log shipping in a real distributed system.
"""
from __future__ import annotations

import json
import os
import threading


class Telemetry:
    def __init__(self, run_dir: str):
        self.dir = os.path.join(run_dir, "nodes")
        os.makedirs(self.dir, exist_ok=True)
        self._files: dict[str, object] = {}
        self._locks: dict[str, threading.Lock] = {}
        self._guard = threading.Lock()

    def emit(self, node: str, event: str, **fields) -> None:
        with self._guard:
            if node not in self._files:
                self._files[node] = open(os.path.join(self.dir, f"{node}.jsonl"), "a", encoding="utf-8")
                self._locks[node] = threading.Lock()
        rec = {"node": node, "event": event, **fields}
        with self._locks[node]:
            self._files[node].write(json.dumps(rec) + "\n")

    def close(self) -> None:
        for f in self._files.values():
            f.close()
        self._files.clear()
