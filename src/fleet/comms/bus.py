"""Simulated V2V (vehicle-to-vehicle) radio.

Cars never call each other directly: they publish messages to this bus and
read their own inbox. The bus models what makes a real network "distributed":
limited radio range, delivery delay (latency) and lost packets.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np


@dataclass
class Message:
    sender: str
    mtype: str          # "INTENT" | "MERGE_REQUEST" | "HAZARD"
    payload: dict
    x: float            # sender position when sent
    step_sent: int
    deliver_at: int = 0

    def size_bytes(self) -> int:
        return len(json.dumps({"s": self.sender, "t": self.mtype, "p": self.payload, "x": round(self.x, 1)}))


@dataclass
class MessageBus:
    range_m: float
    latency_steps: int
    loss_prob: float
    rng: np.random.Generator
    enabled: bool = True
    pending: list[Message] = field(default_factory=list)
    inbox: dict[str, list[Message]] = field(default_factory=dict)
    stats: dict[str, int] = field(default_factory=lambda: {"sent": 0, "delivered": 0, "dropped": 0, "bytes": 0})

    def publish(self, msg: Message) -> None:
        if not self.enabled:
            return
        msg.deliver_at = msg.step_sent + self.latency_steps
        self.pending.append(msg)
        self.stats["sent"] += 1

    def _try_deliver(self, msg: Message, receiver: str, rx_x: float) -> bool:
        if receiver == msg.sender or abs(rx_x - msg.x) > self.range_m:
            return False
        if self.rng.random() < self.loss_prob:
            self.stats["dropped"] += 1
            return False
        self.stats["delivered"] += 1
        self.stats["bytes"] += msg.size_bytes()
        return True

    def deliver(self, step: int, positions: dict[str, float]) -> None:
        """Move every message that is due into the inbox of each car in range."""
        if not self.enabled:
            return
        due = [m for m in self.pending if m.deliver_at <= step]
        self.pending = [m for m in self.pending if m.deliver_at > step]
        for m in due:
            for vid, x in positions.items():
                if self._try_deliver(m, vid, x):
                    self.inbox.setdefault(vid, []).append(m)

    def pop_inbox(self, vid: str) -> list[Message]:
        return self.inbox.pop(vid, [])

    def exchange_intents(self, intents: dict[str, Message], positions: dict[str, float]) -> dict[str, list[Message]]:
        """Fast in-tick handshake round: every car hears its neighbours' intents
        (same range and loss rules, no extra delay)."""
        received: dict[str, list[Message]] = {vid: [] for vid in positions}
        if not self.enabled:
            return received
        for msg in intents.values():
            self.stats["sent"] += 1
            for vid, x in positions.items():
                if self._try_deliver(msg, vid, x):
                    received[vid].append(msg)
        return received
