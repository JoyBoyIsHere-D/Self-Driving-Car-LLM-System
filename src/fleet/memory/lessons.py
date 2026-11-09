"""Shared fleet memory: short text "lessons" written by cars after a drive.

This is the knowledge-sharing part of the system. Instead of sharing model
weights (as federated learning would), cars share plain-language lessons that
any other car's LLM can read in its prompt. Stored as one JSON file.
"""
from __future__ import annotations

import json
import os
import threading


class LessonStore:
    def __init__(self, path: str, max_items: int = 50):
        self.path, self.max_items = path, max_items
        self._lock = threading.Lock()
        self.items: list[dict] = []
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                self.items = json.load(f)

    def add(self, scenario: str, text: str, author: str, episode: str) -> None:
        text = text.strip()
        if not text:
            return
        with self._lock:
            if any(i["text"].lower() == text.lower() and i["scenario"] == scenario for i in self.items):
                return  # de-duplicate
            self.items.append({"scenario": scenario, "text": text, "author": author, "episode": episode})
            self.items = self.items[-self.max_items:]
            os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
            with open(self.path, "w", encoding="utf-8") as f:
                json.dump(self.items, f, indent=2)

    def retrieve(self, scenario: str, k: int = 3) -> list[str]:
        with self._lock:
            same = [i["text"] for i in self.items if i["scenario"] == scenario]
        return same[-k:]
