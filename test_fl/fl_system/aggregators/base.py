from __future__ import annotations

from abc import ABC, abstractmethod


class Aggregator(ABC):
    @abstractmethod
    def aggregate(self, client_states: list[dict], client_weights: list[float]) -> dict:
        pass
