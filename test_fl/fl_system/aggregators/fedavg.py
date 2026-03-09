from __future__ import annotations

import torch

from .base import Aggregator


class FedAvgAggregator(Aggregator):
    def aggregate(self, client_states: list[dict], client_weights: list[float]) -> dict:
        total_weight = sum(client_weights)
        if total_weight <= 0:
            raise ValueError("Total client weight must be positive.")

        out = {}
        keys = client_states[0].keys()
        for k in keys:
            acc = torch.zeros_like(client_states[0][k])
            for state, w in zip(client_states, client_weights):
                acc += state[k] * (w / total_weight)
            out[k] = acc
        return out
