from __future__ import annotations

import random

import torch

from .aggregators import build_aggregator
from .client import FLClient


class FLServer:
    def __init__(self, model, clients: list[FLClient], test_loader, cfg):
        self.model = model
        self.clients = clients
        self.test_loader = test_loader
        self.cfg = cfg
        self.aggregator = build_aggregator(cfg.aggregation)

        rng = random.Random(cfg.seed)
        mal_count = int(len(clients) * cfg.mal_pcnt)
        self.malicious_set = set(rng.sample(range(len(clients)), mal_count))

    @staticmethod
    def _apply_gaussian_noise(state: dict, noise_std: float) -> dict:
        if noise_std <= 0:
            return state
        noised_state = {}
        for k, v in state.items():
            if torch.is_floating_point(v):
                noised_state[k] = v + torch.randn_like(v) * noise_std
            else:
                noised_state[k] = v
        return noised_state

    @staticmethod
    def _hybrid_update(global_state: dict, agg_state: dict, alpha: float) -> dict:
        if not 0.0 < alpha <= 1.0:
            raise ValueError("fedcsap_hybrid_alpha must be in (0, 1].")
        if alpha == 1.0:
            return agg_state
        out = {}
        for k, global_tensor in global_state.items():
            out[k] = global_tensor + alpha * (agg_state[k] - global_tensor)
        return out

    def train(self):
        for rnd in range(1, self.cfg.rounds + 1):
            client_states = []
            client_weights = []
            global_state = {
                k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()
            }
            for c in self.clients:
                state, weight = c.local_train(
                    self.model,
                    self.cfg.lr,
                    self.cfg.local_epochs,
                    self.cfg.momentum,
                    self.cfg.weight_decay,
                )
                attacked = c.maybe_attack(
                    state,
                    global_state,
                    self.cfg.attack,
                    c.client_id in self.malicious_set,
                )
                client_states.append(attacked)
                client_weights.append(weight)

            avg_state = self.aggregator.aggregate(client_states, client_weights)

            # FEDCSAP 混合更新：global <- (1-alpha)*global + alpha*aggregate
            if self.cfg.aggregation == "fedcsap":
                next_state = self._hybrid_update(global_state, avg_state, self.cfg.fedcsap_hybrid_alpha)
            else:
                next_state = avg_state

            # 可选高斯加噪
            next_state = self._apply_gaussian_noise(next_state, self.cfg.gaussian_noise_std)
            self.model.load_state_dict(next_state)

            acc = self.evaluate()
            print(f"[Round {rnd:03d}/{self.cfg.rounds}] test_acc={acc:.4f}")

    @torch.no_grad()
    def evaluate(self):
        self.model.eval()
        correct = 0
        total = 0
        device = self.cfg.device
        self.model.to(device)
        for x, y in self.test_loader:
            x = x.to(device)
            y = y.to(device).long().view(-1)
            logits = self.model(x)
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.numel()
        return correct / max(total, 1)
