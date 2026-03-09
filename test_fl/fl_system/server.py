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
            self.model.load_state_dict(avg_state)
            # eta = float(self.cfg.eta)
            # if not 0.0 < eta <= 1.0:
            #     raise ValueError("eta must be in (0, 1].")

            # blended_state = {}
            # for k, global_tensor in self.model.state_dict().items():
            #     blended_state[k] = global_tensor.detach().cpu() + eta * (avg_state[k] - global_tensor.detach().cpu())
            # self.model.load_state_dict(blended_state)

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
