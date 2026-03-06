from __future__ import annotations

import copy

import torch
from torch import nn

from .attacks import apply_sf_attack


class FLClient:
    def __init__(self, client_id: int, train_loader, device: str):
        self.client_id = client_id
        self.train_loader = train_loader
        self.device = device

    def local_train(self, global_model: nn.Module, lr: float, local_epochs: int):
        model = copy.deepcopy(global_model).to(self.device)
        model.train()
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
        criterion = nn.CrossEntropyLoss()

        for _ in range(local_epochs):
            for x, y in self.train_loader:
                x = x.to(self.device)
                y = y.to(self.device).long().view(-1)
                optimizer.zero_grad()
                loss = criterion(model(x), y)
                loss.backward()
                optimizer.step()

        state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        return state, len(self.train_loader.dataset)

    def maybe_attack(
        self,
        client_state: dict,
        global_state: dict,
        attack: str,
        is_malicious: bool,
    ):
        if not is_malicious or attack == "none":
            return client_state
        if attack == "sf":
            return apply_sf_attack(client_state, global_state)
        raise ValueError(f"Unsupported attack: {attack}")
