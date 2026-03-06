from __future__ import annotations


def apply_sf_attack(state_dict, scale: float = 1.0):
    """Sign-Flipping: 对客户端更新取反。"""
    attacked = {}
    for k, v in state_dict.items():
        attacked[k] = -scale * v
    return attacked
