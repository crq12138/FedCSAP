from __future__ import annotations


def apply_sf_attack(local_state, global_state, scale: float = 1.0):
    """Sign-Flipping attack on client updates relative to the global model.

    Given local weights ``w_local = w_global + delta``, sign-flipping should
    submit ``w_global - scale * delta``.
    """
    attacked = {}
    for k, local_tensor in local_state.items():
        global_tensor = global_state[k]
        delta = local_tensor - global_tensor
        attacked[k] = global_tensor - scale * delta
    return attacked
