import logging
import time

import numpy as np
import torch

from fedcsap_utils.validation_test import validation_test

logger = logging.getLogger('logger')


def _is_enabled(value):
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, (int, float)):
        return value != 0
    return str(value).strip().lower() in {'1', 'true', 'yes', 'y', 'on'}


def _frfl_build_candidate_model(helper, target_model, delta_model):
    candidate_model = helper.new_model()
    candidate_model.copy_params(target_model.state_dict())
    for layer_name, layer_data in candidate_model.state_dict().items():
        update_per_layer = delta_model[layer_name]
        try:
            layer_data.add_(update_per_layer)
        except Exception:
            layer_data.add_(update_per_layer.to(layer_data.dtype))
    return candidate_model


def _frfl_score_candidate_on_validator(helper, candidate_model, validator_id):
    correct_by_class, _, count_per_class = validation_test(helper, candidate_model, validator_id)
    total_correct = float(np.sum([correct_by_class[c] for c in count_per_class.keys()]))
    total_count = float(np.sum([count_per_class[c] for c in count_per_class.keys()]))
    if total_count <= 0:
        return 0.0
    score = total_correct / total_count
    if score > 1.0:
        score = score / 100.0
    return float(np.clip(score, 0.0, 1.0))


def _frfl_consensus_median(score_dict):
    return {name: float(np.median(np.array(scores, dtype=np.float32))) for name, scores in score_dict.items()}


def _frfl_adaptive_selection(helper, consensus_scores):
    all_trainers = list(consensus_scores.keys())
    values = np.array([consensus_scores[name] for name in all_trainers], dtype=np.float32)
    if values.size == 0:
        return [], [], 0.0

    u_bar = float(np.mean(values))
    u_hat = float(np.median(values))
    sigma = float(np.std(values))

    tol = float(helper.params.get('frfl_mean_median_tol', 1e-8))
    xi1 = float(helper.frfl_xi1)
    xi2 = float(helper.frfl_xi2)

    if np.isclose(u_bar, u_hat, atol=tol):
        selected = [c for c in all_trainers if consensus_scores[c] >= u_bar]
    elif u_bar > u_hat:
        selected = [c for c in all_trainers if consensus_scores[c] > (u_bar + xi1 * sigma)]
    else:
        selected = [c for c in all_trainers if consensus_scores[c] > (u_bar - xi2 * sigma)]

    if len(selected) == 0:
        best_client = max(all_trainers, key=lambda name: consensus_scores[name])
        selected = [best_client]

    malicious = [c for c in all_trainers if c not in set(selected)]
    return selected, malicious, u_bar


def _frfl_adapt_xi(helper, current_mean):
    frfl_xi_min = float(helper.params.get('frfl_xi_min', 0.05))
    frfl_xi_max = float(helper.params.get('frfl_xi_max', 5.0))
    frfl_adapt_clip = float(helper.params.get('frfl_adapt_clip', 0.25))

    prev_mean = helper.frfl_prev_mean_score
    delta_mean = 0.0 if prev_mean is None else (current_mean - prev_mean)

    # Paper uses Δu_t / u_t to adapt ξ1 and ξ2.
    # We implement Δu_t as current-round mean consensus score minus previous-round mean consensus score.
    ratio = delta_mean / max(abs(current_mean), 1e-12)
    ratio = float(np.clip(ratio, -frfl_adapt_clip, frfl_adapt_clip))

    helper.frfl_xi1 = float(np.clip(helper.frfl_xi1 * (1.0 - ratio), frfl_xi_min, frfl_xi_max))
    helper.frfl_xi2 = float(np.clip(helper.frfl_xi2 * (1.0 + ratio), frfl_xi_min, frfl_xi_max))
    helper.frfl_prev_mean_score = current_mean


def run_frfl(helper, target_model, updates, epoch, validator_names=None, committee_members=None):
    start = time.time()
    helper.ensure_frfl_state()

    names, delta_models = [], []
    for name, data in updates.items():
        names.append(name)
        delta_models.append(data[2])

    if len(names) == 0:
        logger.warning('No updates for FRFL at epoch %s.', epoch)
        return

    if validator_names is None:
        validators = [n for n in names if n not in set(helper.frfl_blocked_from_validation)]
    else:
        validators = [n for n in validator_names if n in helper.participants_list]

    if len(validators) == 0:
        validators = names

    score_dict = {name: [] for name in names}
    for client_name, delta_model in zip(names, delta_models):
        candidate_model = _frfl_build_candidate_model(helper, target_model, delta_model)
        for validator_id in validators:
            score = _frfl_score_candidate_on_validator(helper, candidate_model, validator_id)
            score_dict[client_name].append(score)

    consensus_scores = _frfl_consensus_median(score_dict)
    selected_clients, malicious_clients, mean_score = _frfl_adaptive_selection(helper, consensus_scores)
    _frfl_adapt_xi(helper, mean_score)

    if _is_enabled(helper.params.get('frfl_exclude_malicious_from_validation', True)):
        helper.frfl_blocked_from_validation = set(malicious_clients)
    else:
        helper.frfl_blocked_from_validation = set()

    selected_indices = [idx for idx, name in enumerate(names) if name in set(selected_clients)]
    selected_delta_models = [delta_models[idx] for idx in selected_indices]

    if len(selected_delta_models) == 0:
        logger.warning('FRFL selected no client at epoch %s; skipping aggregation.', epoch)
        return

    weights = torch.ones(len(selected_delta_models), dtype=torch.float32)
    aggregate_weights = helper.weighted_average_oracle(selected_delta_models, weights)

    for layer_name, layer_data in target_model.state_dict().items():
        update_per_layer = aggregate_weights[layer_name] * helper.params['eta']
        try:
            layer_data.add_(update_per_layer)
        except Exception:
            layer_data.add_(update_per_layer.to(layer_data.dtype))

    logger.info(
        'FRFL epoch %s: validators=%s selected=%s filtered=%s xi1=%.4f xi2=%.4f mean=%.6f time=%.3fs',
        epoch,
        validators,
        selected_clients,
        malicious_clients,
        helper.frfl_xi1,
        helper.frfl_xi2,
        mean_score,
        time.time() - start,
    )
