import logging
import time
import copy
import numpy as np
import torch

from sklearn.cluster import KMeans

import config
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


def _compute_cvar_bottom_q_mean(delta_f1_by_class, bottom_q):
    scores = np.array(delta_f1_by_class, dtype=np.float32)
    if scores.size == 0:
        return 0.0
    q = float(bottom_q)
    q = max(0.0, min(1.0, q))
    k = int(np.ceil(scores.size * q))
    k = max(1, k)
    sorted_scores = np.sort(scores)
    return float(np.mean(sorted_scores[:k]))


def _one_dim_kmeans_split(rep_scores):
    if len(rep_scores) == 0:
        return []
    if len(rep_scores) == 1:
        return [0]

    values = np.array(rep_scores, dtype=np.float32).reshape(-1, 1)
    min_val = float(np.min(values))
    max_val = float(np.max(values))

    if np.isclose(min_val, max_val):
        return [0 for _ in rep_scores]

    init_centers = np.array([[min_val], [max_val]], dtype=np.float32)
    kmeans = KMeans(n_clusters=2, init=init_centers, n_init=1, random_state=0)
    labels = kmeans.fit_predict(values)
    centers = kmeans.cluster_centers_.reshape(-1)
    high_label = int(np.argmax(centers))
    return [1 if int(label) == high_label else 0 for label in labels]


def _sanitize_delta_model(delta_model):
    sanitized = {}
    found_non_finite = False
    for layer_name, layer_tensor in delta_model.items():
        layer_fp32 = layer_tensor.detach().float()
        finite_mask = torch.isfinite(layer_fp32)
        if not bool(torch.all(finite_mask)):
            found_non_finite = True
            layer_fp32 = torch.nan_to_num(layer_fp32, nan=0.0, posinf=0.0, neginf=0.0)
        sanitized[layer_name] = layer_fp32.to(layer_tensor.device).type_as(layer_tensor)
    return sanitized, found_non_finite


def _safe_cosine_sims(flat_updates, idx):
    anchor = flat_updates[idx]
    anchor_norm = np.linalg.norm(anchor)
    if not np.isfinite(anchor_norm) or anchor_norm <= 0:
        return np.zeros(len(flat_updates), dtype=np.float32)

    sims = np.zeros(len(flat_updates), dtype=np.float64)
    eps = 1e-12
    for j, other in enumerate(flat_updates):
        other_norm = np.linalg.norm(other)
        if (not np.isfinite(other_norm)) or other_norm <= 0:
            sims[j] = 0.0
            continue
        dot_val = float(np.dot(anchor, other))
        if not np.isfinite(dot_val):
            sims[j] = 0.0
            continue
        cos_val = dot_val / max(anchor_norm * other_norm, eps)
        if not np.isfinite(cos_val):
            cos_val = 0.0
        sims[j] = np.clip(cos_val, -1.0, 1.0)

    sims = np.maximum(sims, 0.0)
    return sims.astype(np.float32)




def _select_stable_clients(rep_scores, high_cluster_flags):
    num_clients = len(rep_scores)
    if num_clients == 0:
        return []

    selected = [i for i, flag in enumerate(high_cluster_flags) if flag == 1]

    score_arr = np.array(rep_scores, dtype=np.float32)
    spread = float(np.max(score_arr) - np.min(score_arr)) if num_clients > 0 else 0.0

    # If scores are almost indistinguishable, avoid collapsing to a single client.
    if spread < 1e-6:
        min_k = max(1, int(np.ceil(num_clients * 0.5)))
        min_k = min(num_clients, max(2, min_k)) if num_clients > 1 else 1
        top_indices = np.argsort(score_arr)[-min_k:]
        return sorted(int(i) for i in top_indices)

    # If kmeans degenerates to an empty/single selection, keep a small top-k committee.
    if len(selected) <= 1 and num_clients > 1:
        min_k = max(2, int(np.ceil(num_clients * 0.5)))
        min_k = min(num_clients, min_k)
        top_indices = np.argsort(score_arr)[-min_k:]
        selected = sorted(int(i) for i in top_indices)

    return selected


def _scale_update_dict(update_dict, scale):
    if scale >= 1.0:
        return update_dict
    for layer_name, layer_data in update_dict.items():
        update_dict[layer_name] = layer_data * scale
    return update_dict

def run_fedcsap(helper, target_model, updates, epoch, committee_members=None):
    start = time.time()

    names = []
    delta_models = []
    grads = []
    for name, data in updates.items():
        names.append(name)
        grads.append(data[1])
        delta_models.append(data[2])

    if len(names) == 0:
        logger.warning('No updates for fedcsap at epoch %s.', epoch)
        return

    if committee_members is None:
        validators = names
    else:
        validators = [member for member in committee_members if member in helper.participants_list]
        if len(validators) == 0:
            validators = names

    num_of_classes = config.num_of_classes_dict[helper.params['type']]
    bottom_q = helper.params['fedcsap_bottom_q']
    if bottom_q is None:
        bottom_q = 0.2

    # baseline per-class F1/acc profile for previous global model
    baseline_profile = {}
    for val_name in validators:
        val_score_by_class, _, _ = validation_test(helper, target_model, val_name)
        baseline_profile[val_name] = [float(val_score_by_class[i]) for i in range(num_of_classes)]

    sanitized_delta_models = []
    flat_updates = []
    for client_name, delta_model in zip(names, delta_models):
        clean_delta_model, model_non_finite = _sanitize_delta_model(delta_model)
        if model_non_finite:
            logger.warning(
                'fedcsap epoch %s: non-finite update detected for client %s; sanitizing before aggregation.',
                epoch,
                client_name,
            )
        sanitized_delta_models.append(clean_delta_model)

        flat_grad = np.array(helper.flatten_gradient_v2(clean_delta_model), dtype=np.float64)
        if not np.all(np.isfinite(flat_grad)):
            logger.warning(
                'fedcsap epoch %s: flattened update still non-finite for client %s; forcing zeros.',
                epoch,
                client_name,
            )
            flat_grad = np.nan_to_num(flat_grad, nan=0.0, posinf=0.0, neginf=0.0)
        flat_updates.append(flat_grad)

    norms = np.array([np.linalg.norm(grad) for grad in flat_updates], dtype=np.float64)
    if not np.all(np.isfinite(norms)):
        norms = np.nan_to_num(norms, nan=0.0, posinf=0.0, neginf=0.0)

    contrib_adjustment = helper.params['contrib_adjustment'] if helper.params['contrib_adjustment'] is not None else 0.25

    representative_scores = []
    per_client_delta_f1 = {}
    for idx, rep_name in enumerate(names):
        rep_model = helper.new_model()
        rep_model.copy_params(helper.target_model.state_dict())

        # bijective-style representative model: use the target client as anchor and
        # blend in other clients by cosine similarity with contribution control.
        cos_sims = _safe_cosine_sims(flat_updates, idx)

        norm_ref = float(norms[idx]) if np.isfinite(norms[idx]) else 0.0
        eps = 1e-12
        safe_norms = np.where(np.isfinite(norms), np.maximum(norms, eps), eps)
        clip_vals = np.minimum(norm_ref / safe_norms, 1.0).astype(np.float32)
        clip_vals = np.nan_to_num(clip_vals, nan=0.0, posinf=1.0, neginf=0.0)
        weight_vec = cos_sims * clip_vals
        weight_vec = np.nan_to_num(weight_vec, nan=0.0, posinf=0.0, neginf=0.0)
        weight_vec[idx] = 1.0

        others_contrib = float(np.sum(weight_vec) - weight_vec[idx])
        if others_contrib > 0:
            weight_vec = weight_vec * (contrib_adjustment / others_contrib)
            weight_vec[idx] = 1.0 - contrib_adjustment
        else:
            weight_vec = np.zeros_like(weight_vec)
            weight_vec[idx] = 1.0

        total = float(np.sum(weight_vec))
        if total <= 0:
            weight_vec = np.zeros_like(weight_vec)
            weight_vec[idx] = 1.0
        else:
            weight_vec = weight_vec / total

        rep_update = helper.weighted_average_oracle(sanitized_delta_models, torch.tensor(weight_vec))
        for layer_name, layer_data in rep_model.state_dict().items():
            update_per_layer = rep_update[layer_name]
            try:
                layer_data.add_(update_per_layer)
            except Exception:
                layer_data.add_(update_per_layer.to(layer_data.dtype))

        validator_scores = []
        validator_delta_by_class = []
        for val_name in validators:
            rep_score_by_class, _, _ = validation_test(helper, rep_model, val_name)
            delta_by_class = [
                float(rep_score_by_class[i]) - baseline_profile[val_name][i]
                for i in range(num_of_classes)
            ]
            scalar_score = _compute_cvar_bottom_q_mean(delta_by_class, bottom_q=bottom_q)
            validator_scores.append(scalar_score)
            validator_delta_by_class.append(delta_by_class)

        representative_scores.append(float(np.mean(validator_scores)))
        if len(validator_delta_by_class) > 0:
            mean_delta = np.mean(np.array(validator_delta_by_class, dtype=np.float32), axis=0)
            per_client_delta_f1[rep_name] = [float(v) for v in mean_delta.tolist()]
        else:
            per_client_delta_f1[rep_name] = [0.0 for _ in range(num_of_classes)]

    # 1D kmeans with min/max anchors
    high_cluster_flags = _one_dim_kmeans_split(representative_scores)
    selected_indices = _select_stable_clients(representative_scores, high_cluster_flags)

    if len(selected_indices) == 0:
        selected_indices = [int(np.argmax(representative_scores))]

    selected_clients = [names[i] for i in selected_indices]
    low_cluster_clients = [names[i] for i in range(len(names)) if i not in selected_indices]
    helper.update_participant_reputation(low_cluster_clients, names)

    # map representative -> original client update, then average aggregation (flshield-consistent weighted average)
    weight_vec = np.zeros(len(names), dtype=np.float32)
    for i in selected_indices:
        weight_vec[i] = 1.0 / len(selected_indices)

    aggregate_weights = helper.weighted_average_oracle(sanitized_delta_models, torch.tensor(weight_vec))

    committee_takeover_attack = _is_enabled(helper.params.get('fedcsap_committee_takeover_attack', False))
    committee_takeover = False
    if committee_takeover_attack and committee_members is not None:
        committee_size = len(committee_members)
        if committee_size > 0:
            committee_malicious_count = len([m for m in committee_members if m in helper.adversarial_namelist])
            committee_takeover = committee_malicious_count > (committee_size / 2.0)

    if committee_takeover:
        aggregate_weights = _scale_update_dict(aggregate_weights, -1.0)
        logger.warning(
            'fedcsap epoch %s: committee takeover attack triggered (%s/%s malicious); reversing aggregate update direction.',
            epoch,
            committee_malicious_count,
            committee_size,
        )

    positive_norms = norms[norms > 0]
    median_client_norm = float(np.median(positive_norms)) if positive_norms.size > 0 else 0.0
    agg_flat = np.array(helper.flatten_gradient_v2(aggregate_weights), dtype=np.float64)
    agg_norm = float(np.linalg.norm(agg_flat)) if agg_flat.size > 0 else 0.0

    cfg_clip = helper.params.get('fedcsap_global_clip_norm', None) if isinstance(helper.params, dict) else None
    max_agg_norm = float(cfg_clip) if cfg_clip is not None else (2.0 * median_client_norm if median_client_norm > 0 else 0.0)
    if max_agg_norm > 0 and np.isfinite(agg_norm) and agg_norm > max_agg_norm:
        scale = max_agg_norm / max(agg_norm, 1e-12)
        aggregate_weights = _scale_update_dict(aggregate_weights, scale)
        logger.warning(
            'fedcsap epoch %s: aggregate update norm %.6e exceeds clip %.6e; scaling by %.6f.',
            epoch,
            agg_norm,
            max_agg_norm,
            scale,
        )

    for layer_name, layer_data in target_model.state_dict().items():
        update_per_layer = aggregate_weights[layer_name] * helper.params['eta']
        try:
            layer_data.add_(update_per_layer)
        except Exception:
            layer_data.add_(update_per_layer.to(layer_data.dtype))

    logger.info(
        'fedcsap epoch %s: validators=%s, rep_scores=%s, selected_clients=%s, low_cluster_clients=%s, committee_takeover=%s, elapsed=%.2fs',
        epoch,
        validators,
        [round(s, 6) for s in representative_scores],
        selected_clients,
        low_cluster_clients,
        committee_takeover,
        time.time() - start,
    )


    if hasattr(helper, 'experiment_logger') and helper.experiment_logger is not None:
        selected_set = set(selected_clients)
        for i, client_name in enumerate(names):
            is_selected = client_name in selected_set
            trust_instant = 1 if is_selected else 0
            helper.experiment_logger.log_fedcsap_client_metrics(
                epoch=epoch,
                client_id=client_name,
                is_malicious=client_name in helper.adversarial_namelist,
                cvar_score=float(representative_scores[i]),
                cluster_label=int(high_cluster_flags[i]),
                is_selected=is_selected,
                reputation=helper.get_participant_reputation(client_name),
                trust_instant=trust_instant,
            )
            if helper.experiment_logger.should_log_class_delta_f1(epoch):
                helper.experiment_logger.log_fedcsap_class_delta_f1(
                    epoch=epoch,
                    client_id=client_name,
                    delta_f1_vec=per_client_delta_f1.get(client_name, []),
                )

    if hasattr(helper, 'result_dict') and helper.result_dict is not None:
        helper.result_dict['fedcsap_rep_scores'].append(representative_scores)
        helper.result_dict['fedcsap_selected_clients'].append(selected_clients)
        helper.result_dict['fedcsap_low_cluster_clients'].append(low_cluster_clients)
        helper.result_dict['fedcsap_committee_takeover'].append(bool(committee_takeover))
