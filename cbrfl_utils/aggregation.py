import logging
import time
import random

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader

from config import device

logger = logging.getLogger('logger')


def _param_float(params, key, default):
    value = params.get(key, default)
    if value is None:
        return float(default)
    return float(value)


def _compute_mean_loss(helper, model, loader):
    if loader is None:
        return 0.0

    model.eval()
    criterion = torch.nn.CrossEntropyLoss(reduction='mean')
    loss_sum = 0.0
    sample_count = 0

    with torch.no_grad():
        for batch in loader:
            if len(batch) < 2:
                continue
            data, targets = batch[0], batch[1]
            data = data.to(device)
            targets = targets.to(device)
            outputs = model(data)
            batch_loss = criterion(outputs, targets)
            bs = int(targets.shape[0])
            loss_sum += float(batch_loss.item()) * bs
            sample_count += bs

    if sample_count <= 0:
        return 0.0
    return float(loss_sum / sample_count)


def _apply_update_to_model(helper, base_model, update_dict):
    model = helper.new_model()
    model.copy_params(base_model.state_dict())
    for layer_name, layer_data in model.state_dict().items():
        upd = update_dict[layer_name]
        try:
            layer_data.add_(upd)
        except Exception:
            layer_data.add_(upd.to(layer_data.dtype))
    return model


def _flatten_update(helper, update_dict):
    flat = np.array(helper.flatten_gradient_v2(update_dict), dtype=np.float64)
    if not np.all(np.isfinite(flat)):
        flat = np.nan_to_num(flat, nan=0.0, posinf=0.0, neginf=0.0)
    return flat


def _gompertz_weight(u_j, a, b, c):
    # monotonic stable form
    x = -c * float(u_j)
    x = np.clip(x, -30.0, 30.0)
    return float(a * np.exp(-b * np.exp(x)))


def _weighted_sample_without_replacement(items, weight_map, k, rng):
    items = list(items)
    k = max(0, min(int(k), len(items)))
    if k == 0:
        return []
    chosen = []
    pool = list(items)
    while len(chosen) < k and len(pool) > 0:
        weights = np.array([max(float(weight_map.get(i, 0.0)), 0.0) for i in pool], dtype=np.float64)
        if float(np.sum(weights)) <= 0:
            pick_idx = rng.randrange(len(pool))
        else:
            probs = weights / np.sum(weights)
            pick_idx = int(np.random.default_rng(rng.randrange(1 << 30)).choice(len(pool), p=probs))
        chosen.append(pool.pop(pick_idx))
    return chosen


def _build_shared_val_union_loader(helper, committee_client_id):
    if not hasattr(helper, 'cbrfl_shared_val_dataset') or helper.cbrfl_shared_val_dataset is None:
        return None
    if not hasattr(helper, 'val_data') or helper.val_data is None:
        return None
    if committee_client_id >= len(helper.val_data):
        return None

    local_loader = helper.val_data[committee_client_id]
    if local_loader is None or not hasattr(local_loader, 'dataset'):
        return None

    dataset_union = ConcatDataset([local_loader.dataset, helper.cbrfl_shared_val_dataset])
    return DataLoader(dataset_union, batch_size=helper.params['test_batch_size'], shuffle=False)


def run_cbrfl(helper, target_model, updates, epoch, committee_members=None):
    start = time.time()
    helper.ensure_cbrfl_state()

    names = list(updates.keys())
    delta_by_name = {name: updates[name][2] for name in names}
    if len(names) == 0:
        logger.warning('No updates for CBRFL at epoch %s.', epoch)
        return

    committee_members = list(committee_members or [])
    committee_members = [c for c in committee_members if c in helper.participants_list]
    if len(committee_members) == 0:
        logger.warning('CBRFL epoch %s has empty committee, fallback to current trainers.', epoch)
        committee_members = list(names)

    # Step 1-3: evaluate marginal utility by committee members.
    mean_all = helper.weighted_average_oracle([delta_by_name[n] for n in names], torch.ones(len(names), dtype=torch.float32))
    norm_all = float(np.linalg.norm(_flatten_update(helper, mean_all)))

    utility_by_committee = {cm: {} for cm in committee_members}
    for cm in committee_members:
        val_loader = _build_shared_val_union_loader(helper, cm)
        if val_loader is None:
            continue

        model_all = _apply_update_to_model(helper, target_model, mean_all)
        loss_all = _compute_mean_loss(helper, model_all, val_loader)

        for j_name in names:
            other_names = [n for n in names if n != j_name]
            if len(other_names) == 0:
                mean_without_j = {k: torch.zeros_like(v) for k, v in mean_all.items()}
            else:
                mean_without_j = helper.weighted_average_oracle(
                    [delta_by_name[n] for n in other_names],
                    torch.ones(len(other_names), dtype=torch.float32),
                )
            model_minus_j = _apply_update_to_model(helper, target_model, mean_without_j)
            loss_minus_j = _compute_mean_loss(helper, model_minus_j, val_loader)

            norm_j = float(np.linalg.norm(_flatten_update(helper, delta_by_name[j_name])))
            utility = ((loss_minus_j - loss_all) / max(loss_all, 1e-12)) * (norm_all / max(norm_j, 1e-12))
            if not np.isfinite(utility):
                utility = 0.0
            utility_by_committee[cm][j_name] = float(utility)

    # Step 4: OCC-like consensus voting by committee majorities.
    inclusion_count = {name: 0 for name in names}
    delta_threshold = _param_float(helper.params, 'cbrfl_delta', 0.0)
    for cm in committee_members:
        utilities = utility_by_committee.get(cm, {})
        acceptable = [n for n in names if float(utilities.get(n, 0.0)) > delta_threshold]
        for n in acceptable:
            inclusion_count[n] += 1

    majority = (len(committee_members) // 2) + 1
    selected_clients = [n for n, cnt in inclusion_count.items() if cnt >= majority]
    median_utility_dict = {}
    for n in names:
        vals = [utility_by_committee.get(cm, {}).get(n, 0.0) for cm in committee_members]
        median_utility_dict[n] = float(np.median(np.array(vals, dtype=np.float32))) if len(vals) > 0 else 0.0

    if len(selected_clients) == 0:
        selected_clients = [n for n in names if median_utility_dict[n] > 0.0]
    if len(selected_clients) == 0:
        selected_clients = [max(names, key=lambda n: median_utility_dict[n])]

    # Step 5: adaptive weighted aggregation.
    g_a = _param_float(helper.params, 'cbrfl_gompertz_a', 1.0)
    g_b = _param_float(helper.params, 'cbrfl_gompertz_b', 1.0)
    g_c = _param_float(helper.params, 'cbrfl_gompertz_c', 1.0)
    impact_dict = {n: _gompertz_weight(median_utility_dict[n], g_a, g_b, g_c) for n in selected_clients}

    weights = torch.tensor([impact_dict[n] for n in selected_clients], dtype=torch.float32)
    if float(torch.sum(weights).item()) <= 0:
        weights = torch.ones(len(selected_clients), dtype=torch.float32)
    agg_update = helper.weighted_average_oracle([delta_by_name[n] for n in selected_clients], weights)

    # Momentum
    beta = _param_float(helper.params, 'cbrfl_beta', 0.5)
    if helper.cbrfl_momentum is None:
        m_t = agg_update
    else:
        m_t = {}
        for layer_name in agg_update.keys():
            m_t[layer_name] = beta * helper.cbrfl_momentum[layer_name] + (1.0 - beta) * agg_update[layer_name]
    helper.cbrfl_momentum = m_t

    # Adaptive global learning rate
    if helper.cbrfl_global_lr is None:
        helper.cbrfl_global_lr = _param_float(helper.params, 'cbrfl_global_lr_init', 1.0)
    eta_t = helper.cbrfl_global_lr
    if helper.cbrfl_prev_momentum is not None:
        curr_flat = _flatten_update(helper, m_t)
        prev_flat = _flatten_update(helper, helper.cbrfl_prev_momentum)
        dot_val = float(np.dot(curr_flat, prev_flat))
        dot_clip = _param_float(helper.params, 'cbrfl_global_lr_dot_clip', 0.1)
        dot_val = float(np.clip(dot_val, -dot_clip, dot_clip))
        eta_t = eta_t + dot_val
    eta_t = float(np.clip(
        eta_t,
        _param_float(helper.params, 'cbrfl_global_lr_min', 1e-4),
        _param_float(helper.params, 'cbrfl_global_lr_max', 5.0),
    ))
    helper.cbrfl_global_lr = eta_t
    helper.cbrfl_prev_momentum = {k: v.clone().detach() for k, v in m_t.items()}

    for layer_name, layer_data in target_model.state_dict().items():
        update_per_layer = m_t[layer_name] * eta_t
        try:
            layer_data.add_(update_per_layer)
        except Exception:
            layer_data.add_(update_per_layer.to(layer_data.dtype))

    # Step 6: contribution and reputation updates.
    train_contrib = {n: 0.0 for n in helper.participants_list}
    for n in selected_clients:
        train_contrib[n] = eta_t * float(impact_dict.get(n, 0.0))

    val_contrib = {i: 0.0 for i in helper.participants_list}
    selected_set = set(selected_clients)
    primary_committee = random.Random(helper.params['seed'] + int(epoch)).choice(committee_members)
    for cm in committee_members:
        cm_util = utility_by_committee.get(cm, {})
        ac_i = [n for n in names if float(cm_util.get(n, 0.0)) > delta_threshold]
        inter = len(set(ac_i) & selected_set)
        union = max(1, len(set(ac_i) | selected_set))
        iou = float(inter / union)
        contrib = eta_t * iou
        if cm == primary_committee:
            contrib *= 2.0
        val_contrib[cm] = contrib

    lam = _param_float(helper.params, 'cbrfl_lambda', 0.5)
    for p in helper.participants_list:
        helper.cbrfl_train_reputation[p] = lam * float(train_contrib.get(p, 0.0)) + (1.0 - lam) * float(helper.cbrfl_train_reputation[p])
        helper.cbrfl_val_reputation[p] = lam * float(val_contrib.get(p, 0.0)) + (1.0 - lam) * float(helper.cbrfl_val_reputation[p])

    helper.cbrfl_prev_ag_clients = list(selected_clients)
    helper.cbrfl_prev_training_clients = list(names)
    helper.cbrfl_prev_committee = list(committee_members)

    helper.result_dict['cbrfl_selected_clients'].append(list(selected_clients))
    helper.result_dict['cbrfl_committee'].append(list(committee_members))
    helper.result_dict['cbrfl_primary_committee'].append(primary_committee)
    helper.result_dict['cbrfl_median_utilities'].append(dict(median_utility_dict))
    helper.result_dict['cbrfl_impact_weights'].append(dict(impact_dict))
    helper.result_dict['cbrfl_train_reputation'].append(dict(helper.cbrfl_train_reputation))
    helper.result_dict['cbrfl_val_reputation'].append(dict(helper.cbrfl_val_reputation))
    helper.result_dict['cbrfl_global_lr'].append(float(helper.cbrfl_global_lr))
    helper.result_dict['cbrfl_consensus_success'].append(True)

    logger.info(
        'CBRFL epoch %s: committee=%s training=%s selected=%s eta=%.6f elapsed=%.3fs',
        epoch,
        committee_members,
        names,
        selected_clients,
        eta_t,
        time.time() - start,
    )
