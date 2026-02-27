import logging
import time
import copy
import numpy as np
import torch

from sklearn.cluster import KMeans

import config
from fedcsap_utils.validation_test import validation_test

logger = logging.getLogger('logger')


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

    representative_scores = []
    per_client_delta_f1 = {}
    for idx, rep_name in enumerate(names):
        rep_model = helper.new_model()
        rep_model.copy_params(helper.target_model.state_dict())

        rep_update = delta_models[idx]
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
    selected_indices = [i for i, flag in enumerate(high_cluster_flags) if flag == 1]

    if len(selected_indices) == 0:
        selected_indices = [int(np.argmax(representative_scores))]

    selected_clients = [names[i] for i in selected_indices]
    low_cluster_clients = [names[i] for i in range(len(names)) if i not in selected_indices]
    helper.update_participant_reputation(low_cluster_clients, names)

    # map representative -> original client update, then average aggregation (flshield-consistent weighted average)
    weight_vec = np.zeros(len(names), dtype=np.float32)
    for i in selected_indices:
        weight_vec[i] = 1.0 / len(selected_indices)

    aggregate_weights = helper.weighted_average_oracle(delta_models, torch.tensor(weight_vec))
    for layer_name, layer_data in target_model.state_dict().items():
        update_per_layer = aggregate_weights[layer_name] * helper.params['eta']
        try:
            layer_data.add_(update_per_layer)
        except Exception:
            layer_data.add_(update_per_layer.to(layer_data.dtype))

    logger.info(
        'fedcsap epoch %s: validators=%s, rep_scores=%s, selected_clients=%s, low_cluster_clients=%s, elapsed=%.2fs',
        epoch,
        validators,
        [round(s, 6) for s in representative_scores],
        selected_clients,
        low_cluster_clients,
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
