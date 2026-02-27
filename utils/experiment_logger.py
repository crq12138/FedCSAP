import csv
import os
from typing import Iterable


class ExperimentLogger:
    def __init__(self, folder_path, total_epochs, aggr_epoch_interval=1):
        self.folder_path = folder_path
        self.total_epochs = int(total_epochs)
        self.aggr_epoch_interval = int(aggr_epoch_interval) if aggr_epoch_interval else 1

        self.global_metrics_path = os.path.join(folder_path, 'global_metrics.csv')
        self.fedcsap_client_metrics_path = os.path.join(folder_path, 'fedcsap_client_metrics.csv')
        self.fedcsap_round_metrics_path = os.path.join(folder_path, 'fedcsap_round_metrics.csv')
        self.fedcsap_class_delta_f1_path = os.path.join(folder_path, 'fedcsap_class_delta_f1.csv')

        self.class_vector_epochs = self._build_class_vector_epochs()
        self._init_files()

    def _build_class_vector_epochs(self):
        candidates = {
            max(1, self.total_epochs // 4),
            max(1, self.total_epochs // 2),
            max(1, (self.total_epochs * 3) // 4),
            self.total_epochs,
        }
        return candidates

    def _init_files(self):
        os.makedirs(self.folder_path, exist_ok=True)
        with open(self.global_metrics_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['epoch', 'global_acc', 'global_macro_f1'])

        with open(self.fedcsap_client_metrics_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'epoch', 'client_id', 'is_malicious', 'cvar_score', 'cluster_label', 'pass',
                'R', 'trust_instant'
            ])

        with open(self.fedcsap_round_metrics_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'epoch', 'committee_size', 'committee_malicious_count',
                'committee_mal_ratio', 'committee_takeover'
            ])

        with open(self.fedcsap_class_delta_f1_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['epoch', 'client_id', 'class_id', 'delta_f1'])

    def log_global_metrics(self, epoch, global_acc, global_macro_f1):
        with open(self.global_metrics_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch, global_acc, global_macro_f1])

    def log_fedcsap_round_metrics(self, epoch, committee_size, committee_malicious_count):
        committee_size = int(committee_size)
        committee_malicious_count = int(committee_malicious_count)
        committee_mal_ratio = (committee_malicious_count / committee_size) if committee_size > 0 else 0.0
        committee_takeover = 1 if committee_malicious_count > (committee_size / 2.0) else 0
        with open(self.fedcsap_round_metrics_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch, committee_size, committee_malicious_count, committee_mal_ratio, committee_takeover
            ])

    def log_fedcsap_client_metrics(
        self,
        epoch,
        client_id,
        is_malicious,
        cvar_score,
        cluster_label,
        is_selected,
        reputation,
        trust_instant,
    ):
        with open(self.fedcsap_client_metrics_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch,
                client_id,
                int(bool(is_malicious)),
                cvar_score,
                cluster_label,
                int(bool(is_selected)),
                reputation,
                trust_instant,
            ])

    def should_log_class_delta_f1(self, epoch):
        return int(epoch) in self.class_vector_epochs

    def log_fedcsap_class_delta_f1(self, epoch, client_id, delta_f1_vec: Iterable[float]):
        with open(self.fedcsap_class_delta_f1_path, 'a', newline='') as f:
            writer = csv.writer(f)
            for class_id, val in enumerate(delta_f1_vec):
                writer.writerow([epoch, client_id, class_id, val])
