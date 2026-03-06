from dataclasses import dataclass


@dataclass
class FLConfig:
    dataset: str = "cifar10"  # cifar10 | pathmnist
    data_dir: str = "./test/data"
    batch_size: int = 64

    # 参考 exp_04_cifar10.sh run_057 的参数（在本独立实现中可映射的部分）
    rounds: int = 200  # 对应 epochs=200
    num_clients: int = 20  # 固定 20（满足需求）
    total_participants: int = 25  # 对应 number_of_total_participants=25
    committee_size: int = 5
    noniid: str = "sampling_dirichlet"
    dirichlet_alpha: float = 0.9
    lr: float = 0.1  # 对应 eta=0.1
    local_epochs: int = 1
    seed: int = 0

    # 攻击设置
    attack: str = "none"  # none | sf
    mal_pcnt: float = 0.3

    # 聚合设置
    aggregation: str = "fedavg"

    # 快速调试
    max_train_samples_per_client: int | None = None
    max_test_samples: int | None = None
    device: str = "cpu"
