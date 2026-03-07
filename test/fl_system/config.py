from dataclasses import dataclass


@dataclass
class FLConfig:
    dataset: str = "cifar10"  # cifar10 | pathmnist
    data_dir: str = "./data"
    batch_size: int = 64

    # 参考 exp_04_cifar10.sh run_057 的参数（在本独立实现中可映射的部分）
    rounds: int = 200  # 对应 epochs=200
    num_clients: int = 20  # 固定 20（满足需求）
    total_participants: int = 25  # 对应 number_of_total_participants=25
    committee_size: int = 5
    noniid: str = "sampling_dirichlet"
    dirichlet_alpha: float = 0.9
    lr: float = 0.1  # 对应 jinja 中的 lr（本地优化器学习率）
    eta: float = 0.1  # 对应系统参数 eta（服务器端全局更新步长）
    momentum: float = 0.9
    weight_decay: float = 5e-4
    local_epochs: int = 2  # 对应 cifar 场景 internal_epochs=2
    seed: int = 0

    # 攻击设置
    attack: str = "sf"  # none | sf
    mal_pcnt: float = 0.3

    # 聚合设置
    aggregation: str = "fedavg"

    # 快速调试
    max_train_samples_per_client: int | None = None
    max_test_samples: int | None = None
    device: str = "cpu"
