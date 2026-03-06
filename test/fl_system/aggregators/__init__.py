from .base import Aggregator
from .fedavg import FedAvgAggregator


def build_aggregator(name: str) -> Aggregator:
    name = name.lower()
    registry = {
        "fedavg": FedAvgAggregator,
    }
    if name not in registry:
        raise ValueError(f"Unsupported aggregator: {name}. Available: {list(registry)}")
    return registry[name]()
