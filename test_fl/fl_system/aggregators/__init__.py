from .base import Aggregator
from .fedavg import FedAvgAggregator
from .fedcsap import FedCSAPAggregator


def build_aggregator(name: str) -> Aggregator:
    name = name.lower()
    registry = {
        "fedavg": FedAvgAggregator,
        "fedcsap": FedCSAPAggregator,
    }
    if name not in registry:
        raise ValueError(f"Unsupported aggregator: {name}. Available: {list(registry)}")
    return registry[name]()
