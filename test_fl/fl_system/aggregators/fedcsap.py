from __future__ import annotations

from .fedavg import FedAvgAggregator


class FedCSAPAggregator(FedAvgAggregator):
    """Baseline FEDCSAP aggregator for test_fl.

    在 test_fl 中先复用 FedAvg 聚合核，随后由 server 侧执行
    FEDCSAP 混合更新（hybrid update）与可选高斯噪声。
    """

    pass
