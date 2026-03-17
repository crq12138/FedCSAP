from __future__ import annotations

import torch

from .base import Aggregator

class FedAvgAggregator(Aggregator):
    def aggregate(self, client_states: list[dict], client_weights: list[float]) -> dict:
        total_weight = sum(client_weights)
        if total_weight <= 0:
            raise ValueError("Total client weight must be positive.")

        out = {}
        keys = client_states[0].keys()
        for k in keys:
            # 1. 记录原始张量的数据类型
            original_dtype = client_states[0][k].dtype
            
            # 2. 统一使用高精度的 float32 甚至 float64 初始化累加器，避免累加过程中的溢出或截断
            acc = torch.zeros_like(client_states[0][k], dtype=torch.float32)
            
            for state, w in zip(client_states, client_weights):
                # 3. 将本地张量隐式转换为 float 进行代数加权运算
                acc += state[k].to(torch.float32) * (w / total_weight)
                
            # 4. 运算完毕后，将结果恢复为网络结构定义的原始类型（如 Long）
            out[k] = acc.to(original_dtype)
            
        return out