from __future__ import annotations

import random

import torch

from .aggregators import build_aggregator
from .client import FLClient

from .inversefed.reconstruction_algorithms import GradientReconstructor
import torchvision
import os


class FLServer:
    def __init__(self, model, clients: list[FLClient], test_loader, cfg):
        self.model = model
        self.clients = clients
        self.test_loader = test_loader
        self.cfg = cfg
        self.aggregator = build_aggregator(cfg.aggregation)

        rng = random.Random(cfg.seed)
        mal_count = int(len(clients) * cfg.mal_pcnt)
        self.malicious_set = set(rng.sample(range(len(clients)), mal_count))

    @staticmethod
    def _apply_gaussian_noise(state: dict, noise_std: float) -> dict:
        if noise_std <= 0:
            return state
        noised_state = {}
        for k, v in state.items():
            if torch.is_floating_point(v):
                noised_state[k] = v + torch.randn_like(v) * noise_std
            else:
                noised_state[k] = v
        return noised_state

    @staticmethod
    def _hybrid_update(global_state: dict, agg_state: dict, alpha: float) -> dict:
        if not 0.0 < alpha <= 1.0:
            raise ValueError("fedcsap_hybrid_alpha must be in (0, 1].")
        if alpha == 1.0:
            return agg_state
        out = {}
        for k, global_tensor in global_state.items():
            out[k] = global_tensor + alpha * (agg_state[k] - global_tensor)
        return out

    def train(self):

        for rnd in range(1, self.cfg.rounds + 1):
            client_states = []
            client_weights = []
            
            # 记录本轮全局状态
            global_state = {
                k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()
            }
            
            for c in self.clients:
                state, weight = c.local_train(
                    self.model,
                    self.cfg.lr,
                    self.cfg.local_epochs,
                    self.cfg.momentum,
                    self.cfg.weight_decay,
                )
                attacked = c.maybe_attack(
                    state,
                    global_state,
                    self.cfg.attack,
                    c.client_id in self.malicious_set,
                )
                client_states.append(attacked)
                client_weights.append(weight)

            # 聚合与防御阶段
            print("进入聚合阶段")
            avg_state = self.aggregator.aggregate(client_states, client_weights)

            if self.cfg.aggregation == "fedcsap":
                print("进入fedcsap阶段")
                next_state = self._hybrid_update(global_state, avg_state, self.cfg.fedcsap_hybrid_alpha)
            else:
                next_state = avg_state

            next_state = self._apply_gaussian_noise(next_state, self.cfg.gaussian_noise_std)
            self.model.load_state_dict(next_state)

            # ==========================================
            # 🚀 隐私窃取：梯度反转攻击 (基线验证模式)
            # ==========================================
            # 仅在第一轮执行攻击，以验证初始阶段的隐私泄漏情况
            if rnd == 1:
                target_client_idx = 0
                target_state = client_states[target_client_idx]
                pseudo_gradient = []
                # 确保 inversefed 的模型、伪梯度和标准化统计量处于同一设备
                self.model.to(self.cfg.device)
                attack_device = next(self.model.parameters()).device
                os.makedirs("attack_results", exist_ok=True)

                # 在执行 inversefed 之前，先保存目标客户端的一批真实训练图像用于对比
                target_loader = self.clients[target_client_idx].train_loader
                target_x, _ = next(iter(target_loader))
                target_x = target_x[: self.cfg.batch_size].detach().cpu()
                target_save_path = (
                    f"attack_results/target_client{self.clients[target_client_idx].client_id}"
                    f"_bs{self.cfg.batch_size}.png"
                )
                torchvision.utils.save_image(
                    target_x,
                    target_save_path,
                    nrow=max(1, int(self.cfg.batch_size**0.5)),
                )
                print(f"[*] 已保存目标客户端真实训练图像至 {target_save_path}")

                # 提取目标客户端防御后的等效梯度
                alpha = self.cfg.fedcsap_hybrid_alpha if self.cfg.aggregation == "fedcsap" else 1.0
                noise_std = self.cfg.gaussian_noise_std
                for name, _ in self.model.named_parameters():
                    # 仅使用可训练参数来构造梯度，顺序必须与 model.parameters() 一致；
                    # 否则会与 inversefed 内部 autograd 梯度列表错位，触发 shape mismatch。
                    global_param = global_state[name]
                    client_param = target_state[name]
                    # 模拟窃听者计算：防御衰减与噪声叠加
                    grad_tensor = global_param - client_param
                    grad_tensor = global_param - client_param
                    grad_tensor = alpha * grad_tensor
                    if noise_std > 0:
                        grad_tensor += torch.randn_like(grad_tensor) * noise_std
                    pseudo_gradient.append(grad_tensor.to(attack_device))

                # CIFAR-10 数据集统计特征 (需与 datasets.py 中 Normalize 一致)
                dm = torch.as_tensor([0.4914, 0.4822, 0.4465], device=attack_device)[:, None, None]
                ds = torch.as_tensor([0.2023, 0.1994, 0.2010], device=attack_device)[:, None, None]

                # 优化器配置：使用余弦相似度损失，针对小 Batch Size 进行 L-BFGS 或 Adam 优化
                config = dict(signed=True,
                              cost_fn='sim', 
                              indices='def', 
                              weights='equal',
                              lr=0.1, 
                              optim='adam', 
                              restarts=2,          # 基线测试可设为 2 提高稳定性
                              max_iterations=4800, 
                              total_variation=1e-1,
                              init='randn',
                              filter='none',
                              lr_decay=True,
                              scoring_choice='loss')

                print(f"\n[*] 正在对 Client {self.clients[target_client_idx].client_id} 执行梯度反转攻击 (Baseline)...")
                print(f"[*] 参数配置: Batch Size={self.cfg.batch_size}, Alpha={alpha}, Noise={noise_std}")
                
                # 设置重建的图像数量严格等于 batch_size
                rec_machine = GradientReconstructor(self.model, (dm, ds), config, num_images=self.cfg.batch_size)
                
                # 执行重建计算
                output, stats = rec_machine.reconstruct(pseudo_gradient, None, img_shape=(3, 32, 32))
                
                # 保存重建图像
                save_path = f"attack_results/recon_bs{self.cfg.batch_size}_alpha{alpha}_noise{noise_std}.png"
                torchvision.utils.save_image(output, save_path, nrow=int(self.cfg.batch_size**0.5))
                
                print(f"[*] 攻击完成，图像已保存至 {save_path}。最优损失: {stats['opt']: .4f}\n")
            # ==========================================

            acc = self.evaluate()
            print(f"[Round {rnd:03d}/{self.cfg.rounds}] test_acc={acc:.4f}")

    @torch.no_grad()
    def evaluate(self):
        self.model.eval()
        correct = 0
        total = 0
        device = self.cfg.device
        self.model.to(device)
        for x, y in self.test_loader:
            x = x.to(device)
            y = y.to(device).long().view(-1)
            logits = self.model(x)
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.numel()
        return correct / max(total, 1)
