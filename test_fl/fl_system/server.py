from __future__ import annotations

import copy
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
                # 确保 inversefed 的模型、伪梯度和标准化统计量处于同一设备
                self.model.to(self.cfg.device)
                attack_device = next(self.model.parameters()).device
                os.makedirs("attack_results", exist_ok=True)

                # 在执行 inversefed 之前，先保存目标客户端的一批真实训练图像用于对比
                target_loader = self.clients[target_client_idx].train_loader
                target_x, target_y = next(iter(target_loader))
                target_x = target_x[: self.cfg.batch_size].detach().cpu()
                target_y = target_y[: self.cfg.batch_size].detach().cpu().long().view(-1)
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

                # 提取目标梯度（用于重建链路正确性验证）
                # 注意：客户端上传的是“多步本地训练后的参数”，并非单批次原始梯度。
                # 直接用参数差做反演会对应“多样本混合信号”，常见现象就是灰图/均值图。
                # 这里使用目标客户端首个 batch 在全局模型上的真实梯度做基线排查。
                alpha = self.cfg.fedcsap_hybrid_alpha if self.cfg.aggregation == "fedcsap" else 1.0
                noise_std = self.cfg.gaussian_noise_std
                attack_model = copy.deepcopy(self.model).to(attack_device)
                attack_model.load_state_dict(global_state)
                attack_model.eval()

                attack_x = target_x.to(attack_device)
                attack_y = target_y.to(attack_device)
                attack_loss = torch.nn.CrossEntropyLoss(reduction="mean")(attack_model(attack_x), attack_y)
                pseudo_gradient = torch.autograd.grad(attack_loss, attack_model.parameters(), create_graph=False)
                pseudo_gradient = [g.detach() for g in pseudo_gradient]
                if noise_std > 0:
                    pseudo_gradient = [g + torch.randn_like(g) * noise_std for g in pseudo_gradient]

                # 规范化统计量必须与 datasets.py 的预处理保持一致：
                # - CIFAR10: 仅 ToTensor, 无 Normalize -> mean=0, std=1
                # - MNIST/PATHMNIST: 与对应 Normalize 参数保持一致
                if self.cfg.dataset == "cifar10":
                    dm = torch.zeros(3, device=attack_device)[:, None, None]
                    ds = torch.ones(3, device=attack_device)[:, None, None]
                elif self.cfg.dataset == "mnist":
                    dm = torch.as_tensor([0.1307], device=attack_device)[:, None, None]
                    ds = torch.as_tensor([0.3081], device=attack_device)[:, None, None]
                elif self.cfg.dataset == "pathmnist":
                    dm = torch.as_tensor([0.5, 0.5, 0.5], device=attack_device)[:, None, None]
                    ds = torch.as_tensor([0.5, 0.5, 0.5], device=attack_device)[:, None, None]
                else:
                    raise ValueError(f"Unsupported dataset for attack normalization: {self.cfg.dataset}")

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
                rec_machine = GradientReconstructor(attack_model, (dm, ds), config, num_images=self.cfg.batch_size)
                
                # 执行重建计算
                output, stats = rec_machine.reconstruct(pseudo_gradient, attack_y, img_shape=(3, 32, 32))
                
                # 保存重建图像（从模型输入域反归一化到可视化域 [0, 1]）
                output_vis = torch.clamp(output.detach().cpu() * ds.detach().cpu() + dm.detach().cpu(), 0.0, 1.0)
                save_path = f"attack_results/recon_bs{self.cfg.batch_size}_alpha{alpha}_noise{noise_std}.png"
                torchvision.utils.save_image(output_vis, save_path, nrow=int(self.cfg.batch_size**0.5))
                
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
