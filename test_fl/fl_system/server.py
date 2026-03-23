from __future__ import annotations

import copy
import json
import random
from pathlib import Path

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

    @staticmethod
    def _gradient_cos_similarity(
        grad_a: list[torch.Tensor],
        grad_b: list[torch.Tensor],
        eps: float = 1e-12,
    ) -> float:
        dot = 0.0
        norm_a = 0.0
        norm_b = 0.0
        for ga, gb in zip(grad_a, grad_b):
            ga_flat = ga.view(-1).float()
            gb_flat = gb.view(-1).float()
            dot += float(torch.sum(ga_flat * gb_flat).item())
            norm_a += float(torch.sum(ga_flat * ga_flat).item())
            norm_b += float(torch.sum(gb_flat * gb_flat).item())
        denom = (norm_a ** 0.5) * (norm_b ** 0.5)
        if denom <= eps:
            return 0.0
        return dot / denom

    def _build_targeted_mixed_pseudo_gradient(
        self,
        attack_model,
        target_client_idx: int,
        target_x: torch.Tensor,
        target_y: torch.Tensor,
        num_images: int,
        device,
        focus_alpha: float,
        noise_std: float,
    ) -> list[torch.Tensor]:
        if not (0 <= target_client_idx < len(self.clients)):
            raise IndexError(
                f"target_client_idx 越界: {target_client_idx}, num_clients={len(self.clients)}"
            )
        if len(self.clients) <= 1:
            raise ValueError("至少需要 2 个参与方才能执行 FEDCSAP 混合伪梯度攻击。")

        criterion = torch.nn.CrossEntropyLoss(reduction="mean")

        clamped_alpha = float(max(0.0, min(1.0, focus_alpha)))
        client_pseudo_gradients = []

        for idx, client in enumerate(self.clients):
            if idx == target_client_idx:
                attack_x_i = target_x.to(device)
                attack_y_i = target_y.to(device)
            elif getattr(client, "fixed_batch", False):
                bx, by = client.get_fixed_batch()
                if bx.shape[0] < num_images:
                    raise ValueError(
                        f"Client {client.client_id} fixed_batch 样本不足: "
                        f"required={num_images}, got={bx.shape[0]}"
                    )
                attack_x_i = bx[:num_images].detach().to(device)
                attack_y_i = by[:num_images].detach().to(device).long().view(-1)
            else:
                bx, by = self._collect_attack_batch(client.train_loader, num_images)
                attack_x_i = bx.to(device)
                attack_y_i = by.to(device)

            attack_loss_i = criterion(attack_model(attack_x_i), attack_y_i)
            pseudo_gradient_i = torch.autograd.grad(
                attack_loss_i, attack_model.parameters(), create_graph=False
            )
            client_pseudo_gradients.append([g.detach() for g in pseudo_gradient_i])

        target_gradient = client_pseudo_gradients[target_client_idx]
        mix_part = [torch.zeros_like(g) for g in target_gradient]

        for idx, grad_i in enumerate(client_pseudo_gradients):
            if idx == target_client_idx:
                continue
            cos_sim = self._gradient_cos_similarity(target_gradient, grad_i)
            if cos_sim > 0.0:
                mix_part = [m + cos_sim * g for m, g in zip(mix_part, grad_i)]

        mix_gradient = [
            (1.0 - clamped_alpha) * g_target + clamped_alpha * g_mix
            for g_target, g_mix in zip(target_gradient, mix_part)
        ]
        mix_gradient = [g.detach() for g in mix_gradient]
        if noise_std > 0:
            mix_gradient = [g + torch.randn_like(g) * noise_std for g in mix_gradient]
        return mix_gradient

    def _load_attack_config(self) -> dict:
        attack_num_images = self.cfg.num_images or self.cfg.batch_size
        config_path = Path(self.cfg.attack_config_dir) / f"bs{attack_num_images}.json"
        if not config_path.exists():
            raise FileNotFoundError(
                f"Attack config not found for num_images={attack_num_images}: {config_path}"
            )
        with config_path.open("r", encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def _collect_attack_batch(data_loader, num_images: int):
        xs, ys = [], []
        total = 0
        for x, y in data_loader:
            if total >= num_images:
                break
            need = num_images - total
            x = x[:need]
            y = y[:need]
            xs.append(x.detach().cpu())
            ys.append(y.detach().cpu().long().view(-1))
            total += x.shape[0]
        if total < num_images:
            raise ValueError(
                f"Not enough samples to build attack batch: required={num_images}, got={total}."
            )
        return torch.cat(xs, dim=0), torch.cat(ys, dim=0)

    def train(self):
        attack_num_images = self.cfg.num_images or self.cfg.batch_size

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
                # 按 num_images 读取对应的攻击配置（bs1/bs4/bs16...）
                config = self._load_attack_config()
                print(config)
                target_client_idx = 1
                # 确保 inversefed 的模型、伪梯度和标准化统计量处于同一设备
                self.model.to(self.cfg.device)
                attack_device = next(self.model.parameters()).device
                os.makedirs("attack_results", exist_ok=True)

                # 在执行 inversefed 之前，先保存目标客户端的一批真实训练图像用于对比
                if getattr(self.clients[target_client_idx], "fixed_batch", False):
                    target_x, target_y = self.clients[target_client_idx].get_fixed_batch()
                    if target_x.shape[0] < attack_num_images:
                        raise ValueError(
                            "fixed_batch 模式下，攻击样本数(num_images)不能大于训练 batch_size。"
                            f" 当前 fixed batch 仅有 {target_x.shape[0]} 张，请增大 --batch-size，"
                            "或者关闭 --fixed-batch。"
                        )
                    target_x = target_x[:attack_num_images].detach().cpu()
                    target_y = target_y[:attack_num_images].detach().cpu().long().view(-1)
                else:
                    target_loader = self.clients[target_client_idx].train_loader
                    target_x, target_y = self._collect_attack_batch(target_loader, attack_num_images)
                target_save_path = (
                    f"attack_results/target_client{self.clients[target_client_idx].client_id}_tv{config['total_variation']}"
                    f"_bs{attack_num_images}_seed{self.cfg.seed}.png"
                )
                torchvision.utils.save_image(
                    target_x,
                    target_save_path,
                    nrow=max(1, int(attack_num_images**0.5)),
                )
                print(f"[*] 已保存目标客户端真实训练图像至 {target_save_path}")

                # 提取攻击输入：
                # - FEDCSAP 对比方案：使用“混合后更新”构造反演输入；
                # - 其他方案：保持原有基线方式，使用目标客户端首 batch 在全局模型上的真实梯度。
                alpha = self.cfg.fedcsap_hybrid_alpha if self.cfg.aggregation == "fedcsap" else 1.0
                noise_std = self.cfg.gaussian_noise_std
                attack_model = copy.deepcopy(self.model).to(attack_device)
                attack_model.load_state_dict(global_state)
                attack_model.eval()

                attack_y = target_y.to(attack_device)
                if self.cfg.aggregation == "fedcsap":
                    # FedCSAP 攻击输入构造：
                    # 1) 每个参与方按 baseline 方式计算 pseudo_gradient；
                    # 2) 计算其他参与方与受害者 pseudo_gradient 的 cos 相似度；
                    # 3) cos>0 时累加 cos * grad_i 得到 mix_part；
                    # 4) mix_gradient = grad_victim*(1-alpha) + alpha*mix_part；
                    # 5) 像 baseline 一样可加入高斯噪声后用于反演攻击。
                    pseudo_gradient = self._build_targeted_mixed_pseudo_gradient(
                        attack_model=attack_model,
                        target_client_idx=target_client_idx,
                        target_x=target_x,
                        target_y=target_y,
                        num_images=attack_num_images,
                        device=attack_device,
                        focus_alpha=alpha,
                        noise_std=noise_std,
                    )
                    attack_mode = "Cosine-targeted-mixed"
                else:
                    attack_x = target_x.to(attack_device)
                    attack_loss = torch.nn.CrossEntropyLoss(reduction="mean")(attack_model(attack_x), attack_y)
                    pseudo_gradient = torch.autograd.grad(
                        attack_loss, attack_model.parameters(), create_graph=False
                    )
                    pseudo_gradient = [g.detach() for g in pseudo_gradient]
                    if noise_std > 0:
                        pseudo_gradient = [g + torch.randn_like(g) * noise_std for g in pseudo_gradient]
                    attack_mode = "Baseline"

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
                print(config['total_variation'])
                print(f"\n[*] 正在对 Client {self.clients[target_client_idx].client_id} 执行梯度反转攻击 ({attack_mode})...")
                print(f"[*] 参数配置: Num Images={attack_num_images}, Alpha={alpha}, Noise={noise_std}")
                
                # 设置重建的图像数量严格等于 num_images（默认跟随 batch_size）
                rec_machine = GradientReconstructor(attack_model, (dm, ds), config, num_images=attack_num_images)
                
                # 执行重建计算
                output, stats = rec_machine.reconstruct(pseudo_gradient, attack_y, img_shape=(3, 32, 32))
                
                # 保存重建图像（从模型输入域反归一化到可视化域 [0, 1]）
                output_vis = torch.clamp(output.detach().cpu() * ds.detach().cpu() + dm.detach().cpu(), 0.0, 1.0)
                save_path = (
                    "attack_results/"
                    f"recon_bs{attack_num_images}_alpha{alpha}_noise{noise_std}_"
                    f"client{target_client_idx}_tv{config['total_variation']}_seed{self.cfg.seed}.png"
                )
                torchvision.utils.save_image(output_vis, save_path, nrow=int(attack_num_images**0.5))

                # 保存重建对应标签，便于后续图像-标签对齐分析
                with torch.no_grad():
                    recon_logits = attack_model(output.to(attack_device))
                    recon_pred_labels = recon_logits.argmax(dim=1).detach().cpu().tolist()
                label_save_path = (
                    "attack_results/"
                    f"recon_bs{attack_num_images}_alpha{alpha}_noise{noise_std}_"
                    f"tv{config['total_variation']}_seed{self.cfg.seed}_labels.json"
                )
                label_payload = {
                    "target_client_id": int(self.clients[target_client_idx].client_id),
                    "num_images": int(attack_num_images),
                    "dataset": self.cfg.dataset,
                    "ground_truth_labels": attack_y.detach().cpu().tolist(),
                    "reconstructed_pred_labels": recon_pred_labels,
                }
                with open(label_save_path, "w", encoding="utf-8") as f:
                    json.dump(label_payload, f, ensure_ascii=False, indent=2)

                summary_save_path = (
                    "attack_results/"
                    f"attack_summary_seed{self.cfg.seed}_agg{self.cfg.aggregation}_"
                    f"alpha{alpha}_noise{noise_std}_bs{attack_num_images}.json"
                )
                summary_payload = {
                    "seed": int(self.cfg.seed),
                    "aggregation": self.cfg.aggregation,
                    "num_clients": int(self.cfg.num_clients),
                    "fedcsap_hybrid_alpha": float(alpha),
                    "gaussian_noise_std": float(noise_std),
                    "num_images": int(attack_num_images),
                    "target_client_id": int(self.clients[target_client_idx].client_id),
                    "target_image_path": target_save_path,
                    "reconstruction_path": save_path,
                    "label_path": label_save_path,
                    "reconstruction_loss_opt": float(stats["opt"]),
                }
                with open(summary_save_path, "w", encoding="utf-8") as f:
                    json.dump(summary_payload, f, ensure_ascii=False, indent=2)

                print(
                    f"[*] 攻击完成，图像已保存至 {save_path}，标签已保存至 {label_save_path}，"
                    f"汇总已保存至 {summary_save_path}。"
                    f"最优损失: {stats['opt']: .4f}\n"
                )
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
