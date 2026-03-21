# Standalone Federated Learning System (from scratch)

这是一个**完全独立于仓库现有训练代码**的新实现，代码全部位于 `test_fl/fl_system/`。

## 当前能力

1. 客户端数量固定为 20（`--num-clients` 必须等于 20）。
2. 支持 CIFAR10 / MNIST / PATHMNIST。
3. CIFAR10 对齐主系统 FEDCSAP：
   - 数据目录默认 `./data`（与主系统一致）
   - 模型使用 `ResNet18`
   - 训练超参默认值保持一致：`lr=0.1, momentum=0.9, weight_decay=5e-4, local_epochs=2`
4. 聚合支持 `fedavg` 与 `fedcsap`（test_fl 中的 fedcsap 复用聚合核，并在 server 侧实现 FEDCSAP 扩展逻辑）。
5. 新增可控高斯噪声：`--gaussian-noise-std`。
6. 新增 FEDCSAP 混合更新：`--fedcsap-hybrid-alpha`。

## 依赖

- `torch`, `torchvision`
- `medmnist`（仅 PATHMNIST 需要）

安装示例：

```bash
pip install torch torchvision medmnist
```

## 运行

### CIFAR10（FEDCSAP + 混合更新 + 高斯噪声）

```bash
python -m test_fl.fl_system.main \
  --dataset cifar10 \
  --aggregation fedcsap \
  --fedcsap-hybrid-alpha 0.5 \
  --gaussian-noise-std 0.001 \
  --attack sf \
  --mal-pcnt 0.3
```

### CIFAR10（clean, FedAvg）

```bash
python -m test_fl.fl_system.main \
  --dataset cifar10 \
  --attack none \
  --aggregation fedavg
```

### CIFAR10（固定同一批样本用于本地训练与重建对比）

```bash
python -m test_fl.fl_system.main \
  --dataset cifar10 \
  --aggregation fedavg \
  --attack none \
  --fixed-batch \
  --local-epochs 1 \
  --batch-size 1 \
  --num-images 1 \
  --rounds 1
```

> 说明：`--batch-size` 控制本地训练 mini-batch，`--num-images` 控制梯度反演时要重建的图像数量（例如重建 16 张图像时使用 `--num-images 16`）。
>
> 额外说明：如果开启 `--fixed-batch`，重建样本来自同一个固定训练 batch，因此必须满足 `num-images <= batch-size`。若你想用 `--batch-size 1 --num-images 16`，请不要开启 `--fixed-batch`。


### PATHMNIST（clean）

```bash
python -m test_fl.fl_system.main \
  --dataset pathmnist \
  --attack none \
  --aggregation fedavg
```

### MNIST（clean）

```bash
python -m test_fl.fl_system.main \
  --dataset mnist \
  --attack none \
  --aggregation fedavg
```

## 快速冒烟测试（降低数据量）

```bash
python -m test_fl.fl_system.main \
  --dataset mnist \
  --rounds 1 \
  --attack none \
  --aggregation fedcsap \
  --fedcsap-hybrid-alpha 0.7 \
  --gaussian-noise-std 0.0001 \
  --max-train-samples-per-client 32 \
  --max-test-samples 128
```


python -m test_fl.fl_system.main     --dataset cifar10     --aggregation fedavg     --local-epochs 1     --batch-size 1    --gaussian-noise-std 0.0     --num-clients 20 --attack=none --mal-pcnt=0

## 16 张图像（seed=0~15）4x4 对比脚本

当你需要在 `batch-size=1` 与 `num-images=1` 下，连续跑 16 个不同 seed（0~15），并比较：

- 原图
- 原始梯度重建（FedAvg, 无噪声）
- 加噪梯度重建（FedAvg, 有噪声）
- FEDCSAP 混合更新重建

可直接运行：

```bash
python -m test_fl.fl_system.run_compare16 \
  --dataset cifar10 \
  --rounds 1 \
  --local-epochs 1 \
  --batch-size 1 \
  --num-images 1 \
  --num-clients 5 \
  --attack none \
  --mal-pcnt 0 \
  --noisy-std 0.001 \
  --fedcsap-alpha 0.5
```

输出：
- `attack_results/compare16/compare16_original_4x4.png`：16 张原图组成的 4x4 图。
- `attack_results/compare16/compare16_grad_4x4.png`：16 张原始梯度重建图组成的 4x4 图。
- `attack_results/compare16/compare16_grad_noise_4x4.png`：16 张噪声梯度重建图组成的 4x4 图。
- `attack_results/compare16/compare16_fedcsap_4x4.png`：16 张 FEDCSAP 混合更新重建图组成的 4x4 图。
- `attack_results/compare16/compare16_losses.json`：每个 seed 的 loss 汇总与对应文件路径（用于 loss 对比）。
