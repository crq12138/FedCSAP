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
  --rounds 1
```


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