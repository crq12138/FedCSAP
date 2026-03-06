# Standalone Federated Learning System (from scratch)

这是一个**完全独立于仓库现有训练代码**的新实现，代码全部位于 `test/fl_system/`。

## 满足需求

1. 客户端数量固定为 20（`--num-clients` 必须等于 20）。
2. 训练参数参考 `scripts/exp_04_cifar10.sh` 中 run_057 所使用的参数（可映射部分）：
   - rounds=200（对应 epochs=200）
   - clients=20（对应 no_models=20）
   - total_participants=25
   - committee_size=5
   - noniid=sampling_dirichlet
   - dirichlet_alpha=0.9
   - lr=0.1（对应 eta=0.1）
   - seed=0
   - 攻击默认 sf，恶意比例默认 0.3
3. 支持 CIFAR10 与 PATHMNIST，支持 clean 与 SF 攻击模式。
4. 聚合器当前为 FedAvg，并通过注册表预留扩展点。

## 依赖

- `torch`, `torchvision`
- `medmnist`（仅 PATHMNIST 需要）

安装示例：

```bash
pip install torch torchvision medmnist
```

## 运行

### CIFAR10（clean）

```bash
python -m test.fl_system.main \
  --dataset cifar10 \
  --attack none \
  --aggregation fedavg
```

### CIFAR10（SF 攻击）

```bash
python -m test.fl_system.main \
  --dataset cifar10 \
  --attack sf \
  --mal-pcnt 0.3 \
  --aggregation fedavg
```

### PATHMNIST（clean）

```bash
python -m test.fl_system.main \
  --dataset pathmnist \
  --attack none \
  --aggregation fedavg
```

### PATHMNIST（SF 攻击）

```bash
python -m test.fl_system.main \
  --dataset pathmnist \
  --attack sf \
  --mal-pcnt 0.3 \
  --aggregation fedavg
```

## 快速冒烟测试（降低数据量）

```bash
python -m test.fl_system.main \
  --dataset cifar10 \
  --rounds 1 \
  --attack none \
  --max-train-samples-per-client 32 \
  --max-test-samples 128
```
