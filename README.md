# FedCSAP 代码归档说明

> 作者：陈睿齐<br>
> 编写时间：2026.06.10<br>
> 项目来源：本仓库基于 FLShield 实验代码扩展，用于联邦学习鲁棒聚合、投毒攻击防御与 FedCSAP 相关实验归档。

## 1. 项目目的

FedCSAP 用于复现实验和归档毕业论文相关代码，主要围绕联邦学习场景下的恶意客户端、投毒攻击与鲁棒聚合防御展开。仓库保留了原 FLShield 框架，并在此基础上增加/整理了 FedCSAP、FRFL、CBRFL 等聚合方法、委员会选举、委员会接管攻击模拟、实验日志导出和论文绘图脚本。

本 README 重点记录：

- 需要安装的 Python 库与运行环境；
- 关键运行入口、常用命令和参数；
- 主要文件/目录的用途；
- 原始数据、缓存数据、模型、实验结果和图片的存储位置；
- 从实验结果提取数据和绘图的流程。

## 2. 运行环境与 Python 库

### 2.1 Python 版本

推荐使用 Python 3.8.x 或 Python 3.9.x。原始代码中存在较多 PyTorch、torchvision、medmnist、fancyimpute 等依赖，建议使用虚拟环境单独部署。

### 2.2 创建虚拟环境

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

如果使用 Conda，也可以：

```bash
conda create -n fedcsap python=3.9 -y
conda activate fedcsap
python -m pip install --upgrade pip
```

### 2.3 安装依赖

仓库提供了 `requirements.txt`，可直接安装：

```bash
pip install -r requirements.txt
```

`requirements.txt` 中列出的主要第三方库包括：

- 数值计算与数据处理：`numpy`、`pandas`、`scipy`、`scikit-learn` / `scikit_learn`、`sympy`；
- 深度学习与联邦学习实验：`torchvision`、`torchmetrics`、`pytorch_lightning`、`opacus`、`medmnist`；
- 聚类、插补与鲁棒聚合辅助：`hdbscan`、`fancyimpute`；
- 绘图与图像处理：`matplotlib`、`opencv-python`；
- 配置、日志与进度显示：`pyyaml`、`tabulate`、`termcolor`、`tqdm`、`visdom`、`colorama`。

此外，代码运行还会用到以下库，若环境中缺失请手动补充：

```bash
pip install torch jinja2 adjustText pillow
```

说明：

- `torch` 建议根据本机 CUDA 版本从 PyTorch 官网选择合适安装命令；CPU 环境也可以运行小规模实验，但速度较慢。
- `jinja2` 用于渲染 `utils/jinja.yaml` 动态配置。
- `adjustText` 用于部分散点图标签避让；如果只运行训练、不画对应图，可暂时不装。
- 部分中文图使用 `SimHei` 字体；如果绘图时报缺少字体，需要在系统中安装 `simhei.ttf` 或修改 `plot.py` 中的字体设置。

## 3. 项目组成与文件描述

| 路径 | 作用 |
| --- | --- |
| `main.py` | 主训练入口。解析命令行参数，读取/渲染 YAML 配置，创建 Helper，执行联邦训练、聚合、测试和日志保存。 |
| `config.py` | 全局常量、设备选择、聚合方法名称、攻击方法名称、数据集名称、类别数量和目标标签映射。 |
| `helper.py` | 训练过程通用辅助类。负责实验目录创建、日志、模型保存、聚合前后数据组织、验证容器保存等。 |
| `image_helper.py` | 图像类数据集加载与划分，包括 CIFAR-10、MNIST、Fashion-MNIST、EMNIST、PathMNIST、TinyImageNet、CelebA 等。 |
| `loan_helper.py` | Loan 表格数据集加载、按州参与方组织、Loan 模型创建与投毒测试接口。 |
| `train.py` / `image_train.py` / `loan_train.py` | 本地客户端训练、图像任务训练和 Loan 任务训练相关代码。 |
| `test.py` | 全局模型测试、投毒测试、F1/accuracy 等指标计算。 |
| `models/` | 模型定义，包括 MNIST、PathMNIST、Loan、CIFAR ResNet、TinyImageNet ResNet、CelebA ResNet 等。 |
| `flshield_utils/` | FLShield 相关聚类、插补、验证测试和验证后处理代码。 |
| `fedcsap_utils/` | FedCSAP 聚合和验证测试代码。 |
| `frfl_utils/` | FRFL 聚合代码。 |
| `cbrfl_utils/` | CBRFL 聚合代码。 |
| `utils/jinja.yaml` | 动态实验配置模板。多数命令行参数最终会渲染到该文件对应的配置项。 |
| `utils/*_params.yaml` | 不同数据集/任务的静态配置文件，如 `cifar_params.yaml`、`fmnist_params.yaml`、`pathmnist_params.yaml` 等。 |
| `utils/experiment_logger.py` | 实验 CSV 日志写入器，输出 global metrics、FedCSAP 客户端指标、委员会指标和类别 F1 变化。 |
| `utils/csv_record.py` | 训练/测试/投毒测试结果 CSV 保存工具。 |
| `scripts/` | 批量实验、数据读取、结果汇总和辅助绘图脚本。 |
| `scripts/read_runs_data.py` | 从 `runs/run_xxx/` 中读取实验结果并汇总到 `result/`。 |
| `plot.py` | 主要绘图入口，支持 FedCSAP R 值-委员会次数散点图、训练曲线对比图、模块耗时柱状图。 |
| `scripts/plot_commitee.py` | 委员会规模/恶意数量相关热力图脚本。 |
| `attack_of_the_tails/` | Attack of the Tails / edge-case 攻击相关数据和工具。 |
| `attack_results/` | 梯度反演或攻击结果 JSON 示例/归档。 |
| `test_fl/` | 额外联邦学习测试/反演相关代码，含独立 README。 |
| `plot_result/` | 论文图或绘图脚本默认输出目录。 |

## 4. 关键运行路径

### 4.1 主训练入口

```bash
python main.py --aggregation_methods=X --attack_methods=Y --type=Z
```

其中：

- `X`：聚合方法，常用取值包括 `mean`、`fedavg`、`geom_median`、`median`、`krum`、`flame`、`flshield`、`fedcsap`、`fltrust`、`afa`、`foolsgold`、`frfl`、`cbrfl`。
- `Y`：攻击方法，常用取值包括 `targeted_label_flip`、`dba`、`inner_product_manipulation`、`sf`、`mixed_8_tlf_sf_ipm_dba`、`attack_of_the_tails`、`semantic_attack`。
- `Z`：数据集类型，常用取值包括 `cifar`、`mnist`、`fmnist`、`emnist`、`emnist_letters`、`pathmnist`、`loan`、`tiny-imagenet-200`、`celebA`。

### 4.2 使用静态 YAML 配置运行

如果不传额外命令行参数，`main.py` 默认读取：

```bash
python main.py --params utils/fmnist_params.yaml
```

也可以指定其他配置文件：

```bash
python main.py --params utils/cifar_params.yaml
python main.py --params utils/pathmnist_params.yaml
```

### 4.3 使用动态模板运行

只要传入 `--type=...`、`--aggregation_methods=...` 等额外参数，程序会使用 `utils/jinja.yaml` 渲染动态配置。例如：

```bash
python main.py \
  --type=cifar \
  --aggregation_methods=fedcsap \
  --attack_methods=targeted_label_flip \
  --mal_pcnt=0.2 \
  --epochs=210 \
  --resumed_model=false
```

### 4.4 指定 run 目录名运行

可以在命令中加入形如 `--run_001` 的参数，程序会将本次实验输出固定保存到 `runs/run_001/`：

```bash
python main.py \
  --run_001 \
  --type=cifar \
  --aggregation_methods=fedcsap \
  --attack_methods=targeted_label_flip \
  --mal_pcnt=0.2 \
  --epochs=210 \
  --resumed_model=false
```

如果不指定 `--run_xxx`，默认输出到 `saved_models/<参数哈希>/`。

### 4.5 FedCSAP / FLShield 委员会相关参数

- `--no_models=N`：每轮参与训练/提交更新的非委员会客户端数量上限。
- `--committee_size=K`：每轮选出的验证委员会大小。
- `--number_of_total_participants=T`：总参与方数量。
- `--committee_election=reputation`：委员会选举方式，FedCSAP/CBRFL 默认偏向 reputation；FRFL 默认使用 random。
- `--fedcsap_committee_takeover_attack=true`：开启 FedCSAP 委员会接管攻击模拟；默认关闭。

例如：25 个总参与方，其中 5 个委员会成员、20 个训练客户端：

```bash
python main.py \
  --run_130 \
  --type=cifar \
  --aggregation_methods=fedcsap \
  --attack_methods=targeted_label_flip \
  --mal_pcnt=0.2 \
  --epochs=210 \
  --resumed_model=false \
  --number_of_total_participants=25 \
  --committee_size=5 \
  --no_models=20
```

如果运行 FLShield 的 bijective 版本，可加入：

```bash
--bijective_flshield
```

## 5. 数据存储位置

### 5.1 自动下载的数据

图像数据集主要由 `torchvision` 或 `medmnist` 自动下载：

| 数据集 | 默认位置 |
| --- | --- |
| CIFAR-10 | `./data` |
| MNIST | `./data` |
| Fashion-MNIST | `./data` |
| EMNIST digits / letters | `./data` |
| PathMNIST | `./data` |

首次运行相应数据集时会自动下载；如果服务器无网络，需要提前将数据放入 `./data`。

### 5.2 需要手动准备的数据

| 数据/任务 | 默认位置 | 说明 |
| --- | --- | --- |
| Loan | `./data/loan/` | `loan_helper.py` 会读取该目录下的 Loan CSV 文件，并按文件名中的州/参与方标识组织客户端。 |
| Lending Club 原始数据 | `./data/lending-club-loan-data/` | `utils/loan_preprocess.py` 预处理脚本使用。 |
| TinyImageNet | `./data/tiny-imagenet-200/` | 需要手动准备并可用 `utils/tinyimagenet_reformat.py` 处理验证集结构。 |
| CelebA | `./data/celebA/` | 需要手动准备，并可参考 `utils/celebA_reformat.py`、`utils/celebA_noniid.py`。 |
| Attack of the Tails | `attack_of_the_tails/` | 目录内包含 ARDIS CSV、Southwest/green car 相关 pickle 和图片数据。 |

### 5.3 数据缓存位置

代码中保留了将划分后的客户端数据缓存到以下目录的逻辑：

```text
saved_data/<dataset>/<save_data>/
```

典型文件名包括：

- `train_data_<client_id>.pt`
- `test_data.pt`
- `test_data_poison.pt`
- `test_targetlabel_data.pt`

当前主流程中该缓存保存分支默认未启用；如果需要复用数据划分，可根据 `image_helper.py` 中 `save_data` / `load_data` 相关逻辑开启或加载。

## 6. 实验结果、模型和日志存储位置

### 6.1 默认输出目录

| 情况 | 输出位置 |
| --- | --- |
| 命令中传入 `--run_xxx` | `runs/run_xxx/` |
| 未传入 `--run_xxx` | `saved_models/<参数哈希>/` |

### 6.2 每次实验常见输出文件

在 `runs/run_xxx/` 或 `saved_models/<参数哈希>/` 下，常见文件包括：

| 文件 | 说明 |
| --- | --- |
| `params.yaml` | 本次运行最终使用的参数配置。 |
| `log.txt` 或 `run_xxx.log` | 训练日志。使用 `--run_xxx` 时，日志文件名通常为 `run_xxx.log`。 |
| `timing_details.csv` | 每轮关键模块耗时，如训练、聚合等。 |
| `train_result.csv` | 训练结果。 |
| `test_result.csv` | 干净测试集结果。 |
| `posiontest_result.csv` | 投毒测试结果。注意文件名沿用原代码拼写 `posion`。 |
| `recall_result.csv` | recall 相关测试结果。 |
| `poisontriggertest_result.csv` | trigger 投毒测试结果。 |
| `validator_pcnt.csv` | 验证器判断恶意/良性比例相关结果。 |
| `global_metrics.csv` | 每轮全局 `global_acc` 和 `global_macro_f1`。 |
| `fedcsap_client_metrics.csv` | FedCSAP 客户端级指标，如恶意标记、R/reputation、是否通过等。 |
| `fedcsap_round_metrics.csv` | FedCSAP 轮级委员会指标，如委员会大小、恶意成员数、是否接管。 |
| `fedcsap_class_delta_f1.csv` | FedCSAP 部分类别 F1 变化记录。 |
| `epoch_reports.json` | 每轮报告。 |
| `result_dict.pkl` | 运行结束时保存的结果字典。 |
| `model_last.pt.tar` | 最新全局模型。 |
| `model_last.pt.tar.best` | 当前 best 模型。 |
| `model_last.pt.tar.epoch_<N>` | 指定 epoch 的模型快照。 |
| `validation_container_<epoch>.pkl` | 验证容器中间数据。 |
| `grads_<epoch>.npy` / `names_<epoch>.npy` | 部分聚合/分析流程保存的梯度和客户端名称。 |

### 6.3 预训练模型位置

默认恢复模型路径由 `utils/jinja.yaml` 控制，通常位于：

```text
utils/model_bank/<dataset>/model_last.pt.tar.epoch_<N>
```

例如：

- `utils/model_bank/cifar/model_last.pt.tar.epoch_100`
- `utils/model_bank/fmnist/model_last.pt.tar.epoch_35`
- `utils/model_bank/emnist/model_last.pt.tar.epoch_35`
- `utils/model_bank/loan/model_last.pt.tar.epoch_200`

如果不想加载预训练模型，可传入：

```bash
--resumed_model=false
```

## 7. 从 runs 中读取数据到 result

结果汇总脚本为：

```text
scripts/read_runs_data.py
```

通用命令格式：

```bash
python scripts/read_runs_data.py \
  --runs-dir ./runs \
  --start-run 129 \
  --end-run 131 \
  --task fedcsap_last_r_and_committee_takeover \
  --output-dir result
```

支持的常用任务包括：

| task | 输入文件 | 输出内容 |
| --- | --- | --- |
| `poisontest_accuracy_avg` | `posiontest_result.csv` | 统计投毒测试 accuracy 平均值。 |
| `global_macro_f1_max` | `global_metrics.csv` | 统计每个 run 最大 macro-F1 及平均值，并额外输出 raw values。 |
| `fedcsap_last_r_and_committee_takeover` | `params.yaml`、`fedcsap_client_metrics.csv`、`fedcsap_round_metrics.csv` | 汇总客户端最终 R 值、委员会当选次数、是否恶意和委员会接管次数。 |

输出目录默认为：

```text
result/
```

常见输出文件：

- `result/<task>_run_<start>_to_<end>.csv`
- `result/<task>_run_<start>_to_<end>_details.csv`
- `result/<task>_run_<start>_to_<end>_raw_values.txt`

## 8. 画图程序与图片存储位置

### 8.1 主绘图入口：`plot.py`

#### 8.1.1 FedCSAP R 值与委员会当选次数散点图

先用 `scripts/read_runs_data.py` 生成 details CSV，再绘图：

```bash
python plot.py fedcsap_r_vs_committee \
  --details-csv result/fedcsap_last_r_and_committee_takeover_run_129_to_131_details.csv \
  --run-id 130 \
  --output result/fedcsap_run_130_r_vs_committee.png
```

注意：该绘图函数会固定在 `plot_result/` 下同时导出：

- `.png`
- `.eps`

即使 `--output` 写成 `result/xxx.png`，实际散点图仍会以 `plot_result/<文件名>.png` 和 `plot_result/<文件名>.eps` 保存。

#### 8.1.2 多方法训练曲线对比图

```bash
python plot.py compare_training_curves \
  --runs-root runs \
  --output-dir plot_result/compare_training_curves
```

默认使用 `plot.py` 内置 run 映射，读取各 `runs/run_<id>/global_metrics.csv`。如需自定义 run 映射，可提供 CSV：

```bash
python plot.py compare_training_curves \
  --runs-root runs \
  --output-dir plot_result/compare_training_curves \
  --run-map-csv my_run_map.csv
```

`my_run_map.csv` 需要包含列：

```text
run_id,dataset,scheme
```

该命令会为 CIFAR、PathMNIST、MNIST 分别输出 ACC/F1 曲线，格式包括 `.png`、`.eps`、`.pdf`。

#### 8.1.3 FedCSAP 模块耗时柱状图

```bash
python plot.py fedcsap_timing_modules \
  --runs-root runs \
  --cifar10-run run_484 \
  --pathmnist-run run_485 \
  --mnist-run run_486 \
  --output plot_result/fedcsap_timing_modules.png
```

该命令读取每个 run 下的 `timing_details.csv`，输出 `.png` 和 `.eps`。

### 8.2 委员会热力图：`scripts/plot_commitee.py`

运行：

```bash
python scripts/plot_commitee.py
```

脚本内置了 CIFAR-10、PathMNIST、MNIST 的矩阵数据，用于生成不同恶意数量/委员会规模下的热力图。输出图片默认保存在脚本当前设置的路径中，建议统一移动或保存到：

```text
plot_result/
```

### 8.3 图片存储位置汇总

| 图片类型 | 默认/建议目录 |
| --- | --- |
| 论文结果图 | `plot_result/` |
| 训练曲线对比图 | `plot_result/compare_training_curves/` |
| FedCSAP R-委员会散点图 | `plot_result/` |
| 模块耗时柱状图 | `plot_result/` 或命令中 `--output` 的父目录 |
| Dirichlet 数据分布图 | 当前实验目录，如 `runs/run_xxx/Num_Img_Dirichlet_Alpha*.pdf` 或 `saved_models/<hash>/...` |

## 9. 推荐归档流程

一次完整实验归档建议按以下顺序执行：

1. 准备环境并安装依赖。
2. 准备数据到 `./data/`、`./data/loan/` 或其他对应目录。
3. 使用明确的 `--run_xxx` 运行实验，保证输出目录可追踪。
4. 检查 `runs/run_xxx/params.yaml`、日志、CSV、模型文件是否生成。
5. 用 `scripts/read_runs_data.py` 从多个 run 中汇总数据到 `result/`。
6. 用 `plot.py` 或 `scripts/plot_commitee.py` 生成论文图片到 `plot_result/`。
7. 最终归档以下目录/文件：
   - `README.md`
   - 关键源代码：`main.py`、`helper.py`、`image_helper.py`、`loan_helper.py`、`train.py`、`test.py`、`models/`、`*_utils/`、`utils/`、`scripts/`、`plot.py`
   - 配置文件：`utils/jinja.yaml`、`utils/*_params.yaml`
   - 实验结果：`runs/`、`saved_models/`、`result/`
   - 图片：`plot_result/`
   - 必要数据或数据说明：`data/`、`attack_of_the_tails/`、`attack_results/`

## 10. 常见问题

### 10.1 运行时找不到预训练模型

如果报错缺少 `utils/model_bank/<dataset>/...`，可以选择：

```bash
--resumed_model=false
```

或者将预训练模型放到 `utils/model_bank/<dataset>/` 中，并与 `utils/jinja.yaml` 中的 `resumed_model_name` 保持一致。

### 10.2 运行时找不到数据

- 自动下载数据集：确认服务器能访问网络，或提前下载到 `./data`。
- Loan：确认 `./data/loan/` 存在且包含按参与方拆分的 CSV。
- TinyImageNet/CelebA：需要手动准备到 `./data/tiny-imagenet-200/`、`./data/celebA/`。

### 10.3 绘图时报中文字体错误

安装 SimHei 字体，或修改 `plot.py` 中字体查找和 `font.sans-serif` 配置。

### 10.4 `posiontest_result.csv` 是否拼写错误

这是原代码沿用的文件名，虽然英文应为 `poison`，但脚本中读取的是 `posiontest_result.csv`，不要手动改名，否则汇总脚本找不到文件。

## 11. 原 FLShield 引用与致谢

本仓库来源于 FLShield 代码框架，并在其基础上做 FedCSAP 相关实验扩展。若使用原 FLShield 部分，请参考：

```bibtex
@article{kabir2023flshield,
  title={FLShield: A Validation Based Federated Learning Framework to Defend Against Poisoning Attacks},
  author={Kabir, Ehsanul and Song, Zeyu and Rashid, Md Rafi Ur and Mehnaz, Shagufta},
  journal={arXiv preprint arXiv:2308.05832},
  year={2023}
}
```

原项目致谢：

- [AI-secure/DBA](https://github.com/AI-secure/DBA)
- [ebagdasa/backdoor_federated_learning](https://github.com/ebagdasa/backdoor_federated_learning)
- [krishnap25/RFA](https://github.com/krishnap25/RFA)
- [DistributedML/FoolsGold](https://github.com/DistributedML/FoolsGold)
