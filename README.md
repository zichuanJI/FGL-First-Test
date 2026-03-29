# FGL-First-Test

一个面向初学者的联邦图学习（Federated Graph Learning, FGL）实验项目。

当前版本基于 `PyTorch` 和 `PyTorch Geometric`，围绕一个最小可运行的 `GCN + FedAvg` 联邦训练流程做了两件事：

1. 把训练、实验、结果汇总的代码结构重构干净。
2. 把 README 里提到的 Non-IID 性能劣化问题重新复现、解释，并给出一个在当前设定下有效的缓解方案。

项目的核心目标不是实现一个完整联邦系统，而是在单机上模拟多个 client，观察图学习在 IID / Non-IID 划分下的训练行为。

当前默认设定是：

- 数据集：`Cora`
- 模型：两层 `GCN`
- 联邦聚合：`FedAvg`
- client 数量：`3`
- client 数据：共享同一张图的 `x` 和 `edge_index`，但每个 client 只在自己的训练节点上计算监督 loss

这意味着它更准确地说是一个“节点标签划分下的联邦式图学习实验”，非常适合看：

- 数据异质性如何影响训练
- `local_epochs` 如何影响收敛
- 不同划分策略为什么会导致精度掉得很厉害
- 哪些简单缓解方法在这个 toy setting 里确实有用

## 代码结构

```text
fgl-first-test/
├── main.py
├── README.md
├── requirements.txt
├── data/
├── outputs/
└── src/
    ├── client.py
    ├── config.py
    ├── data_loader.py
    ├── experiments.py
    ├── model.py
    ├── server.py
    ├── trainer.py
    └── utils.py
```

模块职责：

- `main.py`：CLI 入口，只负责组装配置和分发模式
- `src/config.py`：训练配置 `TrainConfig`
- `src/data_loader.py`：加载 `Cora`，构造 `IID / Soft Non-IID / Hard Non-IID / Dirichlet` 划分，支持给所有 client 注入共享锚点样本
- `src/client.py`：本地训练，支持 `FedAvg` 和 `FedProx`
- `src/server.py`：参数聚合（FedAvg）
- `src/trainer.py`：联邦训练主循环、评估、训练结果对象
- `src/experiments.py`：批量实验和 sweep 配置生成
- `src/utils.py`：画图、CSV 保存、汇总表打印

## 安装

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## 常用命令

单次训练：

```bash
python main.py --mode train --partition_mode iid --algorithm fedavg
```

比较不同 partition：

```bash
python main.py --mode compare_partitions --algorithm fedavg --num_rounds 10 --local_epochs 10 --seeds 1 2 3
```

做 `local_epochs` sweep：

```bash
python main.py --mode sweep_local_epochs --partition_mode soft_noniid --algorithm fedavg --local_epochs_list 1 5 10 20 --seeds 1 2 3
```

做 Dirichlet sweep：

```bash
python main.py --mode sweep_dirichlet --dirichlet_alphas 0.1 0.3 0.5 1.0 --seeds 1 2 3
```

比较 Non-IID 缓解策略：

```bash
python main.py --mode compare_strategies --partition_mode soft_noniid --shared_per_class 2 --seeds 1 2 3
python main.py --mode compare_strategies --partition_mode hard_noniid --shared_per_class 5 --seeds 1 2 3
```

输出会自动保存到 `outputs/` 或你指定的 `--output_dir`。

## 为什么 Non-IID 会差这么多

在这个项目的设定里，Non-IID 掉得厉害不是偶然，而是由任务定义直接决定的。

### 1. 标签偏斜非常强

`soft_noniid` 里，每个 client 都有明显主类；`hard_noniid` 里，一个 client 甚至只看到 2 到 3 个类别。

例如 `hard_noniid`：

- Client 0 只看类 `0/1`
- Client 1 只看类 `2/3`
- Client 2 只看类 `4/5/6`

这种情况下，每个 client 的局部目标根本不是“全局 7 分类”，而是“自己那几个类的局部分类问题”。

### 2. 局部优化目标互相冲突

FedAvg 假设各 client 更新方向大体一致，但极端 label skew 下这个假设不成立。

- Client 0 会把参数往“更擅长区分类 0/1”的方向推
- Client 1 会把参数往“更擅长区分类 2/3”的方向推
- Client 2 会把参数往“更擅长区分类 4/5/6”的方向推

服务端平均之后，得到的是一个折中模型，但这个折中往往对任何单一 client 都不理想，对全局测试集也不理想。

### 3. 图模型会放大这种不匹配

GCN 会做邻居聚合。虽然每个 client 都看得到整张图的结构，但监督信号只来自自己那一小部分标签。

结果就是：

- 表征传播是全图级的
- 梯度监督却是强偏斜的

这会让不同 client 学出来的决策边界差异更大。

### 4. `hard_noniid` 本质上是“缺类训练”

这点最致命。一个 client 连某些类的训练样本都没见过，就很难给这些类学出稳定表示。只靠参数平均，往往补不回来。

## 基线结果

使用命令：

```bash
python main.py --mode compare_partitions --algorithm fedavg --num_rounds 10 --local_epochs 10 --seeds 1 2 3 --output_dir outputs_refactor
```

在当前代码上得到：

- `iid`：`mean final acc = 0.7957`
- `soft_noniid`：`mean final acc = 0.3467`
- `hard_noniid`：`mean final acc = 0.2140`

对应汇总文件：

- `outputs_refactor/partitions_fedavg_summary.csv`

这说明 README 里“Non-IID 会明显更差”这个结论是成立的，而且差距确实很大。

## 缓解方案

我加了两种策略：

### 1. `FedProx`

通过在 client 本地 loss 上加 proximal term，减少 client update 偏离全局模型太多。

命令示例：

```bash
python main.py --mode train --partition_mode soft_noniid --algorithm fedprox --prox_mu 0.01
```

### 2. 共享锚点样本 `shared_per_class`

这是当前项目里更有效的办法。

做法是：从训练集里每个类别抽少量样本，作为共享小缓冲区，给所有 client 都加上。这样每个 client 至少能看到一点全局类别信息，直接缓解极端 label skew。

命令示例：

```bash
python main.py --mode train --partition_mode soft_noniid --shared_per_class 2
python main.py --mode train --partition_mode hard_noniid --shared_per_class 5
```

## 哪个办法真的有效

在当前 toy setting 下，`shared_per_class` 明显比单独的 `FedProx` 更有效。

### Soft Non-IID

使用命令：

```bash
python main.py --mode compare_strategies --partition_mode soft_noniid --shared_per_class 2 --seeds 1 2 3 --output_dir outputs_refactor
```

得到：

- `fedavg`：`0.3467`
- `fedprox`：`0.2520`
- `fedavg+shared2`：`0.6883`
- `fedprox+shared2`：`0.5420`

### Hard Non-IID

使用命令：

```bash
python main.py --mode compare_strategies --partition_mode hard_noniid --shared_per_class 5 --seeds 1 2 3 --output_dir outputs_refactor
```

得到：

- `fedavg`：`0.2140`
- `fedprox`：`0.1963`
- `fedavg+shared5`：`0.7210`
- `fedprox+shared5`：`0.5963`

对应汇总文件：

- `outputs_refactor/strategies_soft_noniid_summary.csv`
- `outputs_refactor/strategies_hard_noniid_summary.csv`

结论很直接：

- 你现在看到的性能崩塌，主要问题不是“优化器不够强”
- 而是“每个 client 缺了太多类别信息”
- 所以最有效的缓解不是先换聚合器，而是先给每个 client 一点跨类锚点

## 怎么继续探这个问题

建议按下面顺序做实验：

1. 先固定 `num_rounds=10`、`local_epochs=10`，跑 `compare_partitions`
2. 再在 `soft_noniid` 和 `hard_noniid` 下分别跑 `compare_strategies`
3. 逐步调 `shared_per_class`
4. 再看 `major_ratio` 和 `dirichlet_alpha`
5. 最后再考虑更复杂的优化方法，比如 `SCAFFOLD`、更正式的图划分、个性化联邦学习

推荐先试的参数：

- `soft_noniid`：`shared_per_class = 1, 2, 3`
- `hard_noniid`：`shared_per_class = 3, 5, 8`
- `dirichlet_alpha`：`0.1, 0.3, 0.5, 1.0`

## 当前结论

这个项目现在已经比较适合做一件事：把 “Non-IID 为什么会坏” 和 “什么改动真的能缓解” 直接跑出来。

如果你后面要继续做得更像正式科研项目，我建议下一步优先做：

1. 把当前“共享整图，只切标签节点”的设定扩展到真正的子图级 client 划分
2. 加 `SCAFFOLD` 或 server momentum，而不是只停留在 `FedAvg / FedProx`
3. 把共享锚点样本和严格隐私场景区分开，明确它是“工程缓解手段”，不是纯 FL 假设

## 项目终结说明

这个项目到这里就结束，不再继续扩展。

对这个仓库来说，最重要的结论已经明确：

1. 它成功实现了一个最小可运行的 FGL 实验框架。
2. 它清楚复现了 IID 与 Non-IID 之间的性能差异。
3. 它说明了当前设定下 Non-IID 性能显著下降的主要原因：极强的标签偏斜与 client 目标不一致。
4. 它也验证了一个简单直接的缓解办法：给每个 client 注入少量跨类别共享锚点样本。

因此，这个项目最终停留在“一个完整、可解释、可复现实验现象的入门 FGL 项目”这个定位上，不再追求把它继续扩展成更复杂的科研代码库。
