# CDPF 复现说明

## 公开复现范围

| 工作 | 当前提供的内容 | 仍需的资产或入口 |
|---|---|---|
| 重新计算汇总结果 | [当前结果 JSON](../results/cdpf_results.json)、[汇总脚本](../tools/summarize_results.py)及其来源记录 | Python；无需 GPU 或训练依赖 |
| 训练 CoOp 分支 | [训练入口](../train.py)、[训练器](../trainers/coop_textconsistency_ablation.py)、配置与数据读取代码 | 数据集、固定划分、CLIP 权重及运行环境 |
| 十数据集融合评估 | [evaluate_frozen_fusion.py](../scripts/repro/evaluate_frozen_fusion.py)，固定绑定当前精简训练源码 | 用该源码新训练的 checkpoint、配置和指标清单，以及数据与 CLIP 权重 |
| ImageNet 与模板控制的逐图重评估 | 已保存的结果与来源证据 | 公开的统一评估包装、完整冻结源码和 checkpoint 发布仍未齐备 |

因此，当前发布可以独立复算已保存的论文指标，也保留十数据集的新训练/融合评估代码，但尚不能仅凭整理后的默认源码树一条命令重新训练并评估全部十一数据集。旧实验 checkpoint 的重评估另需对应的历史脚本与冻结源码。来源文件中的 checkpoint 哈希用于标识实验资产，本身不包含模型权重；服务器备份路径也不是公开下载地址。

## 先复算汇总

在仓库根目录运行：

```bash
python tools/summarize_results.py
```

汇总对象为十一数据集、三个种子的 Base/Novel accuracy。该过程不读取原始图像，不进行模型训练或推理。JSON 中的来源与哈希用于记录溯源，汇总脚本不会下载或重新核验这些外部资产。逐图准确率需要原始预测或重新推理，不能仅由汇总 JSON 证明。

## 方法与指标口径

CDPF 的训练部分仅优化 CoOp 的共享上下文提示；CLIP 编码器和 logit scale 冻结。当前配置有 4 个 context tokens、2,048 个可训练参数。描述分支不增加可训练参数，也不改变 CoOp 训练损失。

每类描述在 CLIP 分词与截断后去除重复 token 序列，最多保留 20 条有效描述。对各描述嵌入归一化，取有效描述的均值，再次归一化。两个分类器共享归一化图像特征，分别得到 logits，并在各自 softmax 后按 `alpha=0.5` 平均概率。

保留代码中的分支名称含义为：

| 分支名称 | 含义 |
|---|---|
| `student` | CoOp |
| `description_mean` | 冻结均值描述分类器 |
| `simple_fusion` | CoOp 与描述 logits 的均值，作为 logit fusion 控制 |
| `mean_probability_fusion` | **最终 CDPF**，两个后验概率的均值 |

同一种子下，Base 和 Novel 使用同一个 epoch-20 CoOp checkpoint 及同一融合规则。配置中的 Novel 划分键写作 `new`。

设一个数据集三个种子的准确率分别为 `B_s`、`N_s`：

```text
B = mean_s(B_s)
N = mean_s(N_s)
dataset_HM = 2 * B * N / (B + N)
Macro_HM = mean_dataset(dataset_HM)
```

`mean_s(HM_s)` 与 `dataset_HM` 是两种统计量；`H(mean_dataset(B), mean_dataset(N))` 与 `Macro_HM` 也不同。主表使用上面的 `Macro_HM`，从未舍入的数值计算后再显示两位小数。

匹配数量模板控制与 CDPF 共用 CoOp checkpoint、类别划分、图像和 `alpha=0.5`，并匹配每类保留文本数量与原型聚合方式。它是对已评估描述配置的控制，不代表所有可能的文本集成方法。KgCoOp/Ding 的新增本地对照尚未完成；文献报告值不能标为本地受控复现。

## 环境与数据

结果汇总入口仅需 Python 标准库。模型代码沿用 Python 3.12、PyTorch 2.8.0、torchvision 0.23.0 的历史验证版本；完整 GPU 重跑还需按机器驱动安装匹配的 PyTorch CUDA 构建。

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.8.0 torchvision==0.23.0
python -m pip install -r requirements.txt
export PYTHONPATH="$PWD:$PWD/Dassl.pytorch"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
```

上述 PyTorch 命令给出版本；CUDA 构建需与实际机器匹配。[requirements.lock.txt](../requirements.lock.txt) 是历史环境快照，不是跨平台二进制锁文件。Dassl 已包含在 [Dassl.pytorch/](../Dassl.pytorch/)，不要再叠加安装其旧依赖清单。

数据目录与固定划分见 [DATASETS.md](DATASETS.md)。类别描述保存在 [templates/](../templates/)。数据集图像、预训练 CLIP 权重和训练 checkpoint 不包含在当前仓库中。UCF101 使用中间帧图像分类数据。

## 训练入口与兼容性

训练器继续使用注册名 `CoOp_TextConsistency_Ablation`，以兼容已有配置与 checkpoint。当前 CDPF 训练使用 `AGGREGATION=none`、`ABLATION_GROUP=baseline`、`INFERENCE_BRANCH=student`、`KD_WEIGHT=0.0`，仅训练 CoOp。名称中的历史 `Ablation` 不代表 CDPF 使用层级注意力或蒸馏。

现有基线启动器支持原十数据集。以下命令只输出 UCF101 三种子 CoOp 训练与评估计划，供检查配置：

```bash
python scripts/repro/run_pilot.py \
  --data-root /absolute/datasets \
  --output-root runs/ucf101-s123 \
  --dataset ucf101 --seeds 1 2 3 --plan-only
```

该基线计划不会生成 CDPF 融合结果。当前融合评估器的 `DATASETS` 不含 ImageNet；其固定哈希与当前精简训练源码匹配，可评估用该源码新训练且具备完整清单的 checkpoint。

已报告实验的旧 checkpoint 记录了各自的训练源码哈希，与当前精简源码不同，当前评估器不会把它们当作新训练资产接受。重评估这些旧 checkpoint 必须恢复对应的历史评估脚本和冻结源码，并通过原配置、checkpoint 与清单校验。不能把当前源码哈希替换进旧清单，也不能修改预期哈希冒充旧实验身份。

## 效率结果范围

0.74% 是 UCF101 novel split、单张 RTX 4090、batch 64 的 CUDA-event 测量：CoOp 为 90.128 ms/batch，CDPF 为 90.796 ms/batch。每组使用 20 次预热与 100 次计时迭代，共享同一 checkpoint 与输入张量。测量排除模型加载、磁盘读取、CPU 预处理和一次性文本编码，不应推广为其他硬件或数据集的固定开销。

论文版本见 [paper/README.md](../paper/README.md)，旧材料与历史冻结版本见 [HISTORY.md](HISTORY.md)。
