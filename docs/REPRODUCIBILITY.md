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

原 CoOp 实验的训练部分仅优化 CoOp 的共享上下文提示；CLIP 编码器和 logit scale 冻结。当前配置有 4 个 context tokens、2,048 个可训练参数。描述分支不增加可训练参数，也不改变 CoOp 训练损失。

每类描述在 CLIP 分词与截断后去除重复 token 序列，最多保留 20 条有效描述。对各描述嵌入归一化，取有效描述的均值，再次归一化。在 CoOp 实验中，两个分类器共享归一化图像特征，分别得到 logits，并在各自 softmax 后按 `alpha=0.5` 平均概率。MaPLe/MMRL 的辅助图像编码规则见下文扩展协议。

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

`mean_s(HM_s)` 与 `dataset_HM` 是两种统计量；`H(mean_dataset(B), mean_dataset(N))` 与 `Macro_HM` 也不同。原 CoOp 主表与 KgCoOp、MaPLe/MMRL 扩展主表使用上面的 `Macro_HM`，从未舍入的数值计算后再显示两位小数。新增底座同时单列 `H(mean_dataset(B), mean_dataset(N))`，用于与采用这一口径的论文 Average-HM 对照。

匹配数量模板控制与 CDPF 共用 CoOp checkpoint、类别划分、图像和 `alpha=0.5`，并匹配每类保留文本数量与原型聚合方式。它是对已评估描述配置的控制，不代表所有可能的文本集成方法。KgCoOp、MaPLe/MMRL 的十一数据集三种子四配置对照已完成，完整结果见 [KgCoOp](../results/kgcoop_20261110/README.md)与 [MaPLe/MMRL](../results/multimodal_20261110/README.md)。文献报告值不能标为本地受控复现。

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


## MaPLe 与 MMRL 扩展协议

[公开结果目录](../results/multimodal_20261110/README.md)提供 66 个案例的四配置 Base/Novel 正确数与样本数、22 组逐数据集结果及两个独立定义的宏指标。`tools/summarize_results.py` 仍对应原 CoOp 结果；新增结构化数据为单独的数据格式。两个研究范围各自汇总。

MaPLe 固定官方提交 `69bce21ae8eda80ad6187534b2dce09cf6c59e17`，十一集均训练 5 epochs、batch 4。MMRL 固定官方提交 `27d8821f6ac135a1d0c8e4c5ed78af14df0cd82e`，ImageNet 为 5 epochs、batch 32，其余为 10 epochs、batch 4；EuroSAT 保留 `rep_dim=2048` 和官方逐集正则配置。逐集训练轮数、batch 与输入资产哈希见 [protocol.json](../results/multimodal_20261110/protocol.json)。

原方法保持官方推理：MaPLe 使用原生前向；MMRL Base 使用 `0.7 * class_logits + 0.3 * representation_logits`，Novel 使用 class logits。三种融合分别使用单模板、匹配描述数量的普通模板、类别描述，均与同一原方法后验按 0.5/0.5 融合。

MaPLe/MMRL 含有学习后的视觉提示，辅助分支使用单独的原始冻结 CLIP 图像编码器。描述原型预计算；图像前向不能直接复用底座含学习视觉提示的特征。2026-11-12 已完成 UCF101/ImageNet Novel seed1、batch 1/64 的独占 RTX 4090 测量，20 次预热和 100 次 CUDA-event；完整 32 分支、3,200 个原始值见 [计时数据](../results/multimodal_20261110/timing_20261112.json)及 [表格](../results/multimodal_20261110/timing_results.csv)。计时覆盖原生前向以及融合额外的原始 CLIP 图像前向；MaPLe 逐次原生文本编码包含在内，MMRL 保持其官方热缓存。数据 IO、CPU 预处理、H2D、加载和辅助文本原型准备统一排除，不套用 CoOp 的计时数值。显存快照包括所有分支共同常驻的原生训练器及其编码器/优化器/缓存、独立原始 CLIP、三组辅助原型及 input64；不表示原生独立部署或融合新增显存。

类别描述复用 [ATPrompt](https://github.com/zhengli97/ATPrompt) 发布的固定文本资产。ATPrompt 学习带锚定属性的结构化提示；CDPF 在推理阶段融合已训练提示分类器与冻结描述分类器的后验，描述文本的来源与本方法贡献分别标注。

## KgCoOp 扩展协议

[KgCoOp 公开结果](../results/kgcoop_20261110/README.md)包含十一数据集、seed 1/2/3 的 33 组本地结果和同检查点四配置对照，覆盖后来补齐的 ImageNet。它不等于此前十数据集的阶段结果，不能混用旧十集宏指标。

固定 [KgCoOp 官方提交](https://github.com/htyao89/KgCoOp/tree/c43ccc32c8739f64b165cfa205fcc6c0f5bb5b17) `c43ccc32c8739f64b165cfa205fcc6c0f5bb5b17`，采用 100 epochs、batch 128、4 context tokens、16-shot、知识正则权重 8、fp16 和最终 epoch-100 checkpoint。ImageNet 保持该配方，使用经过原始与重计算 logits/正则损失等价性及有限更新预检的 activation checkpointing，以及逐图核验的原始 JPEG 读取方式。

四配置为原方法、单模板后验融合、数量匹配模板后验融合和描述后验融合；同一组 Base/Novel 及四配置共用 checkpoint，融合固定 `alpha=0.5`。KgCoOp 仅学习文本提示，辅助分支可共享原始冻结 CLIP 图像特征；本次没有重新测量 KgCoOp 的额外 GPU 开销，不将 CoOp 或 Ding 的计时套用为 KgCoOp 的实测结果。

逐种子整数正确数、样本数与两种 HM 汇总见 [seed_results.json](../results/kgcoop_20261110/seed_results.json)、[summary.json](../results/kgcoop_20261110/summary.json)。`tools/summarize_results.py` 仍对应原 CoOp 主结果。
