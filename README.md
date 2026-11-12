# CDPF: Class-Description Posterior Fusion

**Class-Description Posterior Fusion for Base-to-Novel Recognition** 的代码与实验结果。

CDPF 将训练后的提示分类器与冻结类别描述分类器的后验概率等权平均。名称中的 Class-Description 指新增的辅助分支，另一侧为已学习的提示分类器；融合不增加可训练参数，也不改变底座训练。对 CoOp 和 KgCoOp，两分支共享冻结 CLIP 图像编码器；对 MaPLe 和 MMRL，描述分支使用单独的原始冻结 CLIP 图像编码器，描述原型可提前缓存。

```text
p_CDPF = 0.5 * softmax(z_prompt) + 0.5 * softmax(z_description)
```

每类最多保留 20 条有效描述：经 CLIP 分词、截断和去重后，分别归一化描述向量，取均值，再归一化为类别原型。所有数据集、类别划分和随机种子使用同一融合权重 `alpha=0.5`。

## CoOp 主结果

CLIP ViT-B/16，16-shot Base-to-Novel，seed 1/2/3，CoOp 训练 20 epochs。下表为十一个数据集的 **Macro HM（%）**。

| 方法 | Macro HM |
|---|---:|
| 匹配的 CoOp | 75.06 |
| CoOp + matched-count conventional-template probability fusion | 77.11 |
| **CDPF** | **78.33** |

数据集包括 ImageNet、Caltech101、OxfordPets、StanfordCars、OxfordFlowers、Food101、FGVC Aircraft、SUN397、DTD、EuroSAT 和 UCF101。每个数据集先对三个种子的 Base/Novel accuracy 分别求均值，再计算 HM；Macro HM 是十一个数据集 HM 的等权平均。它与先跨数据集平均 Base/Novel 再求 HM 的数值不同。

在 **UCF101 novel split、单张 RTX 4090、batch 64** 的已测设置中，CDPF 相对 CoOp 的推理延迟增加 0.74%。该计时不含模型加载、磁盘读取、CPU 预处理和一次性文本编码；它只描述这一测量设置。

## KgCoOp、MaPLe 与 MMRL 扩展结果

2026-11-11 更新：CLIP ViT-B/16，16-shot，十一数据集 × 三个种子，每种底座 33 组，共 99 组本地训练与评估。每组四种推理配置共用同一 checkpoint，融合固定 `alpha=0.5`，不增加训练或按测试结果选择权重。

| 推理配置 | KgCoOp Macro HM | MaPLe Macro HM | MMRL Macro HM |
|---|---:|---:|---:|
| 原方法 | 76.75 | 78.09 | 80.89 |
| 单模板后验融合 | 76.12 | 78.55 | 80.54 |
| 数量匹配模板后验融合 | 76.64 | 78.67 | 80.51 |
| **CDPF 描述后验融合** | **78.18** | **79.78** | **81.68** |

上表沿用 CoOp 主结果的 `Macro HM = mean_dataset(HM)`：每集先平均三个种子的 Base/Novel，再计算该集 HM，最后对十一集等权平均。按 KgCoOp / MaPLe / MMRL 顺序，CDPF 相对原方法分别提高 **1.42 / 1.69 / 0.79 个百分点**，相对数量匹配模板分别提高 **1.53 / 1.11 / 1.16 个百分点**。

采用部分论文 Average 栏的另一口径 `HM(mean_dataset(Base), mean_dataset(Novel))` 时，KgCoOp + CDPF 为 **78.28**，MaPLe + CDPF 为 **79.93**，MMRL + CDPF 为 **81.80**；这些值不能与上表的 Macro HM 混用。

相对原方法，KgCoOp 有 8/11 个数据集提升，Oxford Pets、Flowers102 与 Stanford Cars 下降；MaPLe 有 9/11 个数据集提升，Oxford Pets 与 Stanford Cars 下降；MMRL 有 10/11 个数据集提升，Stanford Cars 下降。完整四分支、逐集/逐种子计数、冻结版本与两个汇总口径见 [KgCoOp 结果](results/kgcoop_20261110/README.md)与 [MaPLe/MMRL 结果](results/multimodal_20261110/README.md)。这些均为本地实验，文献报告值应单独标注。

### 多模态底座推理计时

2026-11-12 补充：单张 RTX 4090，UCF101 与 ImageNet 的 Novel seed1，20 次预热、100 次 CUDA-event 测量。下表为 **batch 64 的延迟中位数，ms/批**，原生分支与融合分支处于相同常驻环境。

| 数据集 | 底座 | 原生分支 | CDPF | 中位延迟增幅 |
|---|---|---:|---:|---:|
| UCF101 | MaPLe | 35.80 | 66.39 | +85.44% |
| UCF101 | MMRL | 77.90 | 107.68 | +38.22% |
| ImageNet | MaPLe | 81.33 | 110.10 | +35.38% |
| ImageNet | MMRL | 77.87 | 107.63 | +38.22% |

三种融合均额外执行一次原始冻结 CLIP 图像前向；MaPLe 原生逐次文本编码包含在计时中，MMRL 保留官方预热文本缓存。测量重复同一驻留 GPU batch，排除 CPU 预处理、磁盘/归档读取、H2D、加载和一次性辅助文本原型准备。两种 batch、四分支和全部原始值见 [计时明细](results/multimodal_20261110/README.md#推理开销)。

这些开销不属于 CoOp 已测共享图像编码器的 0.74% 设置。显存只记录共同常驻上下文，不宣称为原生独立部署或融合新增显存。

## 无需 GPU 的结果复算

在仓库根目录执行：

```bash
python tools/summarize_results.py
```

该入口读取仓库内保存的 CoOp 逐种子结果，校验并重新计算原论文汇总指标，无需下载数据集或训练模型。新增底座的原始正确数与样本数另存于 [KgCoOp](results/kgcoop_20261110/seed_results.json)和 [MaPLe/MMRL](results/multimodal_20261110/seed_results.json) 的逐种子 JSON。结果见 [results/cdpf_results.json](results/cdpf_results.json)，详细口径与复现范围见 [复现说明](docs/REPRODUCIBILITY.md)。结果复算验证已保存的指标；重新训练和逐图推理还需要数据、CLIP 权重、checkpoint 与对应冻结源码。

## 代码与证据

| 位置 | 内容 |
|---|---|
| [train.py](train.py) / [当前训练器](trainers/coop_textconsistency_ablation.py) | CoOp 提示训练；保留注册名 `CoOp_TextConsistency_Ablation` 以兼容已有配置与 checkpoint |
| [templates/](templates/) | 十一个数据集的类别描述 JSON |
| [configs/](configs/) / [datasets/](datasets/) | 训练配置与数据集读取逻辑 |
| [results/](results/) / [tools/summarize_results.py](tools/summarize_results.py) | 当前论文结果与汇总复算入口 |
| [reports/README.md](reports/README.md) | 保留的实验结果与来源索引 |
| [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md) | 环境、指标口径、代码兼容性与公开复现范围 |
| [paper/README.md](paper/README.md) | 论文版本与同步说明 |
| [docs/HISTORY.md](docs/HISTORY.md) | 历史材料位置与旧版本说明 |

训练、融合评估与结果复算的输入和使用方式见 [复现说明](docs/REPRODUCIBILITY.md)。各底座按其冻结源码版本和实验协议解释，历史 checkpoint 使用对应版本。

本地同检查点对照与文献报告值分别标注。CDPF 对不同底座的结果按各自数据范围和协议独立汇总。
