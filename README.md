# CDPF: Class-Description Posterior Fusion

**Class-Description Posterior Fusion for Base-to-Novel Recognition** 的代码与实验结果。

CDPF 将训练后的 CoOp 分类器与冻结类别描述分类器的概率等权平均。两个分支共享冻结 CLIP 图像编码器；描述原型可提前缓存，CoOp 的训练过程保持不变，新增可训练参数为 0。

```text
p_CDPF = 0.5 * softmax(z_CoOp) + 0.5 * softmax(z_description)
```

每类最多保留 20 条有效描述：经 CLIP 分词、截断和去重后，分别归一化描述向量，取均值，再归一化为类别原型。所有数据集、类别划分和随机种子使用同一融合权重 `alpha=0.5`。

## 主要结果

CLIP ViT-B/16，16-shot Base-to-Novel，seed 1/2/3，CoOp 训练 20 epochs。下表为十一个数据集的 **Macro HM（%）**。

| 方法 | Macro HM |
|---|---:|
| 匹配的 CoOp | 75.06 |
| CoOp + matched-count conventional-template probability fusion | 77.11 |
| **CDPF** | **78.33** |

数据集包括 ImageNet、Caltech101、OxfordPets、StanfordCars、OxfordFlowers、Food101、FGVC Aircraft、SUN397、DTD、EuroSAT 和 UCF101。每个数据集先对三个种子的 Base/Novel accuracy 分别求均值，再计算 HM；Macro HM 是十一个数据集 HM 的等权平均。它与先跨数据集平均 Base/Novel 再求 HM 的数值不同。

在 **UCF101 novel split、单张 RTX 4090、batch 64** 的已测设置中，CDPF 相对 CoOp 的推理延迟增加 0.74%。该计时不含模型加载、磁盘读取、CPU 预处理和一次性文本编码；它只描述这一测量设置。

## 无需 GPU 的结果复算

在仓库根目录执行：

```bash
python tools/summarize_results.py
```

该入口读取仓库内保存的逐种子结果，校验并重新计算论文汇总指标，无需下载数据集或训练模型。结果见 [results/cdpf_results.json](results/cdpf_results.json)，详细口径与复现范围见 [复现说明](docs/REPRODUCIBILITY.md)。结果复算验证已保存的指标；重新训练和逐图推理还需要数据、CLIP 权重、checkpoint 与对应冻结源码。

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

当前仓库提供结果复算与训练核心。`scripts/repro/evaluate_frozen_fusion.py` 支持用当前精简源码新训练的十数据集 checkpoint，并严格校验固定源码哈希；旧实验 checkpoint 需要对应的历史脚本与冻结源码。ImageNet 与模板控制的公开评估包装、历史冻结源码和 checkpoint 发布仍需补齐，尚无十一数据集统一重评估入口；详见 [复现范围](docs/REPRODUCIBILITY.md#公开复现范围)。

本地匹配结果、模板控制和文献报告值应分别解读。当前结果不宣称 SOTA；KgCoOp 与 Ding 等方法的新增本地对照尚未完成，不属于已完成的受控比较。
