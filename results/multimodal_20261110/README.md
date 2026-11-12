# MaPLe 和 MMRL 上的 CDPF 对照结果

本地实验，2026-11-11 发布。CLIP ViT-B/16，16-shot Base-to-Novel，十一数据集，seed 1/2/3。MaPLe 与 MMRL 各 33 组，共 66 组；每组训练后，用同一个 checkpoint 比较四种推理配置。融合固定 `alpha=0.5`，没有新增训练或按测试准确率选择融合权重。

模型、预测、标签/样本顺序、结果、日志与收据已通过本地独立验收，完整备份保存在实验归档中。本目录提供正确数/样本数和可独立重算的汇总数据。

## 汇总结果与两种 HM

每个数据集先分别平均三个种子的 Base 和 Novel 准确率，再计算该数据集 HM。两个宏指标分别为：

```text
B_d = mean_seed(100 * base_correct / base_total)
N_d = mean_seed(100 * novel_correct / novel_total)
H(B, N) = 2 * B * N / (B + N)
mean_dataset_hm = mean_dataset(H(B_d, N_d))
hm_of_mean_base_novel = H(mean_dataset(B_d), mean_dataset(N_d))
```

准确率和 HM 单位为 %，差值单位为百分点（pp）。两种指标均先使用未舍入数据计算，表格最后显示四位小数。逐种子 HM 仅用于同种子配对分析，不先平均逐种子 HM 来替代每集汇总。

| 底座 | 配置 | 平均 Base | 平均 Novel | mean_dataset_hm | hm_of_mean_base_novel |
|---|---|---:|---:|---:|---:|
| MaPLe | 原方法 | 82.1344 | 74.9245 | 78.0906 | 78.3639 |
| MaPLe | 单模板后验融合 | 81.2011 | 76.4664 | 78.5533 | 78.7627 |
| MaPLe | 数量匹配模板后验融合 | 81.0390 | 76.7738 | 78.6733 | 78.8488 |
| MaPLe | CDPF 描述后验融合 | 81.5510 | 78.3643 | 79.7789 | 79.9259 |
| MMRL | 原方法 | 85.6583 | 76.9604 | 80.8852 | 81.0767 |
| MMRL | 单模板后验融合 | 84.5810 | 77.1737 | 80.5367 | 80.7077 |
| MMRL | 数量匹配模板后验融合 | 84.6029 | 77.1212 | 80.5142 | 80.6890 |
| MMRL | CDPF 描述后验融合 | 84.8817 | 78.9365 | 81.6792 | 81.8012 |

`79.7789 / 81.6792` 对应原 CoOp 主表使用的逐集 HM 均值；`79.9259 / 81.8012` 对应先平均 Base/Novel 再计算 HM 的另一口径。与文献比较时采用对方明确报告的同一口径，不能交叉使用。

## 同检查点配对结果

| 底座 | CDPF 参照配置 | mean_dataset_hm 变化（pp） | 数据集提升 / 下降 | 种子配对提升 / 下降 |
|---|---|---:|---:|---:|
| MaPLe | 原方法 | +1.6883 | 9 / 2 | 28 / 5 |
| MaPLe | 单模板融合 | +1.2255 | 10 / 1 | 29 / 4 |
| MaPLe | 数量匹配模板融合 | +1.1055 | 10 / 1 | 30 / 3 |
| MMRL | 原方法 | +0.7940 | 10 / 1 | 30 / 3 |
| MMRL | 单模板融合 | +1.1425 | 10 / 1 | 29 / 4 |
| MMRL | 数量匹配模板融合 | +1.1649 | 10 / 1 | 29 / 4 |

MaPLe 的 Oxford Pets 和 Stanford Cars 相对原方法分别下降约 0.1099 / 0.1282 pp。MMRL 的 Stanford Cars 从 77.9109 降至 76.6179，下降约 1.2930 pp，三个种子均下降。两种底座的平均 Novel 准确率提高，同时平均 Base 准确率下降；所有四配置结果均保留。

## 文件

| 文件 | 内容 |
|---|---|
| [seed_results.json](seed_results.json) | 66 组、四配置的 Base/Novel 正确数和样本数、逐种子准确率及 HM |
| [dataset_results.csv](dataset_results.csv) | 两底座 × 十一集 × 四配置，共 88 行；先平均三种子 Base/Novel 再计算每集 HM |
| [summary.json](summary.json) | 两种宏指标、配对分析、负结果和源汇总文件的 SHA-256 |
| [protocol.json](protocol.json) | 冻结官方源码版本、逐集训练轮数与 batch、输入资产哈希及融合/统计规则 |

`seed_results.json` 的正确数和样本数是结果复算输入；可按上面的定义独立重算 CSV 与宏汇总。仓库根目录 `tools/summarize_results.py` 对应原 CoOp 主结果，新增数据格式单独保存。

## 配方与来源

- [MaPLe](https://github.com/muzairkhattak/multimodal-prompt-learning/tree/69bce21ae8eda80ad6187534b2dce09cf6c59e17)：固定提交 `69bce21ae8eda80ad6187534b2dce09cf6c59e17`；十一集均 5 epochs、batch 4、context length 2、prompt depth 9、fp16。
- [MMRL](https://github.com/yunncheng/MMRL/tree/27d8821f6ac135a1d0c8e4c5ed78af14df0cd82e)：固定提交 `27d8821f6ac135a1d0c8e4c5ed78af14df0cd82e`；ImageNet 为 5 epochs、batch 32，其余为 10 epochs、batch 4、官方 AMP。EuroSAT 保留 `rep_dim=2048`、表示投影层和官方逐集正则参数。本轮方法为 CVPR 2025 MMRL。
- MMRL 原生 Base 推理为 `0.7 * class_logits + 0.3 * representation_logits`，Novel 为 class logits；融合以该原生后验为一侧。
- 类别描述来自 [ATPrompt](https://github.com/zhengli97/ATPrompt) 发布的固定文本资产，每类最多 20 条，经 CLIP 分词截断与去重、逐向量归一化、平均后再归一化。普通模板控制使用同一原型聚合方式；数量匹配控制按有效描述数量匹配。没有按测试结果挑选描述。

ATPrompt 学习带锚定属性的结构化提示；CDPF 将训练好的提示分类器后验与冻结描述分类器后验融合。复用描述文本需引用其原始来源，文本来源与算法贡献分别说明。

## 推理开销

2026-11-12 已完成既定的四项代表性测量：单张 RTX 4090，两种底座各在 UCF101 和 ImageNet 的 Novel seed1 上，使用相同的前 64 个真实图像变换；batch 1 使用首个样本。每种 batch、每个分支分别预热 20 次并记录 100 次 CUDA-event GPU prediction 间隔，共 **32 条分支记录、3,200 个原始值**。

下表为同一共同常驻上下文下的延迟中位数，单位为 ms/批；增幅为两个统计量之比，不是逐样本配对比。

| 数据集 | 底座 | batch | 原生分支 | CDPF | 中位延迟增幅 |
|---|---|---:|---:|---:|---:|
| UCF101 | MaPLe | 1 | 15.7292 | 22.6686 | +44.12% |
| UCF101 | MaPLe | 64 | 35.7999 | 66.3880 | +85.44% |
| UCF101 | MMRL | 1 | 7.9104 | 13.5695 | +71.54% |
| UCF101 | MMRL | 64 | 77.9023 | 107.6755 | +38.22% |
| ImageNet | MaPLe | 1 | 68.7045 | 79.4309 | +15.61% |
| ImageNet | MaPLe | 64 | 81.3269 | 110.1041 | +35.38% |
| ImageNet | MMRL | 1 | 6.2684 | 15.6129 | +149.07% |
| ImageNet | MMRL | 64 | 77.8695 | 107.6315 | +38.22% |

[完整 JSON](timing_20261112.json)提供全部 raw_ms、mean/median/p95、环境/精度与输入/源码哈希、原生及融合图像前向次数和共同上下文内存；[32 行 CSV](timing_results.csv)同时列出四分支及 mean/median/p95 相对原生分支的变化。single/matched/description 分支的原始波动均保留。ImageNet/MaPLe batch1 的 p95 从 101.2173 变为 100.4373 ms（约 −0.77%），不能据此宣称稳定加速；其 median 与 mean 均增加。

测量使用每种方法原本的推理数值路径：MaPLe 的原生输出 dtype 为 float16，逐次文本编码计入；MMRL 的原生输出 dtype 为 float32，保留官方预热文本缓存。所有融合每次重新执行完整原生前向，再执行一次独立原始冻结 CLIP 图像编码、归一化、原型打分、分别 softmax、固定 0.5 平均和 argmax。仅缓存辅助文本原型；不复用含学习视觉提示的图像特征。

这是重复相同驻留 batch 的 GPU prediction 测量，排除 CPU 预处理、图像/归档 IO、host-to-device、模型/checkpoint 加载、辅助文本原型准备与结果写入。各方法内部使用同一 checkpoint 和输入；不同方法保留各自原生数值/缓存策略，不能将表中绝对耗时作为统一实现或统一精度的跨方法比较。范围限于这两个数据集、seed1 Novel 和指定硬件，不推广为全部十一数据集吞吐或冷启动请求延迟。

显存是所有四分支共同常驻上下文的实测快照：完整原生训练器（含其自有辅助编码器、优化器对象和原生缓存）、独立原始 CLIP、三组文本原型、input64 始终在内存中；MMRL 还包括其原有 image_encoder_clip。allocated/reserved/peak 字段不表示原生独立部署或融合新增显存。

原 CoOp 的 0.74% 只描述其共享图像特征的已测设置。多模态扩展的证据支持无需额外训练的准确率收益，同时显示新增图像前向带来实际推理成本。

这些是本地同检查点受控比较。文献报告值涉及各自训练预算和数据实现，需要单独标注；本目录结果不构成统一协议的完整领域排名。
