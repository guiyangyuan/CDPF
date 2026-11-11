# KgCoOp 上的 CDPF 对照结果

本地实验，2026-11-11 同步。CLIP ViT-B/16，16-shot Base-to-Novel，十一数据集 × seed 1/2/3，共 33 组。本结果已合并后来完成的 ImageNet 三种子，模型、逐图预测、标签/顺序、日志与备份已通过独立验收。

每组使用同一个 epoch-100 checkpoint 比较原方法、单模板后验融合、数量匹配模板后验融合和类别描述后验融合。融合固定 `alpha=0.5`，不增加训练，也不按测试结果选择权重或描述。

## 两种 HM 汇总

先分别平均每个数据集的三个种子 Base 和 Novel，再计算该集 HM；两个宏指标分别为：

```text
B_d = mean_seed(100 * base_correct / base_total)
N_d = mean_seed(100 * novel_correct / novel_total)
H(B, N) = 2 * B * N / (B + N)
mean_dataset_hm = mean_dataset(H(B_d, N_d))
hm_of_mean_base_novel = H(mean_dataset(B_d), mean_dataset(N_d))
```

准确率和 HM 单位为 %，差值单位为百分点（pp）。三种子先均 Base/Novel 后 HM，不以先算逐种子 HM 再平均来替代；表格显示四位小数。

| 配置 | 平均 Base | 平均 Novel | mean_dataset_hm | hm_of_mean_base_novel |
|---|---:|---:|---:|---:|
| 原方法 | 81.0940 | 73.4305 | 76.7548 | 77.0722 |
| 单模板后验融合 | 78.3674 | 74.3693 | 76.1183 | 76.3160 |
| 数量匹配模板后验融合 | 78.2247 | 75.3705 | 76.6442 | 76.7711 |
| CDPF 描述后验融合 | 79.0165 | 77.5653 | 78.1752 | 78.2842 |

CDPF 相对原方法的 `mean_dataset_hm` 提高 **1.4204 pp**，相对单模板与数量匹配模板分别提高 **2.0570 / 1.5311 pp**。在另一宏口径下，原方法为 **77.0722**，CDPF 为 **78.2842**。两个口径分别报告。

相对原方法，CDPF 在 **8/11 个数据集**、**26/33 个种子配对**上提高 HM。Oxford Pets、Flowers102 与 Stanford Cars 下降，全部下降结果保留在下表和结构化数据中。以上是描述性配对计数，不代表独立性或显著性检验。

## 十一数据集结果

每个单元格为该数据集先平均三个种子 Base/Novel 后的 HM。

| 数据集 | 原方法 | 单模板融合 | 数量匹配模板融合 | CDPF | CDPF − 原方法（pp） |
|---|---:|---:|---:|---:|---:|
| Caltech101 | 96.0936 | 95.9241 | 95.8703 | 96.5586 | +0.4650 |
| Oxford Pets | 96.1809 | 95.7615 | 95.5881 | 96.0449 | -0.1360 |
| Flowers102 | 83.3769 | 81.6462 | 81.1549 | 83.2497 | -0.1272 |
| FGVC Aircraft | 35.2578 | 35.2693 | 34.8624 | 38.0577 | +2.7999 |
| DTD | 66.2006 | 66.1079 | 66.9886 | 70.5772 | +4.3766 |
| EuroSAT | 71.7202 | 70.6821 | 74.9340 | 78.4177 | +6.6975 |
| Stanford Cars | 73.5554 | 72.3869 | 72.8998 | 72.2604 | -1.2950 |
| Food101 | 91.1196 | 90.9196 | 91.0587 | 91.2699 | +0.1503 |
| SUN397 | 78.5389 | 77.3065 | 77.8302 | 79.5341 | +0.9951 |
| UCF101 | 79.6433 | 79.4050 | 79.3534 | 80.3450 | +0.7018 |
| ImageNet | 72.6156 | 71.8918 | 72.5458 | 73.6126 | +0.9970 |

前十数据集阶段的 `77.1687 → 78.6315` 不包含 ImageNet，不能标为十一数据集结果；本页以完整十一集汇总为准。

## 配方与来源

固定 [KgCoOp 官方提交](https://github.com/htyao89/KgCoOp/tree/c43ccc32c8739f64b165cfa205fcc6c0f5bb5b17) `c43ccc32c8739f64b165cfa205fcc6c0f5bb5b17`，采用 100 epochs、train batch 128、4 context tokens、知识正则权重 8、fp16、SGD 与 cosine schedule，使用 epoch-100 最终 checkpoint，无验证集选模。

ImageNet 保持 batch 128 与原始训练配方，增加 activation checkpointing 和原始 JPEG 的索引/读取包装。执行前已核原始/重计算 logits 与正则损失等价、有限优化更新，并用官方 devkit 核验验证图像标签。Base/Novel 类别、原图字节和图像变换保持原协议；前十集已验收模型没有重训。

类别描述复用 [ATPrompt](https://github.com/zhengli97/ATPrompt) 发布的固定文本资产，每类最多 20 条，经 CLIP 分词截断与去重、描述向量归一化、均值后再归一化。普通模板分支采用同样的原型聚合，匹配模板控制按有效描述数量匹配。描述来源与 CDPF 的后验融合贡献分别说明。

KgCoOp 仅学习文本提示，原方法与辅助分支共享冻结 CLIP 图像特征。本次扩展没有重新测量 KgCoOp 的新增 GPU 延迟或显存，不将 CoOp 的 0.74% 或 Ding 对照的计时作为 KgCoOp 实测值。

## 文件与复算

| 文件 | 内容 |
|---|---|
| [seed_results.json](seed_results.json) | 33 组、四配置的 Base/Novel 整数正确数、样本数、逐种子准确率及 HM |
| [dataset_results.csv](dataset_results.csv) | 十一数据集 × 四配置，共 44 行，含未舍入逐集 Base/Novel、HM 和差值 |
| [summary.json](summary.json) | 两种宏口径、同种子配对与负结果、验收源文件 SHA-256 |
| [protocol.json](protocol.json) | 冻结官方源码、官方配方字段、ImageNet 执行方式与融合/统计规则 |

使用整数正确数和样本数按上面的定义即可独立复算。仓库根目录的 `tools/summarize_results.py` 对应原 CoOp 主结果；KgCoOp 数据单独保存。本地训练结果与文献报告值分别标注；不同研究范围不混合计算宏指标。
