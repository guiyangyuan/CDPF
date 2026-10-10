# 项目状态

截至 2026-10-10，当前树只保留论文采用的固定 `0.5/0.5` 概率融合方法及其复现、审计和投稿材料。

## 已完成

- 十一个标准 Base-to-Novel 数据集、三个随机种子的统一实验已完成；原十数据集平均 HM `78.75`，加入 ImageNet 后十一数据集等权平均 HM `78.33`。
- ImageNet 当前模型结果：Base `76.657`、Novel `71.621`、HM `74.054`，相对同批 CoOp 的 Novel `+2.900pp`、HM `+1.665pp`。
- 十数据集本地同协议对比：CoCoOp `76.15`，同批 CoOp `75.33`，Zero-shot CLIP `71.88`。
- 固定融合相对本地 CoCoOp 在已有的 10/10 数据集对照上为正；加入 ImageNet 后，相对同批 CoOp 在 33/33 个种子级比较中为正。
- 单张 RTX 4090 推理效率实测：batch 64 时固定概率融合相对 CoOp 延迟增加 `0.74%`，新增可训练参数为 0。
- 英文投稿源码、中文对照稿、正式表格、图和 PDF 均保留在 `论文投稿版本-20260917/`。

## 结论边界

- ImageNet 结果来自当前 `CoOp_TextConsistency_Ablation` 与固定 `alpha=0.5` 冻结描述概率融合，不使用已停止的 V2 FG 模型。
- 十一数据集结果现已完整，但本地结果与文献公开结果继续分开标注；正式跨方法排名前仍需核对每项协议和来源。
- 当前结果不宣称 SOTA，也不宣称超过 PromptSRC。

## 当前入口

- 训练器：`trainers/coop_textconsistency_ablation.py`
- 融合评估：`scripts/repro/evaluate_frozen_fusion.py`
- 十数据集结果：`reports/experiments/斯坦福汽车与SUN397扩展验证-20260917/`
- ImageNet结果：`reports/experiments/ImageNet扩展验证-20260918/`
- 效率结果：`reports/experiments/统一推理效率-20260916/`
- 论文源码：`论文投稿版本-20260917/最终提交源码/`
