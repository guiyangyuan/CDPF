# TextConsistency

本仓库当前只维护论文采用的固定概率融合方法：训练标准 CoOp 提示，在推理时把 CoOp 类别概率与冻结类别描述概率按 `alpha=0.5` 等权平均。

## 当前方法

- 训练器：`CoOp_TextConsistency_Ablation`，只训练共享 CoOp 上下文提示。
- 描述分支：冻结 CLIP 文本编码器，每类最多使用 20 条类别描述并取均值。
- 推理：`0.5 * softmax(student_logits) + 0.5 * softmax(description_logits)`。
- 新增可训练参数：0；描述分支保持冻结。
- 主评估：十一个标准 Base-to-Novel 数据集、三个随机种子；加入 ImageNet 后等权平均 HM `78.33`。
- 效率：单张 RTX 4090、batch 64 实测，相对 CoOp 延迟增加 `0.74%`。

## 主要入口

- 训练：`train.py`
- 当前训练器：`trainers/coop_textconsistency_ablation.py`
- 固定融合评估：`scripts/repro/evaluate_frozen_fusion.py`
- 投稿结果生成：`scripts/repro/生成投稿主结果.py`
- 当前论文：`论文投稿版本-20260917/最终提交源码/pag.tex`
- 中文对照稿：`论文投稿版本-20260917/论文投稿版_中文对照稿.md`

## 当前证据

- 核心三数据集：`reports/experiments/attribution-study-20260911-42527/`
- 十数据集统一结果：`reports/experiments/斯坦福汽车与SUN397扩展验证-20260917/`
- ImageNet 当前模型结果：`reports/experiments/ImageNet扩展验证-20260918/`
- 推理效率：`reports/experiments/统一推理效率-20260916/`
- 零样本与联合上下文基线：`reports/experiments/投稿零样本基线-20260916/`、`reports/experiments/联合上下文提示基线-核心三数据集-20260916/`、`reports/experiments/联合上下文提示基线-扩展两数据集-20260916/`

## 结果边界

当前模型的 ImageNet 三种子结果已从训练服务器同步并通过身份、协议、汇总指标和哈希核验；固定概率融合在 ImageNet 上取得 Base `76.657`、Novel `71.621`、HM `74.054`。它与原十数据集合并后的十一数据集等权平均 HM 为 `78.33`。这些数字是本地复现实验，不是文献公开结果；跨方法比较仍须逐项核对相同协议，当前结果不宣称 SOTA。
