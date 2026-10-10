# TextConsistency

本仓库当前只维护论文采用的固定概率融合方法：训练标准 CoOp 提示，在推理时把 CoOp 类别概率与冻结类别描述概率按 `alpha=0.5` 等权平均。

## 当前方法

- 训练器：`CoOp_TextConsistency_Ablation`，只训练共享 CoOp 上下文提示。
- 描述分支：冻结 CLIP 文本编码器，每类最多使用 20 条类别描述并取均值。
- 推理：`0.5 * softmax(student_logits) + 0.5 * softmax(description_logits)`。
- 新增可训练参数：0；描述分支保持冻结。
- 主评估：十个 Base-to-Novel 数据集、三个随机种子，平均 HM `78.75`。
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
- 推理效率：`reports/experiments/统一推理效率-20260916/`
- 零样本与联合上下文基线：`reports/experiments/投稿零样本基线-20260916/`、`reports/experiments/联合上下文提示基线-核心三数据集-20260916/`、`reports/experiments/联合上下文提示基线-扩展两数据集-20260916/`

## 结果边界

当前十数据集结果不包含 ImageNet。仓库历史中的 ImageNet 数字来自已经停用的旧模型，不能作为当前固定融合方法的结果，也不能据此宣称标准十一数据集最优。论文因此只报告十数据集结果，并明确区分本地复现实验与文献公开结果。
