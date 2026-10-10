# ImageNet 当前模型扩展验证

本目录同步自 2026-09-18 至 2026-09-19 在单张 NVIDIA GeForce RTX 4090 上完成的 ImageNet 三种子实验。有效服务器结果目录为：

`/root/autodl-tmp/textconsistency/outputs/ImageNet投稿主实验-20260918-45ff8ba-重跑1`

该实验使用当前保留的方法，而不是已停止的 V2 FG 模型：

- 训练器：`CoOp_TextConsistency_Ablation`
- 训练协议：ImageNet、16-shot、ViT-B/16、20 epochs、seed 1/2/3
- 推理：`0.5 * softmax(CoOp logits) + 0.5 * softmax(冻结均值描述 logits)`
- 融合权重：固定 `alpha=0.5`，不针对 ImageNet 搜索

## 三种子均值

| 方法 | Base | Novel | HM |
|---|---:|---:|---:|
| CoOp | 76.469 | 68.721 | 72.389 |
| 冻结均值描述 | 74.368 | 71.248 | 72.775 |
| 固定 logit 均值融合 | 76.669 | 71.531 | 74.011 |
| 固定概率融合 | **76.657** | **71.621** | **74.054** |

固定概率融合相对同批 CoOp 的 Novel 提升为 `+2.900` 个百分点，HM 提升为 `+1.665` 个百分点。

## 保留证据

- `实验协议.json`、`数据核验.json`、`执行计划.json`、`执行状态.json`
- `结果汇总.json` 与服务器生成的 `结果报告.md`
- 三个种子的运行清单、有效配置、训练指标和 checkpoint SHA-256
- 三个种子的最终 Base/New 融合评估清单和配置；当前四分支的逐种子指标统一保存在 `结果汇总.json`

服务器完整备份为 `/root/ImageNet完整结果备份-20260922.tar.gz`，其 SHA-256 记录于 `服务器备份.sha256`。服务器还保留原始 checkpoint、逐样本预测和完整日志，可用该备份进行深度审计。

## 未纳入普通 Git 的文件

以下大文件不提交到普通 Git 历史：

- 三个约 72 MB 的 `model.pth.tar-20` checkpoint
- 六个约 27 MB 的 `predictions.jsonl`
- 含已停止探索分支的原始详细融合 `metrics.json`；当前四分支结果已由服务器汇总文件完整保留
- 重复的完整命令日志
- 第一次启动失败的 `ImageNet投稿主实验-20260918-45ff8ba` 目录
- 修复类别同名校验前的失败/补丁前中间输出

这里保留 checkpoint 内容哈希和完整服务器备份哈希，因此省略大文件不会改变正式汇总结果的可追溯性。

## 结论边界

这是当前方法的本地复现实验结果，可以补齐标准十一数据集集合中的 ImageNet 项；它不是文献公开数字，也不自动构成 SOTA 结论。任何跨方法排名仍须使用完全一致的十一数据集协议逐项核对。
