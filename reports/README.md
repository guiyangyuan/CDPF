# CDPF 实验与复核材料

这里保留 CoOp 主实验，以及 KgCoOp、MaPLe/MMRL 扩展的 CDPF 结果、协议和审计材料。

| 材料 | 入口 |
| --- | --- |
| MaPLe/MMRL 代表性推理计时（4 组案例、32 分支，2026-11-12） | [测量范围与汇总](../results/multimodal_20261110/README.md#推理开销)、[全部原始值](../results/multimodal_20261110/timing_20261112.json)、[CSV](../results/multimodal_20261110/timing_results.csv) |
| KgCoOp 十一数据集三种子四配置（33 组，2026-11-11） | [完整结果与双 HM 口径](../results/kgcoop_20261110/README.md)、[逐种子正确数/样本数](../results/kgcoop_20261110/seed_results.json) |
| MaPLe/MMRL 十一数据集三种子四配置（66 组，2026-11-11） | [完整扩展结果与双 HM 口径](../results/multimodal_20261110/README.md)、[逐种子正确数/样本数](../results/multimodal_20261110/seed_results.json) |
| 十数据集结果与独立复核 | [StanfordCars / SUN397 汇总](experiments/斯坦福汽车与SUN397扩展验证-20260917/结果报告.md) |
| ImageNet 三种子协议、配置、指标与模型哈希 | [ImageNet 结果包](experiments/ImageNet扩展验证-20260918/README.md)（仓库路径：`reports/experiments/ImageNet扩展验证-20260918/`） |
| 核心三数据集融合结果 | [结果与融合审计](experiments/attribution-study-20260911-42527/README.md) |
| 五数据集主表与逐种子统计 | [统一结果表](experiments/投稿实验统一结果-20260915/统一结果表.md) |
| 零样本与本地 CoCoOp 基线 | [Zero-shot CLIP](experiments/投稿零样本基线-20260916/结果报告.md)、[CoCoOp 核心三数据集](experiments/联合上下文提示基线-核心三数据集-20260916/结果报告.md)、[CoCoOp 扩展两数据集](experiments/联合上下文提示基线-扩展两数据集-20260916/结果报告.md) |
| 推理效率 | [实测与独立审计](experiments/统一推理效率-20260916/结果报告.md) |

原 CoOp 十一数据集等权平均 HM 为 `78.33`。KgCoOp 的对应 Macro HM 从 `76.75` 提高到 `78.18`（+1.42 pp），先平均 Base/Novel 再求 HM 的另一口径为 `78.28`。MaPLe/MMRL 的对应 Macro HM 分别由 `78.09/80.89` 提高至 `79.78/81.68`；先平均 Base/Novel 再求 HM 的另一口径分别为 `79.93/81.80`，两种口径单列。本地实验与文献公开结果分开标注，当前结果不宣称 SOTA。

`preparation/` 保留官方 CoOp 来源、上游差异、冻结协议和数据来源校验。服务器开关机记录、重复命令计划和下载过程日志已移出当前工作树。`experiments/` 中的结果 JSON、CSV、配置、指标、manifest 和复核记录继续保留；生成主结果仍需要其中的原始汇总输入。

完整实验压缩包从当前工作树移出，其原路径、字节数、SHA-256 和历史版本链接见 [归档索引](archived-artifacts.json)。历史失败路线及旧运维材料可从 [清理前提交](https://github.com/guiyangyuan/CDPF/tree/3f9b4be38adebf05418d663a597f9d2e873b530a/reports) 找回；未重写 Git 历史。旧收据仍记录归档原路径，请按索引取回后核对哈希。
