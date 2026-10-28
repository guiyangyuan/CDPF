# 历史材料与版本

当前默认源码树面向 **CDPF: Class-Description Posterior Fusion for Base-to-Novel Recognition**。它保留当前训练核心、类别描述、结果汇总和必要来源证据。

历史材料可从整理前的固定提交 [3f9b4be38adebf05418d663a597f9d2e873b530a](https://github.com/guiyangyuan/textconsistency_v2_tune/tree/3f9b4be38adebf05418d663a597f9d2e873b530a) 查阅。该链接固定在历史版本，不随默认分支更新。

## 旧材料的位置

| 材料 | 历史位置 |
|---|---|
| 旧论文源码、PDF、中文对照及 Overleaf 替换副本 | [论文投稿版本-20260917/](https://github.com/guiyangyuan/textconsistency_v2_tune/tree/3f9b4be38adebf05418d663a597f9d2e873b530a/论文投稿版本-20260917) |
| 旧项目状态、研究与运行过程 | [PROJECT_STATUS.md](https://github.com/guiyangyuan/textconsistency_v2_tune/blob/3f9b4be38adebf05418d663a597f9d2e873b530a/PROJECT_STATUS.md) |
| 停止的模型、探索计划、旧图表及运维记录 | [历史源码树](https://github.com/guiyangyuan/textconsistency_v2_tune/tree/3f9b4be38adebf05418d663a597f9d2e873b530a) |

历史目录中的“当前”“最终”“下一步”和服务器状态只适用于记录当时。旧 TextConsistency/V2 FG 稿件、层级聚合、KD、残差和门控研究不属于当前 CDPF 方法；旧表格不能与十一数据集 CDPF 主表混用。

旧稿的多个源码目录曾并行存在，部分正文不同。它们保留在 Git 历史中用于溯源，当前论文版本与同步规则见 [paper/README.md](../paper/README.md)。

## 历史源码与结果核验

实验清单中的训练源码 SHA-256 以对应实验的冻结文件集合为准；它与 Git commit ID 是不同标识。当前十数据集评估器固定绑定精简后的训练源码，用于该版本新训练的 checkpoint。已报告的旧实验则需要其对应的历史评估器、冻结源码、checkpoint 和配置。整理前提交提供历史查阅入口，但不保证其中所有文件都对应某一次实验的冻结源码哈希。

当前精简树不应冒充旧冻结源码。重新评估旧 checkpoint 需取得与记录匹配的源码和资产，保留校验，不通过替换预期哈希跳过检查。当前可直接执行的结果复算及尚缺的公开资产见 [REPRODUCIBILITY.md](REPRODUCIBILITY.md)。
