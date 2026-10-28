# 当前融合方法：核心三数据集证据

本目录保留 UCF101、DTD 和 FGVC-Aircraft 三个数据集、三个随机种子的当前方法结果。

保留的四个分支为：

- `student`：同批 CoOp；
- `description_mean`：冻结类别描述；
- `simple_fusion`：固定 logit 均值融合；
- `mean_probability_fusion`：论文采用的固定概率融合，`alpha=0.5`。

`analysis.json` 是投稿结果生成器读取的机器可读来源；三个 `audit/fusion_*.json` 是对应数据集的融合审计。CSV 文件仅包含上述当前分支及其直接比较，不含已经停用的模型与诊断分支。
