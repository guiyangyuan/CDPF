# 方法与实验表述边界

2026-09-10。描述当前代码实际执行的算法。UCF101 三种子结果见 [项目进度](PROJECT_STATUS.md)；创新性、跨数据集收益和各组件贡献尚未全面验证。论文 PDF/Overleaf 尚未修改。

## 计算路径

冻结 CLIP 图像/文本编码器及 logit scale。图像生成归一化全局向量 v；共享可学习 soft context 与类别名经文本编码器得到 student 类别向量 s_c，student logits 为 exp(logit_scale) · vᵀs_c。冻结文本编码器参数并不切断 soft context 的梯度。

每类文本描述来自仓库 JSON。先按原有顺序规范空白、去重并取最多 N_VIEWS 条，再对 CLIP 77-token 截断后相同的 token 序列去重。不足的视图使用零 padding 和布尔 mask，不重复首条描述。描述向量预先编码并冻结。

- mean：对有效描述向量求均值，再归一化。
- flat：图像查询与每条描述的可学习 key 做注意力，mask 后加权、归一化。
- hierarchical：将描述按顺序分成连续块，先组内注意力，再组间注意力，最后归一化。采用向上取整的块长度与 mask；不丢弃非整除尾部。分组是计算结构，未标注外观/属性/上下文/辨别性四种语义。
- MLP 开启时：查询输入为 normalize(v + MLP(v))。它处理全局向量，用于描述注意力；最终分类相似度仍使用原始 v。代码没有 patch 级/区域级特征抽取，不应称其为已实现的局部细粒度建模。

得到图像条件下的类别向量 a_c(v)，teacher logits 为 exp(logit_scale) · vᵀa_c(v)。这里 teacher 是训练中的自适应描述分支，不是独立预训练完成的教师网络。

## 损失与梯度

总损失：

L = CE(z_s, y) + λ_a CE(z_t, y) + λ_k T² KL(stopgrad(softmax(z_t/T)) || softmax(z_s/T)) + λ_e Σw log w。

adaptive CE 训练注意力/MLP，KD 训练 student 且不向 teacher 反传。默认 λ_a=1、λ_k=1、T=2、λ_e=0。baseline 无 teacher，实际只用 student CE；mean 的 teacher 不含可学习聚合参数。

负熵 Σw log w 的正权重会鼓励分布更均匀。它既不能保证文本语义多样性，也不是冻结图像之间的对比学习。旧的“对比损失”只依赖冻结特征，没有需要优化的参数，因此本次入口禁止启用，论文应删除相应已验证贡献的表述。

## 推理和实验解释

- student：仅 z_s。
- teacher：仅 z_t。
- ensemble：(1−α)z_s + αz_t，默认 α=0.5，是 logits 加权。

同一分支/α 同时用于 Base 与 Novel，从同一个完整 checkpoint 加载。不同分支作为额外分析单列。不能将 student 的 Base 与 teacher 的 Novel 拼成一个方法的 HM；不能仅凭 ensemble 的提升断言 KD 改善了 student。

当前 six-variant pilot 只拆分描述聚合与全局 MLP。要论证 KD，需要在同一架构/数据/推理下补 KD=0 对照；要论证分组结构，需至少比较 flat 与 hierarchical，并承认两者参数量也不同。若要主张语义层次，还需补真实组标签或重新设计并重新实验。

## 论文数字处理

原论文和旧结果文档保留作历史记录。旧 checkpoint 不完整、seed 标签不符、分支混用的结果不能直接作为此次修订版结果。每个待投稿数字必须能追溯到固定代码、有效配置、数据/文本版本、实际 seed、完整 checkpoint 和评估日志。

程序正确性证据包括行为测试、小型模型集成、官方 CLIP 的真实图像一步训练/独立进程重载、4 类 CLI smoke，这些 smoke 不可填写性能表。另已完成完整 UCF101 三种子实验，见 [真实结果报告](reports/experiments/TextConsistency-UCF101三种子实验报告.md)。二者必须分开表述。

三种子表同时报告 Base/Novel 的均值与标准差。HM 需明确采用“Base 与 Novel 均值的调和平均”还是“每 seed HM 的均值”，不要混用；脚本同时输出两者供核对。

扩展数据集、跨数据集/域泛化、更多 backbone、训练效率、显存、统计显著性和最终期刊适配均未完成验证。
