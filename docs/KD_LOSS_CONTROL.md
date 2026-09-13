# H1：删除目标置信度匹配的受控消融

本实现服务于[已确定的H1方案](../reports/diagnostics/kd-failure-20260912/NEXT_STEP.md)，是已知KL分解的消融，尚不构成新的算法贡献。实现与CPU检查不代表GPU验证或精度提高。

## 损失定义

`TRAINER.TEXTCONSISTENCY.KD_LOSS_MODE` 有两个值：

- `full_kl`：默认行为，保留原来的温度化全分布KL。
- `weighted_nckd`：删除目标类别二元KL，保留原加权非目标项。

对一个样本，令`q=softmax(teacher/T)`，`q_rest`和`p_rest`分别为先删除真值类别、再对剩余logits计算的条件softmax。新项为：

`KD = (1-q_y) * T² * KL(q_rest || p_rest)`，随后对batch求均值。

教师条件分布和教师权重均detach。实现通过求和教师非目标概率计算`1-q_y`，避免浮点数中`q_y`接近1时的相减损失。学生CE与在线教师CE保持原样；本方案固定T=2、全局KD权重1、教师CE权重1、其他损失设置沿用原配置。该分解来自[DKD](https://arxiv.org/abs/2203.08679)，并非完整DKD最优设置的复现。

单个样本的KD对真值logit导数为零，不表示prompt参数梯度也会保持该分量不变。两类问题只剩一个非目标类，条件KL恒为零；因此非平凡集成测试需要至少三个类别。

## 入口与实验身份

沿用受控`train.py`入口及数据集原配置，通过YACS选项指定：

```text
TRAINER.TEXTCONSISTENCY.KD_LOSS_MODE weighted_nckd
```

不指定时仍是`full_kl`。不要用旧`run_pilot.py`启动H1；它承载历史全KL和Base/Novel评估协议，本轮未把新消融混入该队列。未来H1的Base验证及固定`SN+M_logits, alpha=.5`比较需单独准备。

新输出必须记录损失模式、温度、有效系数和非目标权重定义，完整checkpoint保存这些身份字段。重载必须保持配置和身份相符，不能把全KL checkpoint改名成H1。原`code_sha256`校验继续生效，旧权重仍配合各自冻结源码使用；新增配置不会自动赋予旧checkpoint新的身份。

汇总审计遇到新版损失身份或显式模式配置时，必须获得协议中的预期KD权重与温度，并核对各输出一致。只有全部可用记录均不含新版标记的历史全KL记录，才允许使用缺少这些字段的旧协议。

## 验证边界与后续

CPU检查包含损失值/导数、真实tiny CLIP架构的训练步与重载、历史数值logits上的独立分解核验。tiny模型和合成输入只用于代码正确性，不纳入论文精度表。现有logits也不代表新方法训练结果。

后续GPU预算仍按原方案：一次DTD seed1历史全KL anchor通过后，再做三个数据集各seed1的单一对照。固定失败停止规则，不搜索温度、KD强度或融合alpha；未通过即停止此次修补投入。通过也需先论证超出已有方法的贡献，再决定是否扩种子及做独立确认。
