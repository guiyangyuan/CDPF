# 类别名基础上的图像条件描述偏差：固定小实验

2026-09-14。用户“可以你继续”授权继续上轮提出的错例诊断和模型改进，沿用现有AutoDL开卡、固定训练、备份、关机及GitHub归档授权。不自动扩量。训练前固定本文与源码。

## 改动

上轮类别名适配38.5%，两种静态描述37.5%。[错例诊断](../reports/diagnostics/text-adamw-errors-20260914/README.md)显示描述收益伴随近似数量的新增错误，当前类内描述均值不随图片变化。本轮保留类别名输入，用当前图片选择描述，并检验减去描述均值的效果。

令单位图像特征为f，单位类别名文本为n_c，单位描述为d_cv，mask为m_cv。h(t)=ReLU(W_down t)，均匀权重u_cv=m_cv/sum_v(m_cv)，冻结CLIP尺度s=exp(logit_scale)。

```
a_bcv = masked_softmax_v(s * dot(f_b, d_cv))
attentive: delta_bc = 0.2 W_up [h(n_c) + sum_v a_bcv h(d_cv)]
centered:  delta_bc = 0.2 W_up [h(n_c) + sum_v (a_bcv-u_cv) h(d_cv)]
R_bc = s * dot(f_b, delta_bc)
Z = S + R
loss = CE(S,y) + CE(stopgrad(S)+R,y)
```

主候选centered在描述权重均匀时精确退回类别名适配函数；两种模式都使用相同W_down/W_up，没有额外头、类索引参数或门控参数。减均值去除的是代数上的均匀描述分量，不保证等价于“去掉所有无用语义”。图像注意力使用原冻结CLIP尺度，不新增温度搜索。

保留类别名作为输入不代表已训练类别名适配器的输出受到保护：本轮两种新模型重新训练共享矩阵，类别名路径也可能变化。只有学生S的梯度严格来自自身CE，沿用原SGD和调度；残差仍为AdamW lr0.001、wd0.0005恒定学习率。67,584可训练参数，和上轮相同。

新入口`CoOp_TextDetail`，`TEXT_DETAIL_POOLING=attentive/centered`。旧入口/模板不改；新模式和公式进入checkpoint签名。Novel时从当次候选类别重建文本，训练类别缓存不随模型加载覆盖。此次不做Novel/test推理。

## 固定训练与比较

- 两种新模式，各从原始CLIP及相同初始化规则训练20轮，再独立重载最后一轮导出S/M/R/Z/F。Aircraft seed1、Base50类、16-shot800训练图、200验证图，ViT-B/16 FP32，batch16，原随机裁剪/翻转、20描述。
- 直接复用上轮e442f75的类别名/原始描述/近邻描述原始预测作冻结参照，文件hash锁定；没有为了对照再训练第三个模型。两新模型的S/M/F必须与历史逐元素一致，样本/类别顺序也必须相同，否则停止比较。历史和本轮不同源码身份明确保留。
- 主比较：centered.Z必须比类别名Z、旧描述Z、旧近邻Z、新attentive.Z以及S/F都至少高1pp（200图净多对2张）。不因为主比较失败而改主假设。
- 次比较：attentive.Z相对三个历史Z以及S/F均至少+1pp。若仅它通过，不能宣称中心化有效。
- 两项是开发筛查条件，不是显著性检验、Novel泛化或论文录用门槛。该验证集已多次用于开发；任何潜在信号都需要新种子/数据集确认。
- 固定两训练/两导出，最多30分钟队列，无自动重试、超参搜索、数据集/种子扩量。完成后完整备份两个模型和优化器，再关闭现有实例。

## 最小检查

只运行`tests.repro.test_text_detail`两项短检查，0.693秒通过：均匀权重极限、padding屏蔽、图片依赖、类别重排/独立批处理；两个更新的学生保持、同参数、两优化器恢复、完整重载、不同类别数量与签名拒绝。无全套测试或服务器重复合成检查。[记录](../reports/preparation/text-detail-20260914/checks.json)。

训练入口：`PYTHONPATH=.:Dassl.pytorch python scripts/repro/text_detail_pilot.py run --data-root DATA --clip-path CLIP --output-root FRESH_OUTPUT`。需CUDA；不得复用已有输出或复跑历史队列。

## 执行结果（训练后追加，原规则未改）

固定两组已完成，8.71分钟：attentive.Z40.0%、centered.Z38.0%，历史类别名Z38.5%。主假设失败、预登记次假设通过；新旧S/M/F完全一致。36文件和两个完整模型已备份核验，AutoDL网页确认已关机，备用定时已取消。禁止复跑本批，未追加种子/数据集。后续建议补同样保留类别名的均值描述对照，再做多种子/Novel确认；当前不能把组合增益全归因于图像注意力。见[完整报告](../reports/experiments/text-detail-20260914/README.md)。
