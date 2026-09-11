# 关卡后的错误诊断与下一步决策

2026-09-11。本次完成本机只读诊断、相关工作核对和最小方案设计。没有连接已关闭的服务器，没有训练、加载完整 CLIP 或修改模型代码。GPU 批次仍以[上一轮完整报告](../../experiments/attribution-study-20260911-42527/README.md)为准。

**结论：描述信息有互补价值，但已训练教师与固定融合的组合还没有稳定优势。** 不值得立即增加层级、注意力头或新损失。建议先用已有模型做一次固定的交叉融合检查，区分学生改变和教师参与推理的影响；这只是诊断与精简候选，不是已经证明的论文创新。

## 已完成的检查

- 18 个逐样本预测文件，覆盖 UCF101、DTD、Aircraft，seed1/2/3，各自 Base/Novel 测试集，共 26,424 条记录、8,808 张唯一图片。相同测试图片在不同 seed 中重复出现，不能当作 26,424 个独立样本计算显著性。
- 预测、指标、完成状态和分析文件共 38 个输入文件，SHA 与已校验的完整备份逐项一致；分析前后再核对，原始数据未改变。
- 输出 126 行预测转移统计、54 行融合分解、42 行跨种子转移汇总，以及全部类别的统计。没有选最佳种子、类别或分支，也没有拟合权重或门控规则。
- 在 CPU 上比较 DTD/Aircraft 六对 KD0/KD1 最终 checkpoint；先核对 SHA 和 sidecar，再读取已验证的自产模型。所有教师参数逐张量完全相同，学生参数六对均发生变化。模型源码摘要与原训练版本 `9bd4a7f...` 一致。
- 为未来验证集检查核对 27 个旧模型、9 组 dataset/seed 输入；每组 CoOp、KD0、KD1 的训练/验证/测试样本哈希一致。UCF旧包 SHA 重新验证，6 个 UCF KD 模型与包内文件一致。该检查不等于真实 loader 检查，也不等于 GPU 就绪。

## 1. 简单融合纠正了什么，又带错了什么

“纠正”指原 CoOp 预测错、融合预测对；“带错”指原 CoOp 对、融合错。下面是每个 seed 的图片数再取三种子平均，因此允许小数；净准确率变化 = (纠正−带错)/该 split 图片数。使用固定 0.5 logits 融合，未试其他权重。

| 数据集 | 测试类别 | 每 seed 图片数 | 平均纠正数 | 平均带错数 | 准确率净变化（百分点） |
|---|---|---:|---:|---:|---:|
| UCF101 | Base | 1934 | 52.00 | 69.00 | −0.88 |
| UCF101 | Novel | 1849 | 176.00 | 26.00 | +8.11 |
| DTD | Base | 864 | 35.67 | 36.33 | −0.08 |
| DTD | Novel | 828 | 105.67 | 15.67 | +10.87 |
| Aircraft | Base | 1666 | 100.00 | 108.00 | −0.48 |
| Aircraft | Novel | 1667 | 183.67 | 48.00 | +8.14 |

这说明固定描述分支补回了相当多的 Novel 错误，同时也会覆盖部分正确的 CoOp 预测。Base 上纠正与带错接近，Novel 上净纠正明显较多。它支持研究分支互补性，**不代表我们已经知道每张新图片该信哪一个分支**；事后用真值挑对的分支不能当可部署方法。

融合也不是简单在两个 top-1 答案之间二选一。以 Aircraft Novel 为例，每个 seed 平均约 37.67 张图片上，CoOp 与描述均值都错，但融合得到了正确的第三个类别。其余数据集也存在这种情况。因此，“任一输入分支正确就算正确”的 oracle 只能是二选一规则的参考，不能叫作 logits 融合的性能上限。全部四种正确性组合见 [fusion-partitions.csv](evidence/fusion-partitions.csv)。

四格统计统一以 CoOp 与固定描述均值为参照；ADK 适配还包含图像条件描述信息，其四格不是对全部输入分支的正确性分解。

概率融合相对 logits 融合也会同时纠正和带错。Aircraft Novel 平均纠正 48.00 张、带错 50.67 张；不能用某一类别或某一分支的局部好结果宣称概率融合普遍占优。所有比较见 [transition-summaries.csv](evidence/transition-summaries.csv)，包含固定描述、图像相关描述、简单融合、概率融合、ADK 公式适配。

## 2. KD 掉点已经确认到哪一层

KD0/KD1 使用相同结构和训练协议。源码的 KD 目标是 `teacher.detach()`，只向学生传递梯度；教师由自己的分类损失训练。教师计算依赖冻结图像/描述特征及聚合器，不依赖学生 prompt。来源见 [training_losses](../../../trainers/textconsistency_common.py#L319)、[模型 forward](../../../trainers/coop_textconsistency_v2_fg.py#L91)、[训练调用](../../../trainers/textconsistency_runtime.py#L131)。

此次逐张量比较发现：六对模型中，教师的 787,970 个参数全部精确一致；学生 prompt 参数（共 2,048 个）在六对模型中均有变化，不表示逐个参数都必然改变。因此，本次 KD 开关下最终融合的差异不能解释为“KD 又把教师训练坏了”。它仍可能来自学生的分类边界、概率分布或与教师的互补性改变；这些机制尚未测出。

| 数据集 | 开 KD 后学生 Base Δ | 学生 Novel Δ | 融合 Base Δ | 融合 Novel Δ | 融合配对 ΔHM 均值 ± SD |
|---|---:|---:|---:|---:|---:|
| DTD | −4.05 | +7.25 | −3.70 | +1.57 | −0.49 ± 1.20 |
| Aircraft | −2.30 | +2.80 | −2.32 | +0.10 | −1.16 ± 0.90 |

准确率差是三种子均值之差；最后一列是逐 seed HM 差的均值，统计口径不同。直观上，学生端的 Novel 改善没有等量转化为融合端改善，而 Base 损失保留下来。这是观测事实；“学生与教师变得过于相似”仍只是解释之一，不能根据两个准确率数字证明。

## 3. 已训练教师是否比固定描述更好

| 数据集 | 固定描述均值 Base / Novel | 已训练教师 Base / Novel | 教师相对均值的 Base / Novel 差 |
|---|---|---|---|
| DTD | 60.07 / 66.91 | 69.75 / 60.87 | +9.68 / −6.04 |
| Aircraft | 30.49 / 40.61 | 32.65 / 39.01 | +2.16 / −1.60 |

教师在训练类别上的分类更好、在 Novel 上更差，符合“对 Base 的适配牺牲了部分迁移能力”的表现，但不是完整的过拟合因果证明。也不能因此用 Base/Novel 身份在推理时切换不同教师或不同权重。

## 4. 当前数据回答不了的问题

历史完整模型的评估只保存汇总指标；这轮六分支 JSONL 只保存预测类别，没有 logits。因而还不能测出：完整模型 KD0/KD1 的逐样本纠正/带错、学生与教师错误重合度、logits 大小与 margin、置信度校准、真实 prompt 梯度冲突。旧 checkpoints 足够以后补评估，无需重训来补这些信息。

当前已保存的文件都是被查看过的测试结果。此次用真值做事后描述，不拟合门控或阈值。下一步开发数据只能使用训练/验证部分；Base 验证集不是未见类别验证集，不能把它的准确率称作 Novel 或 HM。候选方法若继续，仍需固定方案后的独立确认实验。

## 决策与交付物

- [最小下一步方案](MINIMAL_NEXT_STEP.md)：只改变推理组合、保留全部交叉对照、先采 Base 验证集；写明通过和停止条件。
- [相关工作与重合边界](PRIOR_WORK.md)：核对 7 项直接相关工作；普通置信度加权、选择描述再蒸馏、冻结教师约束都不能直接当独有创新。
- [输入清单](evidence/proposed-probe-inputs.json)：27 个旧模型的路径、SHA、源码版本和样本哈希。状态是 `local_input_inventory_verified_not_gpu_ready`。
- [逐样本诊断](evidence/diagnostic.json)、[所有类别统计](evidence/class-transitions.csv)、[全部转移计数](evidence/transitions.csv)、[KD 张量核对](evidence/kd-tensor-comparison.json)。

目前值得继续的是一次有边界的机制检查；当前结果尚不支持追加大规模训练或宣称已找到新方法。完成上述诊断和方案是本次任务的交付，补评估尚未执行。

## 复算

在仓库根目录运行（`../../../` 指向原任务工作区，换机器时替换为实际目录；输出目录必须全新）：

```bash
python reports/diagnostics/error-analysis-20260911/analyze_predictions.py --study ../../../outputs/textconsistency-attribution-study-20260911-42527/unpacked/attribution-study-20260911-v1 --receipt ../../../outputs/textconsistency-attribution-study-20260911-42527/local-backup-receipt.json --output /tmp/tc-error-analysis-recheck
```

张量核对还需 PyTorch；只读取 SHA 验证后的本机自产 checkpoint：

```bash
python reports/diagnostics/error-analysis-20260911/compare_kd_checkpoints.py --study ../../../outputs/textconsistency-attribution-study-20260911-42527/unpacked/attribution-study-20260911-v1 --receipt ../../../outputs/textconsistency-attribution-study-20260911-42527/local-backup-receipt.json --history ../../../outputs/textconsistency-crossdataset-study-20260911-55424 --history-receipt reports/experiments/crossdataset-study-20260911-55424/backup-receipt.json --repo . --output /tmp/tc-kd-tensors-recheck.json
python reports/diagnostics/error-analysis-20260911/inventory_probe_inputs.py --workspace ../../.. --output /tmp/tc-probe-inputs-recheck.json
python reports/diagnostics/error-analysis-20260911/verify_evidence.py --workspace ../../.. --output /tmp/tc-diagnostic-verification.json
```

[最终核对收据](evidence/verification.json)检查所有126组计数与前批审计、2232行类别统计的54组加总、全部模型/manifest SHA，并验证篡改输入摘要会在写输出前被拒绝。四个工具均不连接服务器、不构建 CLIP、不训练，源备份保持不变。

执行中修正了首次相对路径多一级、UCF汇总目录为符号链接、原CoOp训练模板SHA为null这三处工具假设。检查在写结果前拒绝，未改变输入或放宽模型SHA约束；最终输出全部通过。CoOp没有使用描述进行训练，其模板null与KD模型有描述模板本来就是不同语义。
