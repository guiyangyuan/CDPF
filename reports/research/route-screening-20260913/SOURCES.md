# 两条候选路线的接近工作核对

核对日期2026-09-13。只使用论文原文、作者摘要、正式会议页面或作者仓库。下列“对本项目的判断”属于基于来源与本项目结果的推断，不是作者对本项目的结论。检索有边界，不证明新颖性，也不按跨协议论文分数排列优劣。

| 工作 | 本轮核对范围与已有内容 | 对本项目的判断 |
|---|---|---|
| [AAPE：Aggregate-and-Adapt，NeurIPS2024](https://arxiv.org/abs/2410.23698) | 作者摘要：图像对齐的描述聚合，以及靠近聚合摘要的prompt生成与任务学习。HTML全文本轮访问失败，细节不据失败补写 | 描述聚合加蒸馏已有直接近邻。不能将本项目自己的组件消融写成对AAPE作者方法的否证 |
| [BiomedCoOp，CVPR2025](https://arxiv.org/html/2411.15232v2#S3.SS3) | 方法§3.3–3.4：图文相似度评分、MAD离群提示筛除及知识蒸馏；医学BiomedCLIP背景 | 筛掉不可靠描述再KD已有先例；不同领域不自动构成算法新颖性，亦不能拿其医学结果作本项目同协议分数 |
| [CA-MKD，ICASSP2022](https://arxiv.org/abs/2201.00007) | 作者摘要：借助真值为教师预测赋予逐样本可靠性，结合中间层传递 | “教师对真值概率高就多蒸馏”不作为本轮新点；训练可用标签不等于推理可用标签 |
| [DKD，CVPR2022](https://arxiv.org/abs/2203.08679) | 作者摘要及既有分解核对：目标/非目标蒸馏分解与解耦 | H1是已知分解的一项消融。局部改善也不能将其重新命名为新损失 |
| [ADK，2025预印本](https://arxiv.org/abs/2512.17313) | 作者摘要：离线描述均值与图像相关无参数描述注意力，结合原PEFT分支；公式细节沿用已有审计 | 简单组合与实例相关描述已经存在。仓库历史ADK结果是公式适配，不能改称作者复现；本轮未重新核验其Git仓库是否更新 |
| [CAPEL，作者标注Pattern Recognition2025](https://arxiv.org/html/2510.09867v1) | 摘要、引言及方法组织：保留多提示分类结构、logits集成、簇保持正则和提示自适应加权；有计算成本比较 | 普通提示加权或“保持描述多样性”不足以形成区别；成本研究也须有明确的同协议问题，不能仅说以前没人量过成本 |
| [2SFS，CVPR2025](https://arxiv.org/abs/2503.11609) | 作者摘要：分析少样本适配的阶段性动态，在固定计算预算中分配PEFT与线性分类器训练；固定超参跨多设置评估 | Base适配与Novel迁移的取舍已有研究；当前只限固定20轮协议，不能宣称长训必然过拟合或发现全部训练阶段因果 |
| [Rethinking Few Shot CLIP Benchmarks，ICCV2025](https://openaccess.thecvf.com/content/ICCV2025/papers/Kravets_Rethinking_Few_Shot_CLIP_Benchmarks_A_Critical_Analysis_in_the_ICCV_2025_paper.pdf) | 正式页面检索结果及PDF摘要：研究预训练接触导致的归纳评估问题，并提出遗忘式评估。HTML后续直开403，未把访问失败当作证据 | 新增Pets/EuroSAT只表示未用于本轮决策的确认域，不表示CLIP预训练没有见过其数据或类别；本项目不复现该遗忘框架 |
| [VDT-Adapter，ICCV Workshop2023](https://arxiv.org/abs/2307.11661)；[作者代码](https://github.com/mayug/VDT-Adapter) | 作者摘要及README：视觉描述增强与少样本描述选择。作者仓库可见trainers/configs、GPT-4数据和MIT许可；README要求按dataset/shot调residual ratio | 可作为后续外部实现审计对象，本轮未安装或复现。必须区分作者预算与统一预算，不能直接沿用按测试集选好的残差比例，也不能把其低预算适配称作者原表复现 |

## 本轮决策

A路线的“可靠性筛选/加权”没有找到可明确解释、可隔离验证的新增机制，故不进入实现。B路线可以复用已有受控证据，但只是一项投稿选题候选；近邻都有组件或效率分析，本项目仍需形成特定问题、统一预算对照、跨域确认和真实成本测量。没有足够证据时，应缩小或放弃论文主张，不能依靠换标题把内部复盘变成原创研究。

## 检索记录与范围

查询组合包括：`CLIP class descriptions redundancy duplicate prompts few shot classification`、`CLIP prompt ensemble textual descriptions budget selection diversity`、`CLIP textual descriptions empirical study aggregation distillation`、`CLIP simple fusion distillation descriptions`、`Rethinking Few-Shot Farina vision language`。先看检索结果，再追溯上述第一手来源。

发现但未作为决策依据：AdaBoosting Text Prompts作者项目页、若干新的应用/鲁棒性论文。没有核对具体版本和完整方法，不将它们列为已排除的相同工作。没有穷尽所有期刊/预印本，没有预测录用概率，没有找到或确认某个期刊的毕业认可资格。

已有两轮资料仍保留：[先前相关工作边界](../../diagnostics/error-analysis-20260911/PRIOR_WORK.md)、[蒸馏修补文献](../../diagnostics/kd-failure-20260912/LITERATURE.md)。本轮仅新增/复核与两个候选有关的信息，不重复给旧实验赋予新分数。
