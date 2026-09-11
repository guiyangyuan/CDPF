# 相关工作与创新边界

核对日期：2026-09-11。以下依据论文原文、正式会议页面或作者仓库，不是穷尽性新颖性检索，也不据跨协议表格判断谁优于谁。本轮没有运行这些作者实现。

| 工作与主要来源 | 已有机制 | 对当前方向的含义与边界 |
|---|---|---|
| [ProGrad，ICCV 2023](https://openaccess.thecvf.com/content/ICCV2023/html/Zhu_Prompt-aligned_Gradient_for_Prompt_Tuning_ICCV_2023_paper.html)；[论文](https://arxiv.org/abs/2205.14865) | 以预定义 prompt 的通用知识约束 prompt 更新，仅保留对齐或不冲突的梯度方向 | “减轻梯度冲突、保留通用知识”已有明确方法；当前代码只做 logit KD，尚未测梯度，不能直接归因为该机制 |
| [PromptSRC，ICCV 2023](https://openaccess.thecvf.com/content/ICCV2023/html/Khattak_Self-regulating_Prompts_Foundational_Model_Adaptation_without_Forgetting_ICCV_2023_paper.html)；[论文](https://arxiv.org/abs/2307.06948) | 冻结模型一致性、训练轨迹的自集成、文本多样性正则 | “冻结 CLIP 约束可学习 prompt”本身不是新点；需要与其具体目标和训练方式区分 |
| [AAPE，NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/file/93e67adb632d636bb2bcedf4eef1a6fe-Paper-Conference.pdf)；[论文](https://arxiv.org/abs/2410.23698) | 图像对齐的描述聚合，再通过表示蒸馏与任务损失训练 prompt 生成器 | 描述聚合加蒸馏已有直接先例；本项目静态 prompt、logit KD、双分支推理有所不同，差别是否有用需对照 |
| [ADK，2025 年预印本，方法 §3.4](https://arxiv.org/html/2512.17313v1#S3.SS4) | 结合描述均值和图像条件描述信息，以概率融合利用分支知识 | 无参数描述选择/概率融合是强简单对照。本仓库已跑的是适配 CoOp 的公式对照，不是作者实现或原表格复现 |
| [BiomedCoOp，CVPR 2025，方法 §3.3–3.4](https://arxiv.org/html/2411.15232v2)；[作者代码](https://github.com/HealthX-Lab/BiomedCoOp) | 用描述相似度和稳健离群统计筛选 prompt，结合语义一致性与选择性知识蒸馏 | “过滤不可靠描述后再蒸馏”已有直接先例。其任务是医学 BiomedCLIP，不能把其数字当作本项目自然图像 CLIP 的同协议结果 |
| [CA-MKD，ICASSP 2022](https://arxiv.org/abs/2201.00007)；[原文](https://arxiv.org/html/2201.00007v3) | 以真值交叉熵衡量多教师可靠性并加权蒸馏，也考虑中间特征 | 按可靠性选择/加权教师不是新概念；训练时有标签的可靠性不等于部署时可用的 softmax 置信度。它不是直接匹配的 CLIP few-shot 基线 |
| [TheProfessor，2026-06-22 预印本](https://arxiv.org/abs/2606.23897)；[原文](https://arxiv.org/html/2606.23897v1) | 在 PromptKD 设置中结合两种大教师，比较按最大 softmax 置信度融合与等权概率融合 | 普通置信度融合与教师互补已有更近的研究。该稿使用另一种大教师蒸馏设置，报告范围包含单种子实验；不能视为已录用结论或本项目公平基线 |

BiomedCoOp 的会议站点访问受限时，使用了论文完整 arXiv 版本和作者仓库核对；没有将访问失败当作“没有相关工作”。AAPE/ADK 先前的实现可用性与公式适配边界见[上一轮来源审计](../../preparation/attribution-20260911/closest-methods.md)，本次重读方法不代表新完成作者代码复现。

## 当前可作出的判断

1. 不把“多描述 + 注意力 + KD”、普通置信度门控、筛描述后蒸馏直接写成独有创新。它们可以是实现组成，但需要清楚的新问题、具体差异及有效证据。
2. [交叉融合方案](MINIMAL_NEXT_STEP.md)有诊断价值：把学生变化和推理教师变化分开，避免把所有变化归因于层级结构。它本身是一种受控检查，不是新算法。
3. S1+M 的精简候选即使提高准确率，也首先只证明一种训练/推理组合在所测协议下有效。要形成论文贡献，还需可解释的差异、强简单对照、独立确认与必要的成本报告。
4. 当前证据尚未支持增加新损失或复杂门控。先用旧模型检查可检验的假设；若收益或差异不足，停止这一轮扩量，而不是依靠继续叠模块寻找漂亮数字。

这些是基于上述来源和本项目实测结果的研究判断，不是期刊接收预测。投稿定位仍需等可复现的最终方法与完整证据形成后再定。
