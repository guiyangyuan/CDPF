# 已有证据、可复用资产与缺口

本页只整理已存在的结果，不新增准确率。可核对文件及SHA见[input-inventory.json](input-inventory.json)。历史结果是探索证据；本轮零新模型、零新推理。

| 要回答的问题 | 已有结果 | 仍不能据此声称 | 原始报告 |
|---|---|---|---|
| 描述本身有没有收益？ | 简单C+M相对C的平均配对ΔHM：UCF +4.35、DTD +6.95、Aircraft +4.06，各3/3种子正向 | 简单融合是新算法，或对所有数据集稳定有效 | [贡献验证](../../experiments/attribution-study-20260911-42527/README.md) |
| 层级是否有独立优势？ | 参数匹配后UCF平均配对ΔHM约+0.26，DTD−0.29、Aircraft−0.42；分组变化有敏感性 | 顺序分组是语义分组，或所有层级模型无效 | [结构研究](../../experiments/ucf101-structure-study-20260910/README.md)、[跨域研究](../../experiments/crossdataset-study-20260911-55424/README.md) |
| KD是否帮最终融合？ | UCF平均配对ΔHM +1.14；DTD−0.49，Aircraft−1.16；教师权重/输出一致性已核对 | 教师被KD训练坏，或全部KD方法无效 | [UCF KD](../../experiments/ucf101-kd-study-20260910/README.md)、[跨域贡献](../../experiments/attribution-study-20260911-42527/README.md) |
| 去掉目标置信度匹配能否修好？ | H1的seed1 Base验证三个固定数值条件均未通过；anchor最大logit差0 | 已验证新的Novel/HM，或H1有可投稿的新算法贡献 | [H1完成报告](../../experiments/h1-validation-20260913-42527/README.md) |
| 复杂模型究竟更贵多少？ | 已有参数量与各批次运行日志 | 同一硬件/计时口径下的延迟、能耗或成本优势；尚无公平成本基准 | [贡献验证中的边界](../../experiments/attribution-study-20260911-42527/README.md) |

可复用：三域已核验数据/模型备份、官方matched20 CoOp一致性检查、结构与KD控制、严格checkpoint身份和输入SHA、原始预测/logits与审计工具。原代码已有文本去重和token截断后去重，因此“新增完全相同描述去重”连本仓库的新功能都不算，不作为第三个候选。

Pets/EuroSAT的loader、dataset YAML、模板JSON存在；原始描述分别37类×20条、10类×25条。它们的像素数据、固定划分、可用路径和历史指标接触情况未在本轮确认。对Downloads中相关zip的窄搜索无结果，只表示该搜索范围没有匹配，不能推断用户没有数据或需要重下全部数据。

新研究还缺：学校对研究形式的要求、目标期刊与导师对范围的认可、新确认域完整性、外部作者实现的公平适配、第二描述来源、真实成本数据。现有三个域/单骨干/单训练预算的结果应在论文范围中明确，不能用一个“强简单基线”把这些缺口抹掉。

本轮没有调阅或改写Overleaf/PDF，没有联系导师、期刊或论文作者，没有购买服务/开卡；GitHub只保存选题资料和进度。
