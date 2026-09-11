# 描述融合与KD贡献验证：无卡准备完成

2026-09-11。**当前只完成环境、输入、脚本及实验协议准备，尚未运行本批GPU训练或评估，没有新涨点结论。** 服务器实际为无卡模式，cgroup内存2GB；GPU开关由用户操作。

- [就绪收据](final/ready.json)：最终操作代码43e537c，真实准备通过；本机整合后的代码与服务器分别109项测试通过。无CUDA时启动器明确拒绝，正式study未创建。
- [固定实验计划](../../../docs/plans/2026-09-11-attribution.md)：11阶段、124条普通实验命令，阶段审计及最终审计另计。包括2次短训练、2次历史KD1重现训练、6次正式KD0训练、18次融合评估，所有种子/分支保留，不用Novel选参数或结果。
- [输入收据](input-receipt.json)：9个CoOp基线、6个DTD/Aircraft历史KD1、3个UCF完整方法参考；实际18个模型均与[本机完整备份](local-backup-correspondence.json)逐个SHA一致。
- [数据预检](input-preflight.json)和[历史样本核对](historical-sample-check.json)：UCF101/DTD/Aircraft三个seed的真实loader、类别、样本和描述覆盖已核对。记录原图路径/大小/mtime清单，不声称逐图内容SHA。
- 最终完整配置/依赖/历史文件/CLIP/模板/分词词表/有效描述顺序均冻结。原训练源码、模型和数据文件没有修改。
- [整体审查](review/final-review.md)及[最终复审](review/final-rereview.md)：所有问题已修复，没有剩余审查项。首次真实准备失败由Aircraft分词后重复描述引起，已按原encoder精确修正并保留[失败记录](attempt-7e02cb1/validation-7e02cb1.json)。中间版本通过的准备也保留在attempt-778d9d0。

最终服务器入口：`connect.bjb1.seetacloud.com:55424`。代码目录`/root/autodl-tmp/textconsistency/releases/attribution-ops-20260911-43e537c`；准备目录`/root/autodl-tmp/textconsistency/setup-logs/attribution-prep-20260911-v1/prepared-43e537c`。

开卡后使用[启动脚本](execution/run_gpu_study.sh)：
```bash
bash /root/autodl-tmp/textconsistency/setup-logs/attribution-prep-20260911-v1/run_gpu_study.sh
```
正式输出固定为`/root/autodl-tmp/textconsistency/runs/attribution-study-20260911-v1`。开跑前再次确认GPU空闲和输入哈希；发现已有目录、失败状态或正在运行的队列时，先查状态，禁止覆盖/重复启动。两次历史重现检查必须精确匹配正确数，才能继续正式实验；失败即停止调查。

完整准备证据以[preparation.json.gz](final/preparation.json.gz)和[protocol.json.gz](final/protocol.json.gz)保存在Git，分别附解压后及压缩包SHA收据；本机与服务器同时保留原始JSON。这里没有数据集或模型权重，模型完整备份位置见PROJECT_STATUS及历史实验报告。

本轮比较包含简单logit融合、概率均值融合以及ADK公式受控适配。作者发布证据、公式和边界见[接近方法核对](closest-methods.md)。ADK作者仓库目前未公开可用实现，本轮不能称官方代码复现或论文表格指标复现。公式/温度解释如有误，需修正并重评该对照；UCF旧基线仅接受已证明有效KD为0的一项历史声明差异，该兼容性如不成立需重跑匹配基线。三项决策及代价完整保存在[工作记录](review/progress.md)。
