# UCF101 基线核验与 KD 对照：2026-09-10

本批于 2026-09-10 18:02（北京时间）启动，官方 matched20 三种子已完成，KD 两组运行中；蒸馏贡献尚待本批完成。用户已开卡并授权执行；后续状态以 [PROJECT_STATUS.md](../../../PROJECT_STATUS.md) 和服务器实时 `status.json` 为准。

## 已通过的启动检查

- RTX 5090 32GB，容器内存上限从无卡时的 2GiB 恢复到 120GiB；原数据盘、三个数据集及已验证 release 保留。
- [GPU 运算检查](start-evidence/runtime.json)：torch 2.8.0+cu128，CUDA 前向/反向通过。
- [完整模型数值复验](start-evidence/coop-equivalence/result.json)：在内存充足的服务器上运行相同 CPU 诊断，官方/当前实现 logits、loss、prompt 初值、梯度、一次 SGD 更新差均为0。不是准确率实验。
- [一轮 GPU 短训练](start-evidence/results.json)：baseline 和 hierarchical_adapter 共两次训练、八次重载评估通过，汇总器验证四个变体/分支行。仅检验真实训练/保存/重载，不纳入正式结果表。

## 正式实验安排

1. 独立、固定上游版本的官方 CoOp，`matched20`，三种子，20轮、16-shot、batch16、FP32、4随机上下文 token；分别重载第20轮 checkpoint 评估 Base/Novel。
2. 当前层级 / 层级+MLP 两个版本，各跑 KD0 与 KD1、seed1/2/3、20轮。只改变 KD 权重，保留 teacher CE=1、融合 alpha=.5 和模型结构。每个模型独立评估 student、teacher、ensemble。

共15次正式训练、78次独立评估。官方阶段通过后，KD0/KD1 两组作为独立进程并行；数据、批量和轮数均不变，耗时受共享 GPU 影响，不作为方法速度比较。200轮官方参考、DTD/Aircraft 和分组/参数量对照不在本批队列。

## 公平性与证据

官方源码保持固定 commit `ff61507c790454bce7c5052c3ac39e60772f1f89` 未修改；使用本项目 vendored Dassl 和 FP32，属于明确记录的环境适配。官方入口保留 cudnn benchmark=True，当前 TC 使用确定性算法；不同 backbone 构建过程消耗随机数不同，同一 seed 不代表两条路径的初始 prompt 逐位相同。本批匹配主要训练超参数并报告多种子结果，不宣称完整训练轨迹一致。

官方执行器在每次调用前保存有效配置、训练/测试样本哈希及数量，核对训练采样与已验证 TC 预检相同；训练后核对指定20轮 checkpoint，并在每次评估前后检查 SHA256 不变，要求日志明确加载该文件，再从正确数/总数计算准确率。保存上游、依赖和 CLIP 来源。

KD 结果使用现有汇总器检查模型/配置、seed、源码、样本和所有分支的 checkpoint 身份。本批完成后还需成对比较 KD0/1 的配置和采样，保留全部种子及分支，禁止按 Novel 测试结果挑 epoch 或删掉不好的种子。

## 恢复运行

- 根目录：`/root/autodl-tmp/textconsistency/runs/ucf101-kd-study-20260910-38402`。
- 队列状态：`status.json`；官方细分状态：`official_matched20/status.json`。
- 日志：根目录 `official.log`、`kd0.log`、`kd1.log`；官方逐条日志在 `official_matched20/evidence/`，TC 逐模型日志在各 `seedN/` 中。
- 启动信息：[launch.json](start-evidence/launch-resumed.json)。通过独立进程会话启动，SSH 断开不影响队列；不会自行开卡或关机。
- 执行包装器快照：[官方执行器](start-evidence/execution-v2/official_runner.py)、[队列](start-evidence/execution-v2/study_queue.py)。它们不改变模型实现。
- 正式代码 commit：`03c04e8694398800a5a8c58486d017d4132df8b0`。

已有输出目录不能直接重复执行；失败时先检查日志，保留失败证据后再制定续跑方式。正式结果完成后更新本文件和 PROJECT_STATUS.md，并备份权重及运行证据。

## 已记录的兼容性修复

官方 seed1 的20轮训练已正常结束，首次包装器在 checkpoint 审计时因 PyTorch 的 weights_only 默认行为与旧调度器对象序列化不兼容而停止。最小复现确认调度器类型为 CosineAnnealingLR；仅在官方子进程对本轮自产的可信模型使用 `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1`，直接审计时显式 `weights_only=False`。当前 TC 训练进程没有加入此环境覆盖，模型和官方源码均未修改。

[复现与加载验证](start-evidence/checkpoint-compatibility.json)证明 checkpoint 未改变；[恢复记录](start-evidence/launch-resumed.json)记录相同 SHA。保留首次失败状态、执行器和日志，seed1 不重新训练，恢复后从其 Base/Novel 评估继续。

## 已完成：官方 matched20 三种子

[完整结果](official-summary.json)：Base均值83.5574、Novel均值67.0813、HM74.4183。三个种子各自的 Base/Novel 准确率与历史本仓库基线完全一致，证明当前20轮设置的基线结果可以由独立官方实现复现。该结论只适用于本协议；不能替代200轮参考配置或说明已复现所有官方论文指标。
