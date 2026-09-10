# TextConsistency：工作计划与当前进度

最后更新：2026-09-10（GPU 对照运行中；北京时间）。本文件是后续接手和恢复工作的第一入口。每完成一批工作，先更新此处的状态、证据和下一步，再提交 GitHub。**准备完成、训练完成、贡献被验证是三个不同状态。**

**GPU 对照实验已启动。** 用户已开启 RTX 5090 32GB / 120GB 内存实例；完整模型数值复验及 GPU 短训练通过。官方 CoOp matched20 三种子已完成，与旧基线逐种子的 Base/Novel 准确率完全一致（HM 74.42）；KD=0/1 两组受控实验正在并行运行。详见 [本批运行记录](reports/experiments/ucf101-kd-study-20260910/README.md)。UCF101、DTD、FGVC Aircraft 已部署并通过预检，本地和服务器各 36 项测试通过。详见 [无卡准备完成报告](reports/preparation/无卡准备完成.md) 和 [机器可读检查记录](reports/preparation/cpu-prep-20260910/ready.json)。

## 目标与当前授权

- 目标：完成可复现、有实际贡献的期刊投稿，争取在 2027 年春节前获得录用；尚未确认目标期刊和学校认可条件，未承诺录用。
- 当前优先代码和实验，暂不修改论文/Overleaf。
- 无卡准备已完成。用户于 2026-09-10 开卡并提供新 SSH 入口，授权继续执行实验；本批运行官方 CoOp matched20 与 UCF101 KD 开/关三种子对照。
- 用户授权将相关代码、报告和进度推送到本仓库。SSH 密码、私钥、访问令牌不得写入仓库。

## 已完成的可信实验

2026-09-10 已完成 UCF101、16-shot、ViT-B/16、20 epochs、seed 1/2/3，6 个变体共 18 次训练、96 次独立评估。使用相同第 20 epoch checkpoint，Base/Novel 同一分支规则；baseline 为 student，其余主结果为 alpha=0.5 融合。

| 方法 | Base 均值 | Novel 均值 | HM（均值计算） |
|---|---:|---:|---:|
| 本仓库 CoOp 风格基线 | 83.56 | 67.08 | 74.42 |
| 描述均值 | 77.65 | 77.65 | 77.65 |
| 平面聚合 | 80.18 | 78.73 | 79.45 |
| 平面聚合 + MLP | 81.06 | 77.39 | 79.18 |
| 层级聚合 | 82.76 | 77.77 | 80.19 |
| 层级聚合 + MLP | 82.23 | 79.14 | 80.66 |

完整方法相对 baseline / mean 三个种子均提升；层级对平面的平均配对 HM 增益约 +0.74；MLP 在层级版本中仅 2/3 种子提升。当前不能声称 KD 有独立贡献、所有模块稳定有效、达到 SOTA 或达到录用条件。

- [完整三种子报告](reports/experiments/TextConsistency-UCF101三种子实验报告.md)
- [机器可读汇总与逐种子审计](reports/experiments/ucf101-s123-20260910/)
- 已验证训练源码 SHA256：`d74a9770d0d814c56269486e2492327881633d98314281d0f13a584ee279f17f`。
- 对应旧 Git 快照：`17debe268ecb86614fa9900305d8f44e1e160a4e`。
- 本机完整备份：工作区 `outputs/textconsistency-ucf101-s123-20260910-163256/`，含 18 个模型；压缩包 SHA256 为 `191a0d426e0783738d1f532ebf3c10b0e042616c1044674effd7bf7744b50106`。Git 只保存报告和审计，不保存权重。

## 当前工作进度

| 工作 | 状态 | 证据/完成条件 |
|---|---|---|
| 修复训练/保存/评估与记录 | 已完成 | 原 23 项测试 + 三种子审计 |
| UCF101 三种子实验 | 已完成 | 上述报告；禁止当成所有数据集结论 |
| 服务器连接与环境检查 | 已完成 | RTX 5090 32GB / 120GiB 内存，torch 2.8.0+cu128，CUDA 前后向通过 |
| DTD / Aircraft ZIP 本机校验 | 已完成 | reports/preparation/dataset-archives.json |
| UCF101 / DTD / Aircraft 服务器部署与预检 | 已完成 | ZIP 校验、固定划分、图像/描述覆盖、每类 16-shot、三种子采样均通过；见 ready.json |
| 官方 CoOp 差异核查与独立诊断 | 本机、开卡后服务器均通过 | 无卡时期 SIGKILL 保留；开卡后完整模型诊断 logits/梯度/更新差均为0，见本批 start-evidence |
| KD 开/关受控入口与多数据集入口 | 已实现并通过审查 | 本地/服务器各 36 项测试；匹配 teacher CE/融合，验证实际配置、变体身份、分支 checkpoint 一致性和每类 16-shot |
| 无卡准备验证和启动计划 | 已完成 | 36 项测试、3 份预检、6 份命令计划、CLI 和无 GPU 拒绝训练检查；11 个脚本/配置 SHA256 与本地一致 |
| KD 对照正式 GPU 训练 | KD0 / KD1 两组并行运行中 | UCF101，hierarchical/hierarchical_adapter，KD 0/1，seed 1/2/3 |
| 独立官方 CoOp GPU 训练 | matched20 三种子已完成 | Base/Novel 各种子准确率与历史基线完全一致；HM 74.42。200轮参考配置未启动 |
| 层级分组/参数量对照 | 待设计和执行 | 顺序/随机分组，平面参数量控制；当前代码未实现语义分组 |
| DTD / Aircraft 三种子正式训练 | 待执行 | 固定协议、相同描述来源、保留所有 seed |
| AAPE / ADK 接近方法比较 | 待适配和执行 | 先核对官方实现与协议，不能用跨协议分数直接比较 |
| 投稿实验定稿及论文修改 | 待上述证据 | 用户当前要求暂不处理论文 |

## 实验顺序和判断规则

恢复本批工作的第一步：先读取下述 status.json，确认进程和已完成输出；不要重复启动同一批次。已通过开卡检查，服务器仍有 `/root/autodl-tmp/textconsistency`，执行 `source /root/autodl-tmp/textconsistency/activate-current.sh`，检查 CUDA、显存、容器内存和数据路径。GPU 前后向、真实短训练和官方等价诊断已通过，正在执行下面的受控实验。若平台切换实例没有保留数据盘，应恢复部署，不能直接假设准备产物存在。具体命令在 [运行说明](docs/CPU_PREPARATION.md)。

1. 核验基线：本仓库基线与官方 CoOp 的模型计算是否一致；配置是否一致。这与“官方论文指标已复现”不同。
2. UCF101 KD=0/1 成对实验：同一个模型结构、teacher CE、seed/采样、训练轮数、融合权重，只改变 KD。分别报告 student、teacher、ensemble。
3. 层级结构核验：检查描述顺序、随机分组与参数量带来的影响；不将顺序分组称为语义分组。
4. DTD、Aircraft 三种子，检验跨任务稳定性；与适用的接近方法做公平比较。
5. 如层级收益稳定且相对接近方法有性能/效率价值，固定框架补齐投稿实验；若只赢自建基线或模块增益不稳定，针对证据精简或改进，不盲目叠模块。

本轮三个数据集用于研究决策，并不是任何期刊通用的“足够发表”门槛。不根据 Novel 测试结果选择模型、超参数或删掉不好的 seed。训练轮数更长不保证 Novel/HM 更高，基线需按明确协议判断。

## 接近工作及创新边界

- [AAPE，NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/file/93e67adb632d636bb2bcedf4eef1a6fe-Paper-Conference.pdf)：已有图像对齐的描述聚合及表示蒸馏；与当前 logit KD / 双分支推理不同，区别的效果需验证。
- [ADK，2025 预印本](https://arxiv.org/pdf/2512.17313)：已有描述均值与图像引导的无参数描述选择。
- 不能把“多描述 + 注意力 + 蒸馏”整体作为未经对照证实的独有创新；当前最需要验证两级聚合的独立价值。

## 恢复工作时必读

1. 读本文件，再读最近 preparation / experiments 报告和对应 JSON；以实际日志和产物为准。
2. 查看 `git status` / commit，检查服务器是否为预期实例，确认 CUDA 和磁盘。端口可能随实例/模式变化，向用户最新提供的入口连接。
3. 本轮服务器的根目录为 `/root/autodl-tmp/textconsistency`。旧 `src/`、`data/ucf101` 保留；曾跑 GPU 的实例和本次无卡实例不是同一份磁盘，不能假设旧 `runs/` 一定存在。
4. 新代码必须部署到独立 release 目录，运行使用新输出目录；不覆盖旧结果或伪装复用了已跑数据。
5. GitHub 保存源码、配置、计划、汇总、审计；数据集和 CLIP/训练权重另存。ImageNet-1K 尚未准备，不属于当前无卡阶段。
6. 每次完成后更新本文件：实际完成项、结果/失败、证据路径、下一条命令。训练中如需跨会话跟进，应先记录运行目录与状态，不靠聊天记忆。

## 当前后台运行位置

- GPU：RTX 5090 32GB；容器内存上限 120GiB；torch 2.8.0+cu128。
- 本批目录：`/root/autodl-tmp/textconsistency/runs/ucf101-kd-study-20260910-38402`。
- 队列 PID：`2741`（首次进程 2155 因旧 checkpoint 加载兼容性停止，现已保留模型恢复），状态入口为上述目录下 `status.json`；PID 只用于本实例本次进程，重启后应重新核对。
- 子目录：`official_matched20/`、`kd0/`、`kd1/`；官方阶段已通过，KD0 PID 4062、KD1 PID 4063 正在并行运行。
- 根日志：`/root/autodl-tmp/textconsistency/setup-logs/gpu-start-20260910/queue-resume.log`；阶段日志在本批目录 `official.log`、`kd0.log`、`kd1.log`。
- 正式训练代码 commit：`03c04e8694398800a5a8c58486d017d4132df8b0`；执行包装器快照保存在本批报告 start-evidence/execution-v2，首次版本和失败证据亦保留。
- 如失败，先读取状态与日志；保留失败目录和已完成模型，禁止覆盖结果或将部分种子报告成三种子均值。

- 本批已解决的执行问题：官方 checkpoint 含旧学习率调度器对象，PyTorch 默认安全加载不兼容。仅对本轮可信自产的官方 checkpoint 使用旧加载模式，模型 SHA256 不变；没有改动训练模型、超参数或数据。首次已完成的 seed1 第20轮模型保留，直接续评估。
