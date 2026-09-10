# TextConsistency：工作计划与当前进度

最后更新：2026-09-10（结构对照GPU队列已启动；北京时间）。本文件是后续接手和恢复工作的第一入口。每完成一批工作，先更新此处的状态、证据和下一步，再提交 GitHub。**准备完成、训练完成、贡献被验证是三个不同状态。**

**当前运行：结构对照 GPU 队列已启动。** 用户提供新入口 `connect.bjb2.seetacloud.com:27946`；实测RTX5090 32GB、90GiB内存、驱动580.76.05，准备的源码/数据/权重仍在，源码SHA256一致，CUDA前后向通过。当前运行目录为 `/root/autodl-tmp/textconsistency/runs/structure-study-20260910-27946-v2`，队列PID1553、单轮验证runner PID1574。5次单轮训练/30次独立重载及审计已通过，现在执行15次正式训练/90次独立重载。原结构seed1的三个分支Base/Novel准确率已逐项复现历史结果，差值均为0。正式结果尚未齐全，不能据此判断结构效果。详见 [本轮运行记录](reports/experiments/ucf101-structure-study-20260910/README.md)。首次启动器命名冲突在导入PyTorch前终止，未产生训练结果，失败证据保留，已在独立目录修复。

**本批 GPU 基线与 KD 对照已完成并备份。** 共15次正式训练、78次独立重载评估，完整审计通过。官方 CoOp matched20 与旧基线的三个种子准确率完全一致；本轮KD1也逐种子复现历史结果。完整方法KD0→KD1的融合HM为79.50→80.66，3/3种子提高；普通层级为79.91→80.19，仅2/3种子提高。详见 [完整报告](reports/experiments/ucf101-kd-study-20260910/README.md)。三个数据集仍已准备好，DTD/Aircraft正式训练尚未执行。

## 目标与当前授权

- 目标：完成可复现、有实际贡献的期刊投稿，争取在 2027 年春节前获得录用；尚未确认目标期刊和学校认可条件，未承诺录用。
- 当前优先代码和实验，暂不修改论文/Overleaf。
- 用户已开卡并提供新入口，授权立即执行结构实验。CPU准备完成，后台队列运行中；GPU开关仍由用户操作。
- 无卡准备已完成。用户于 2026-09-10 开卡并提供新 SSH 入口，授权继续执行实验；本批已完成官方 CoOp matched20 与 UCF101 KD 开/关三种子对照；完整权重已备份。
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

完整方法相对 baseline / mean 三个种子均提升；层级对平面的平均配对 HM 增益约 +0.74；MLP 在层级版本中仅 2/3 种子提升。这批历史比较当时未单独验证KD；后续受控KD实验已完成，见本轮报告。仍不能声称所有模块稳定有效、达到SOTA或达到录用条件。

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
| 服务器连接与环境检查 | 新GPU入口已核实 | bjb2:27946，5090 32GB/90GiB内存，驱动580.76.05，torch2.8.0+cu128，CUDA前后向通过 |
| DTD / Aircraft ZIP 本机校验 | 已完成 | reports/preparation/dataset-archives.json |
| UCF101 / DTD / Aircraft 服务器部署与预检 | 已完成 | ZIP 校验、固定划分、图像/描述覆盖、每类 16-shot、三种子采样均通过；见 ready.json |
| 官方 CoOp 差异核查与独立诊断 | 本机、开卡后服务器均通过 | 无卡时期 SIGKILL 保留；开卡后完整模型诊断 logits/梯度/更新差均为0，见本批 start-evidence |
| KD 开/关受控入口与多数据集入口 | 已实现并通过审查 | 本地/服务器各 36 项测试；匹配 teacher CE/融合，验证实际配置、变体身份、分支 checkpoint 一致性和每类 16-shot |
| 无卡准备验证和启动计划 | 已完成 | 36 项测试、3 份预检、6 份命令计划、CLI 和无 GPU 拒绝训练检查；11 个脚本/配置 SHA256 与本地一致 |
| KD 对照正式 GPU 训练 | 已完成并审计 | 12次训练、72次独立评估；完整方法平均配对ΔHM +1.14，3/3正向；普通层级+0.28，2/3正向 |
| 独立官方 CoOp GPU 训练 | matched20 三种子已完成 | Base/Novel 各种子准确率与历史基线完全一致；HM 74.42。200轮参考配置未启动 |
| 层级分组/参数量对照 | 正式训练运行中，单轮验证全部通过 | 双头平面与层级均790018训练参数；原顺序/交错/固定随机17、29；本地/服务器各46测试通过，三数据集预检通过；新队列1553，正式结果待完成；不是语义分组 |
| DTD / Aircraft 三种子正式训练 | 待执行 | 固定协议、相同描述来源、保留所有 seed |
| AAPE / ADK 接近方法比较 | 待适配和执行 | 先核对官方实现与协议，不能用跨协议分数直接比较 |
| 投稿实验定稿及论文修改 | 待上述证据 | 用户当前要求暂不处理论文 |

## 实验顺序和判断规则

上一批 `status.json` 已为 completed，模型和评估均齐全；恢复时先阅读其报告和备份收据，避免重复启动已结束的队列。结构无卡准备已完成，现已启动新GPU队列，先在新release做五变体单轮GPU冒烟，审计后自动执行固定正式批次；不要重复启动。随后扩到DTD/Aircraft。新实验使用新目录，先检查CUDA、磁盘、代码版本和数据；结构命令见 [新协议](docs/STRUCTURE_CONTROLS.md)。

1. 已完成matched20基线核验：本仓库基线与官方 CoOp 的模型计算是否一致；配置是否一致。这与“官方论文指标已复现”不同。
2. 已完成UCF101 KD=0/1 成对实验：同一个模型结构、teacher CE、seed/采样、训练轮数、融合权重，只改变 KD。分别报告 student、teacher、ensemble。
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

## 结构对照准备入口（准备阶段历史，当前队列见顶部）

- 代码commit：`13b34978eeb8b23532b53abd1066af4719a0f909`；训练源码SHA256：`9bd4a7f248f09c47be17c1bb64921bd9a29d9f3b46e03e8caac9b934b2a95bba`。后续报告提交不改变这份训练代码。
- 新release：`/root/autodl-tmp/textconsistency/releases/structure-20260910-v1`；激活：`source /root/autodl-tmp/textconsistency/activate-structure-v1.sh`。旧 `activate-current.sh` 和旧release未修改。
- 服务器证据：`/root/autodl-tmp/textconsistency/setup-logs/structure-prep-20260910/ready.json`，state为`cpu_preparation_complete`；新GPU冒烟与正式训练标记均为false。
- 单轮冒烟计划：同目录 `smoke-plan.json`，5次训练+30次重载；正式计划 `formal-plan.json`，15次训练+90次重载。计划文件不是执行结果。
- 预留新输出：`runs/structure-smoke-v1`、`runs/ucf101-structure-v1`；准备时均未创建。开跑前再次核实不存在、没有重复队列，并记录实际运行目录和PID。
- 本机证据和代码压缩包：工作区 `work/structure-prep-20260910/`；GitHub收据/日志/计划位于 [准备报告](reports/preparation/structure-20260910/README.md)。三个数据集预检与上一轮完全一致；9个关键源码/脚本/文档文件与本机SHA256逐项一致。
- 以上为已完成的准备记录。现在使用顶部新GPU运行目录，先读取其status.json和日志，禁止重复执行准备计划。队列完成后审计全部分数和checkpoint，再备份到本机、更新报告。

## 最近已完成批次与备份

- 本批目录：`/root/autodl-tmp/textconsistency/runs/ucf101-kd-study-20260910-38402`，`status.json` 为 completed；15个最终模型和78次独立评估均已审计。
- 本批队列PID2741、KD进程4062/4063均为历史记录，不能据此启动或停止新进程。首次PID2155因旧checkpoint加载兼容性停止，已保留同一seed1模型恢复，首次失败记录未删除。
- 正式模型训练源码commit：`03c04e8694398800a5a8c58486d017d4132df8b0`；训练源码SHA256仍为`d74a9770d0d814c56269486e2492327881633d98314281d0f13a584ee279f17f`。执行器和审计脚本快照在本批报告start-evidence中。
- 完整备份位于本机工作区 `outputs/textconsistency-kd-study-20260910-38402/`，含压缩包和解压副本。15个checkpoint和压缩包均校验通过；压缩包SHA256：`66b962ba121eca46ff19a77bd996c325886f61bb95c56f020712738a7847ad19`。
- 同一压缩包还位于服务器 `/root/autodl-tmp/textconsistency/backups/ucf101-kd-study-20260910-38402.tar.gz`。GitHub保存报告、配置/采样/模型哈希证据与完整分析JSON，数据和权重另行保存。
- [完整本批报告](reports/experiments/ucf101-kd-study-20260910/README.md) / [审计JSON](reports/experiments/ucf101-kd-study-20260910/analysis.json) / [备份收据](reports/experiments/ucf101-kd-study-20260910/backup-receipt.json)。
