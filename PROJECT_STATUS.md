# TextConsistency：工作计划与当前进度

最后更新：2026-09-11（DTD/Aircraft完成、审计通过、全部模型已备份；北京时间）。本文件是后续接手和恢复工作的第一入口。每完成一批工作，先更新此处的状态、证据和下一步，再提交 GitHub。**准备完成、训练完成、贡献被验证是三个不同状态。**

**新的当前工作：描述融合与跨数据集KD贡献验证准备中（2026-09-11）。** 用户在评估后授权开始。bjb1:55424已连接，实际为无卡模式、cgroup内存2GB；尚未启动本批GPU训练。计划见[固定贡献验证计划](docs/plans/2026-09-11-attribution.md)。18个参考模型已与本机完整备份逐个核对；三套真实loader及18组baseline元数据检查通过。融合评估器已审查通过；124条固定实验命令及独立审计已实现，本地104项测试通过，队列审查和服务器完整无卡准备正在进行。接近方法采用明确标注的ADK公开公式适配，作者仓库暂未提供实现。旧批次全部结果仍保留如下。

**上一批工作：DTD/Aircraft固定三种子验证已完成。** 正式18次训练/84次独立重载全部通过，另有6次单轮训练/28次重载单列保存；4个阶段和最终总审计均通过，141条命令无失败或重跑。模型代码和固定协议不变。完整压缩包及24个模型均已下载到本机并逐个校验。见[完整实验报告](reports/experiments/crossdataset-study-20260911-55424/README.md)及[备份收据](reports/experiments/crossdataset-study-20260911-55424/backup-receipt.json)。

**本批结论：整体方法提高HM，但层级独立优势未被验证。** DTD原方法/基线/参数匹配平面的HM为69.51/63.79/69.76；Aircraft为36.22/34.26/36.63。原方法相对基线在两个数据集各3/3种子的HM提高，伴随Base下降、Novel上升。层级−平面的平均配对HM分别为−0.29（1/3正向）和−0.42（0/3正向）。结合UCF101仅+0.26、2/3正向，目前不能把层级优于平面作为稳固贡献。下一步优先精简层级主张、补接近方法的公平比较；尚未启动新的实验，不改论文。

**上批 GPU 完成时状态（当前已切无卡）：bjb1:55424。** RTX5090 32GB、驱动595.71.05、torch2.8.0+cu128，队列墙钟约42.65分钟。PID1576为已完成队列的历史记录，恢复时先核对真实进程及status.json，不可重复启动。实验目录为`/root/autodl-tmp/textconsistency/runs/crossdataset-study-20260910-v1`。最终实测GPU无计算进程、利用率0%、显存2MiB；服务器电源由用户控制。

**上一批次：结构对照已完成、审计通过、完整模型已备份。** 共15次正式训练/90次独立重载，另外5次单轮验证/30次重载不计入正式结果。原层级融合HM **80.66**，参数匹配双头平面 **80.38**；原层级平均配对优势仅 **+0.26个百分点（SD1.18，2/3种子提升）**，层级结构的独立收益尚不稳定。交错、随机17、随机29的HM分别为79.86、81.20、80.49，全部保留，不能挑选最佳Novel成绩改称最终方法。原结构全部3seed/3分支的18个Base/Novel准确率精确复现历史结果。详见 [完整结构对照报告](reports/experiments/ucf101-structure-study-20260910/README.md)。本批没有仍在运行的训练；后续DTD/Aircraft验证也已完成，结论见顶部；不继续用Novel调参。

结构批次开卡时入口 `connect.bjb2.seetacloud.com:27946`；当时RTX5090 32GB、90GiB内存、驱动580.76.05。当前GPU批次见顶部。实验目录 `/root/autodl-tmp/textconsistency/runs/structure-study-20260910-27946-v2`；PID1553/1574/3154仅为历史记录。队列完成时GPU显存/利用率均为0。原始status.json的`awaiting_analysis_and_backup`是执行器结束时快照；后续分析与本机备份均已完成，以本批analysis.json和backup-receipt.json为证。服务器电源仍由用户控制。

**本批 GPU 基线与 KD 对照已完成并备份。** 共15次正式训练、78次独立重载评估，完整审计通过。官方 CoOp matched20 与旧基线的三个种子准确率完全一致；本轮KD1也逐种子复现历史结果。完整方法KD0→KD1的融合HM为79.50→80.66，3/3种子提高；普通层级为79.91→80.19，仅2/3种子提高。详见 [完整报告](reports/experiments/ucf101-kd-study-20260910/README.md)。这是历史批次；DTD/Aircraft正式验证现已完成，见顶部。

## 目标与当前授权

- 目标：完成可复现、有实际贡献的期刊投稿，争取在 2027 年春节前获得录用；尚未确认目标期刊和学校认可条件，未承诺录用。
- 当前优先代码和实验，暂不修改论文/Overleaf。
- 用户开卡并提供新入口，授权的结构对照批次现已完成并备份；GPU开关仍由用户操作。
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
| 服务器连接与环境检查 | 新GPU入口已核实 | bjb1:55424，5090 32GB，驱动595.71.05，CUDA前后向通过；准备目录保留 |
| DTD / Aircraft ZIP 本机校验 | 已完成 | reports/preparation/dataset-archives.json |
| UCF101 / DTD / Aircraft 服务器部署与预检 | 已完成 | ZIP 校验、固定划分、图像/描述覆盖、每类 16-shot、三种子采样均通过；见 ready.json |
| 官方 CoOp 差异核查与独立诊断 | 本机、开卡后服务器均通过 | 无卡时期 SIGKILL 保留；开卡后完整模型诊断 logits/梯度/更新差均为0，见本批 start-evidence |
| KD 开/关受控入口与多数据集入口 | 已实现并通过审查 | 本地/服务器各 36 项测试；匹配 teacher CE/融合，验证实际配置、变体身份、分支 checkpoint 一致性和每类 16-shot |
| 无卡准备验证和启动计划 | 已完成 | 36 项测试、3 份预检、6 份命令计划、CLI 和无 GPU 拒绝训练检查；11 个脚本/配置 SHA256 与本地一致 |
| KD 对照正式 GPU 训练 | 已完成并审计 | 12次训练、72次独立评估；完整方法平均配对ΔHM +1.14，3/3正向；普通层级+0.28，2/3正向 |
| 独立官方 CoOp GPU 训练 | matched20 三种子已完成 | Base/Novel 各种子准确率与历史基线完全一致；HM 74.42。200轮参考配置未启动 |
| 层级分组/参数量对照 | 已完成并审计、备份 | 15正式训练/90独立重载；五变体均790018参数；原层级对参数匹配平面平均配对HM仅+0.26，2/3正向；分组存在敏感性；不是语义分组 |
| DTD / Aircraft 固定验证准备 | 本机及服务器无卡准备已完成 | 服务器59项测试、实际数据loader校验、无CUDA拒绝训练通过；操作脚本6ce9c6e独立部署，原模型源码不变；见本轮ready.json |
| DTD / Aircraft 三种子正式训练 | 已完成、审计通过并备份 | 18正式训练/84重载；6短训练/28重载单列；两数据集完整方法均赢基线，层级均值不及参数匹配平面 |
| AAPE / ADK 接近方法比较 | 待适配和执行 | 先核对官方实现与协议，不能用跨协议分数直接比较 |
| 投稿实验定稿及论文修改 | 待上述证据 | 用户当前要求暂不处理论文 |

## 实验顺序和判断规则

KD批次及结构批次均已完成，模型和评估齐全，已备份到本机；恢复时先阅读报告和备份收据，避免重复启动。结构代码的固定协议见 [结构协议](docs/STRUCTURE_CONTROLS.md)。DTD/Aircraft的CoOp、原方法及参数匹配平面对照协议现已固定，保留三个seed和统一主分支；独立操作脚本已部署，CPU准备完成。开卡后启动器再次检查CUDA空闲、代码版本、数据及新输出目录。该固定队列现已全部完成并通过审计，24个模型已完整备份。下一步基于完整证据决定简化框架和接近方法比较，不重复此批实验。

1. 已完成matched20基线核验：本仓库基线与官方 CoOp 的模型计算是否一致；配置是否一致。这与“官方论文指标已复现”不同。
2. 已完成UCF101 KD=0/1 成对实验：同一个模型结构、teacher CE、seed/采样、训练轮数、融合权重，只改变 KD。分别报告 student、teacher、ensemble。
3. 已完成UCF101层级结构核验：参数匹配后的层级优势较小且非所有seed正向，分组敏感性存在；不能据此认定层级贡献稳定，不将顺序分组称为语义分组。
4. 已完成DTD、Aircraft三种子：整体方法相对基线的HM均提升；层级对参数匹配平面在两个数据集都没有平均优势。接近方法的公平比较仍待执行。
5. 如层级收益稳定且相对接近方法有性能/效率价值，固定框架补齐投稿实验；若只赢自建基线或模块增益不稳定，针对证据精简或改进，不盲目叠模块。

本轮三个数据集用于研究决策，并不是任何期刊通用的“足够发表”门槛。不根据 Novel 测试结果选择模型、超参数或删掉不好的 seed。训练轮数更长不保证 Novel/HM 更高，基线需按明确协议判断。

## 接近工作及创新边界

- [AAPE，NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/file/93e67adb632d636bb2bcedf4eef1a6fe-Paper-Conference.pdf)：已有图像对齐的描述聚合及表示蒸馏；与当前 logit KD / 双分支推理不同，区别的效果需验证。
- [ADK，2025 预印本](https://arxiv.org/pdf/2512.17313)：已有描述均值与图像引导的无参数描述选择。
- 不能把“多描述 + 注意力 + 蒸馏”整体作为未经对照证实的独有创新。UCF101及DTD/Aircraft参数匹配对照均已完成：跨数据集证据尚不支持两级聚合的稳定独立价值；下一步优先降低层级贡献主张和接近方法比较，不依据Novel挑分组。

## 恢复工作时必读

1. 读本文件，再读最近 preparation / experiments 报告和对应 JSON；以实际日志和产物为准。
2. 查看 `git status` / commit，检查服务器是否为预期实例，确认 CUDA 和磁盘。端口可能随实例/模式变化，向用户最新提供的入口连接。
3. 本轮服务器的根目录为 `/root/autodl-tmp/textconsistency`。旧 `src/`、`data/ucf101` 保留；曾跑 GPU 的实例和本次无卡实例不是同一份磁盘，不能假设旧 `runs/` 一定存在。
4. 新代码必须部署到独立 release 目录，运行使用新输出目录；不覆盖旧结果或伪装复用了已跑数据。
5. GitHub 保存源码、配置、计划、汇总、审计；数据集和 CLIP/训练权重另存。ImageNet-1K 尚未准备，不属于当前无卡阶段。
6. 每次完成后更新本文件：实际完成项、结果/失败、证据路径、下一条命令。训练中如需跨会话跟进，应先记录运行目录与状态，不靠聊天记忆。

## DTD/Aircraft完成记录与历史启动入口

- 操作代码commit：`6ce9c6e2770c6359dcd1f8dfea326e3755aaa79c`；release：`/root/autodl-tmp/textconsistency/releases/crossdataset-ops-20260911-6ce9c6e`。模型仍使用`releases/structure-20260910-v1`及原SHA，后续文档提交不改变已校验脚本。
- 准备证据：`/root/autodl-tmp/textconsistency/setup-logs/crossdataset-prep-20260911-v2/prepared-v2/`；同级`ready.json`为`cpu_preparation_complete`，59项测试通过，GPU短测试/正式训练均为false。
- 历史启动命令（本批已经完成，不可重复执行）：`bash /root/autodl-tmp/textconsistency/setup-logs/crossdataset-prep-20260911-v2/run_gpu_study.sh`。本批PID1576已完成，status.json为passed；不可再次执行，恢复时先读取状态和报告。
- 预留输出：`/root/autodl-tmp/textconsistency/runs/crossdataset-study-20260910-v1`，准备结束时不存在，本批已创建并完整备份。若恢复时目录已存在，先读`status.json`和日志；不得覆盖或直接重启。
- 固定顺序：DTD短测试→Aircraft短测试→DTD正式→Aircraft正式；本批四阶段及总审计全部通过，无失败重试。
- [准备报告](reports/preparation/crossdataset-20260910/README.md)保存真实日志、数据采样证据、完整命令计划、版本收据和启动脚本。图像清单记录路径/大小/mtime，未逐图计算内容SHA；完整ZIP、CLIP与代码文件已核对SHA。

## 最新双数据集批次与完整备份

- [完整报告](reports/experiments/crossdataset-study-20260911-55424/README.md)、[完整分析](reports/experiments/crossdataset-study-20260911-55424/analysis.json)、[执行状态](reports/experiments/crossdataset-study-20260911-55424/completed-status.json)、[备份收据](reports/experiments/crossdataset-study-20260911-55424/backup-receipt.json)。
- 本机工作区目录`outputs/textconsistency-crossdataset-study-20260911-55424/`，包含完整压缩包和解压副本；原始DTD子集提前备份也保留。
- 完整包`crossdataset-study-20260911-55424.tar.gz`，112,686,945字节，SHA256为`4a681854c299562d8d4dca771584abf09cf167daf919de89ff2f631187e9e34e`；包和全部24个checkpoint已在本机逐个校验。
- 服务器同一包：`/root/autodl-tmp/textconsistency/backups/crossdataset-study-20260911-55424.tar.gz`。GitHub保存配置/指标/哈希/审计和报告，权重本体及数据不入Git。

## 结构对照准备入口（准备阶段历史，当前队列见顶部）

- 代码commit：`13b34978eeb8b23532b53abd1066af4719a0f909`；训练源码SHA256：`9bd4a7f248f09c47be17c1bb64921bd9a29d9f3b46e03e8caac9b934b2a95bba`。后续报告提交不改变这份训练代码。
- 新release：`/root/autodl-tmp/textconsistency/releases/structure-20260910-v1`；激活：`source /root/autodl-tmp/textconsistency/activate-structure-v1.sh`。旧 `activate-current.sh` 和旧release未修改。
- 服务器证据：`/root/autodl-tmp/textconsistency/setup-logs/structure-prep-20260910/ready.json`，state为`cpu_preparation_complete`；新GPU冒烟与正式训练标记均为false。
- 单轮冒烟计划：同目录 `smoke-plan.json`，5次训练+30次重载；正式计划 `formal-plan.json`，15次训练+90次重载。计划文件不是执行结果。
- 预留新输出：`runs/structure-smoke-v1`、`runs/ucf101-structure-v1`；准备时均未创建。开跑前再次核实不存在、没有重复队列，并记录实际运行目录和PID。
- 本机证据和代码压缩包：工作区 `work/structure-prep-20260910/`；GitHub收据/日志/计划位于 [准备报告](reports/preparation/structure-20260910/README.md)。三个数据集预检与上一轮完全一致；9个关键源码/脚本/文档文件与本机SHA256逐项一致。
- 以上为已完成的准备记录。顶部GPU批次也已完成，禁止重复执行准备计划。全部分数和checkpoint已审计，完整备份在本机，见下方收据。

## 最新结构批次与备份

- 服务器实验目录：`/root/autodl-tmp/textconsistency/runs/structure-study-20260910-27946-v2`，15个正式模型和90次重载均已审计；5个短测试模型单独保存，不纳入正式结果。
- 本机完整备份目录：工作区 `outputs/textconsistency-structure-study-20260910-27946/`，含压缩包及解压副本。大小127,542,691字节，SHA256：`d83f5d813784ea989084ce746f5d630cbfafad368883d9dec36202a806a19be9`；本机压缩包和全部20个checkpoint逐个验证通过。
- 服务器同一压缩包：`/root/autodl-tmp/textconsistency/backups/structure-study-20260910-27946-v2.tar.gz`。不含数据集和冻结CLIP基础权重，GitHub仅保存轻量报告/审计/配置/指标。
- 源码commit：`13b34978eeb8b23532b53abd1066af4719a0f909`，训练源码SHA256：`9bd4a7f248f09c47be17c1bb64921bd9a29d9f3b46e03e8caac9b934b2a95bba`。首次启动器命名冲突在PyTorch导入时失败、没有训练，失败状态/脚本/日志已保留；修复只涉及操作脚本，不改变训练源码。
- [完整结构报告](reports/experiments/ucf101-structure-study-20260910/README.md) / [完整分析](reports/experiments/ucf101-structure-study-20260910/analysis.json) / [备份收据](reports/experiments/ucf101-structure-study-20260910/backup-receipt.json)。

## 上一轮KD批次与备份

- 本批目录：`/root/autodl-tmp/textconsistency/runs/ucf101-kd-study-20260910-38402`，`status.json` 为 completed；15个最终模型和78次独立评估均已审计。
- 本批队列PID2741、KD进程4062/4063均为历史记录，不能据此启动或停止新进程。首次PID2155因旧checkpoint加载兼容性停止，已保留同一seed1模型恢复，首次失败记录未删除。
- 正式模型训练源码commit：`03c04e8694398800a5a8c58486d017d4132df8b0`；训练源码SHA256仍为`d74a9770d0d814c56269486e2492327881633d98314281d0f13a584ee279f17f`。执行器和审计脚本快照在本批报告start-evidence中。
- 完整备份位于本机工作区 `outputs/textconsistency-kd-study-20260910-38402/`，含压缩包和解压副本。15个checkpoint和压缩包均校验通过；压缩包SHA256：`66b962ba121eca46ff19a77bd996c325886f61bb95c56f020712738a7847ad19`。
- 同一压缩包还位于服务器 `/root/autodl-tmp/textconsistency/backups/ucf101-kd-study-20260910-38402.tar.gz`。GitHub保存报告、配置/采样/模型哈希证据与完整分析JSON，数据和权重另行保存。
- [完整本批报告](reports/experiments/ucf101-kd-study-20260910/README.md) / [审计JSON](reports/experiments/ucf101-kd-study-20260910/analysis.json) / [备份收据](reports/experiments/ucf101-kd-study-20260910/backup-receipt.json)。
