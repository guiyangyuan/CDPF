# TextConsistency：工作计划与当前进度

最后更新：2026-09-14（文本侧残差三组GPU实验完成，42文件完整备份核验通过，AutoDL已关机，结果未达到继续门槛）。

**最新工作已完成并关机：文本侧残差三组先导。** 用户授权直接实施/训练/报告，代码d5152e1。三次20轮训练和三次独立重载均通过，用时15.63分钟；S32.0%、F30.5%、类别名文本残差32.0%、原始描述文本残差32.0%、近邻中心化描述残差32.5%。候选相对S和两种文本对照各+0.5pp（改对1、改错0），只比F高2pp，未通过四项均至少+1pp的条件；停止此配置扩量，不声称方法或创新已验证。三组学生提示及logits完全一致。3完整模型/42文件已下载逐项核验，AutoDL网页确认已关机；无待执行队列。短检查仅2项0.962秒，未跑全量测试。见[最终报告](reports/experiments/text-residual-20260914/README.md)及[事前方案](docs/TEXT_RESIDUAL_PILOT.md)。

**执行偏好更新：用户要求加快迭代、简化检查。** 后续模型改动默认仅做1–2个小批次及与改动直接相关的核心断言；保存/类别路径改动才加对应核对。不默认跑全套CPU测试、两端重复验证或多轮审查，尽快进入小规模真实实验，有信号后再扩大验证。配置、种子、版本、真实结果和备份继续保留。具体见[快速迭代约定](reports/diagnostics/visual-adapter-followup-20260914/README.md#快速迭代约定2026-09-14用户要求简化检查)。随后用户授权“开始下一步”，现已完成下述本机实现。

**上一轮GPU工作已完成并关机：描述纠错三组小实验。** 固定ff4aa5e，RTX4090，三次20轮训练/三次独立重载，14.69分钟；原学生32.0%、简单融合30.5%、类别名纠错32.0%、详细描述纠错32.5%。详细描述相对学生/类别名各+0.5pp（改对1、改错0），未通过三项均至少+1pp的门槛，停止扩量。最终学生提示和逐图学生logits与基线最大差均为0。三个完整模型、日志和原始分数已下载，42文件核验通过。用户已授权通过AutoDL操作现有实例开卡/备份后关机；本次网页确认“已关机”，没有后台任务。见[完整结果与关机记录](reports/experiments/description-correction-20260914/README.md)。

**上一轮本机工作已完成：保护学生训练的描述纠错原型。** 新入口 `CoOp_TextConsistency_Correction`：学生用原始冻结图像特征，适配器只提供R=Da−D0，最终Z=S+R；损失为CE(S)+CE(stopgrad(S)+R)。三份配置为学生/类别名纠错/描述纠错；同等20轮，后两者同参数。只运行两项短检查，最终0.934秒通过：两步学生更新与原基线完全一致、纠错器更新、冻结参数不变、完整重载、类别数切换与正确导出。初次测试辅助函数学习率不匹配已修正，没有放宽比较。专用启动/导出入口完成，GPU协议固定为Aircraft seed1三次训练/三次重载，描述Z须同时比学生S、简单融合F、类别名Z至少+1pp；没有新增真实训练或GPU推理。该准备已被上方GPU批次完整执行，服务器现已关机。 见[实现、检查与运行说明](docs/DESCRIPTION_CORRECTION.md)。

**当前：Aircraft A/B/C 先导实验已完成、复核通过、备份完成。** RTX4090固定1a38307，恰好三次20轮训练/三次导出，16.07分钟；F准确率A/B/C为30.50/30.50/29.00%，C−A -1.50pp、C−B -1.50pp，**未通过继续扩量的门槛**。这是seed1 Base验证开发实验，不是Novel或独立确认结果。三个完整模型和全部结果已下载逐项核对；GPU结束时空闲，可以关卡。见[本轮完整报告](reports/experiments/visual-adapter-westc-20260914/README.md)。

**原型阶段（已完成）：视觉适配＋描述监督实现与 CPU 验证。** 新的共享残差MLP直接参与CoOp和描述分类，已配置A简单融合/B普通视觉适配/C描述监督视觉适配；固定64维瓶颈、残差0.2、零升维初始化、20轮总预算，暂不加入KD。完整205项测试通过，12项新测试覆盖梯度、骨干冻结、完整保存/重载和不同Novel类别；8组旧行为的132张量与改动前精确一致。独立审查发现的H1直接导出混用风险已修复并复审通过。见[完成报告与验证记录](reports/preparation/visual-adapter-cpu-20260913/README.md)、[配置与使用](docs/VISUAL_ADAPTER_PROTOTYPE.md)。训练源码SHA为`38d9afc3cdc0440695cb5eca5c59069f73ace1aba4478d5afc5b656050911840`。这些是tiny CLIP/合成输入的软件验证，没有新数据集成绩、GPU运行或论文修改。

**研究目标已纠正：优化/结合模型取得准确率、泛化或特定场景优势，形成方法论文。** 用户已拒绝纯实验研究主线；旧8/24次矩阵取消，不等待学校对实验研究形式的答复。首个原型关注细粒度识别，保留Base-to-Novel信息边界；实现完成不等于涨点或创新已成立。已有描述/KD分析继续作为模型开发证据，不能恢复已拒绝的选题。

**历史 GPU 批次：H1 固定对照已结束，按事前规则停止扩量。** RTX5090 32GB，固定efcd3db；北京时间13:31:27–13:46:06，恰好4次20轮训练、4次导出，无重试。DTD full_kl anchor的S/T/M与历史logits最大差均为0。SN+M的seed1 Base验证为UCF83.82、DTD72.92、Aircraft29.00；相对C+M logits/概率及S0+M的等权平均分别−1.00/−1.00327/−1.84069个百分点，三个条件全未通过。study为stopped，前12阶段通过，第13汇总因数值门槛返回2；训练均成功。见[完整结果、独立审计与备份](reports/experiments/h1-validation-20260913-42527/README.md)。73个文件含4模型已下载逐项核对；已告知可以关卡，电源由用户操作。该历史批次无待运行队列，禁止重启PID1531对应的历史study。

**此前无卡准备：H1独立Base验证队列已部署并完成准备。** 代码efcd3db；本机、服务器各193项CPU测试及四份实际prepare通过。2,516个输入文件核验一致，24个服务器证据文件已下载核对；当时没有真实H1训练。该准备已被上方GPU批次完整执行，原启动命令仅保留为历史。见[无卡准备报告](reports/preparation/h1-server-20260913/README.md)。

**上一阶段本机工作：H1加权非目标项对照已实现并通过CPU核验。** 新增`KD_LOSS_MODE=weighted_nckd`，默认仍是`full_kl`；保留原非目标权重、在线教师CE、架构及推理。完整153项测试通过，包括真实tiny CLIP/SGD训练步、两模式三个分支重载、损失身份拒绝和旧格式审计兼容。三数据集六份配置除损失模式外相同；九组旧logits上的36组默认损失/师生梯度与旧公式精确一致，18组新项分解检查通过。修复了稀疏协议跳过身份校验、现代全KL记录温度不一致仍被接收、非有限参数未提前拒绝的问题。见[CPU准备报告和证据](reports/preparation/h1-cpu-20260913/README.md)。训练源码SHA为`387ea6e46aa0fa6cba5109e64baab94a548b35c200a1945267d392bc35a36677`。该CPU实现阶段没有登录服务器或启动GPU；tiny合成输入优化器步仅为软件测试，没有新方法准确率。

**前轮本机工作：KD学生退步归因与文献核对已完成。** 复用九组Base验证logits，1,500条记录中KD学生新增124条错误、纠正75条；92条新错对应最终教师也错，但35条纠正也发生在教师错误记录上。三数据集学生与教师KL均下降约72%–77%，学生平均准确率和NLL却变差。18组端点导数与冻结历史完整损失核验通过，原53个GPU证据文件和所有输入SHA不变。这里只验证输出logit空间的数学关系与端点关联，没有证明训练轨迹因果。见[完整报告、逐样本证据与归档](reports/diagnostics/kd-failure-20260912/README.md)。

**最新研究判断：仍未确认可投稿的新算法贡献，H1这次已知分解消融停止扩量。** H1实现、准备、实际训练与独立核验均完成；相对原full_kl融合在UCF/DTD分别+2.94/+2.08个百分点，Aircraft−1.00，但三数据集均低于无KD对照，不能视为稳定有效修补。已知分解不是原创算法，也不因局部涨点变成原创贡献。见[既有文献边界](reports/diagnostics/kd-failure-20260912/LITERATURE.md)、[H1固定方案及完成更新](reports/diagnostics/kd-failure-20260912/NEXT_STEP.md)。本轮没有Novel/HM或论文修改。

**历史阶段：本机诊断与最小方案已完成。** 当时用户已告知关卡，该诊断阶段没有连接服务器或新增训练。18个预测文件（26,424条记录、8,808张唯一图片）和38个输入文件的SHA核对通过；保留全部种子/类别统计。DTD/Aircraft六对KD0/KD1的787,970个教师参数逐张量完全相同，学生均改变；KD导致的融合差异不能解释为教师被KD重新训练坏了。已训练教师相对固定描述均值的Novel准确率差分别为−6.04、−1.60个百分点，但原因尚未证实。见[诊断与完整证据](reports/diagnostics/error-analysis-20260911/README.md)及[相关工作边界](reports/diagnostics/error-analysis-20260911/PRIOR_WORK.md)。

**准备阶段：验证集导出/分析工具、本机准备及审查已完成。** 新增独立只读入口，使用原模型类，强制Base验证集、完整参数加载和输入身份核对；CPU分析保留六原始分支及18固定融合。18项针对性测试和完整128项测试通过，修复后九组真实CPU预检通过：27模型、1,500条验证记录、1,280张唯一图片，输入快照不变。UCF补核后九对教师参数全部相同。任务复审4/4问题已解决；最终审查的3项完整性问题也已修复并通过复审。该本机准备阶段没有新增训练、GPU推理或方法成绩；随后真实CUDA检查已通过，见最新GPU工作。见[本机准备报告与证据](reports/preparation/validation-probe-20260911/README.md)。

**此前服务器准备：无卡准备与备份已完成（2026-09-12核实）。** 用户无卡开启bjb1:42527后，部署独立操作release `validation-ops-20260911-de57efd`。服务器128项测试及九组真实CPU prepare通过，27个旧模型和1,500条验证记录核对通过；1,280张验证图片逐张SHA与本机一致。44个证据文件已下载逐项校验，完整正负记录保留。两处测试解释器路径修正，Aircraft seed2两次SIGKILL后经缓存建议在新目录成功；未改变模型或固定协议。见[服务器准备报告与备份](reports/preparation/validation-server-20260911-42527/README.md)。

**此前GPU工作：旧模型验证集交叉融合已完成、审计通过并备份。** 固定工具de57efd，19条命令成功，零新增训练；9组Base验证导出包含1500条记录和全部24分支/融合。53个证据文件已下载核对，本机复算通过。见[历史完整报告](reports/experiments/validation-probe-20260912-42527/README.md)。study runs/validation-probe-20260911-v1 已完成，PID1502为历史PID，禁止重复运行。

**此前三种子结论：固定候选S1+M_logits未通过三个数值筛查条件，已停止扩量。** 三数据集三种子平均Base验证分别UCF81.86、DTD70.83、Aircraft33.00；简单C+M logits为85.95、76.74、35.50。候选等权平均相对C+M logits/概率及S0+M分别−4.16/−4.54/−4.44个百分点，正向案例1/0/0（各9例）。该历史失败候选不重新启动；本次H1属于随后单独执行的固定损失对照，结果见顶部。

**当前下一步：** 当前描述纠错候选停止扩量；先基于这一小幅改善及既有失败证据决定下一项模型假设，不自动追加训练或调参。所有结果和完整模型已备份，AutoDL网页确认实例已关机；PID1287及旧PID1269都是历史进程，不可重启。用户允许为已确认实验操作现有实例开卡、备份后关机；不据此新租实例或无限扩量。

**此前GPU工作：贡献验证已完成、审计通过、完整模型已备份（2026-09-11）。** 用户开卡入口bjb1:42527，实际RTX5090 32GB、90GiB内存；固定操作代码43e537c。队列于北京时间16:31:47结束，11个阶段及总审计均通过，136条执行记录全部成功，无失败或重跑；墙钟37.42分钟。新增10个模型（2个短测试、2个历史复现anchor、6个正式KD0）及835个实验文件全部下载到本机并逐项校验。完整结果见[本批报告](reports/experiments/attribution-study-20260911-42527/README.md)。服务器study为`/root/autodl-tmp/textconsistency/runs/attribution-study-20260911-v1`，状态passed；PID1479已完成，禁止重复启动。完成时GPU无计算占用；之后用户告知已关闭，本机诊断阶段未再次登录核实电源。

**当前结论：简单融合的收益稳定，复杂框架的额外贡献尚不稳定。** 三种子HM（Base/Novel均值计算）：UCF101基线/简单融合/原完整方法为74.42/78.76/80.66，DTD为63.79/70.75/69.51，Aircraft为34.26/38.31/36.22。简单融合在三个数据集均3/3种子赢CoOp；原完整方法相对简单融合分别2/3、1/3、0/3种子正向。完整模型KD1−KD0的平均配对融合ΔHM在DTD为−0.49±1.20、Aircraft为−1.16±0.90；此前UCF为+1.14±0.46。不能把层级/KD作为跨数据集稳定有效的贡献，也不能按测试分数为不同数据集挑KD设置。

**研究边界：暂停原框架的扩量训练。** 精简候选已按预设Base验证规则执行，三个数值条件均未通过，停止本候选扩量。当前三个数据集已参与研究决策，后续要保留独立确认实验；Base验证不能当作Novel验证，不直接据此宣称已达到录用条件。论文/Overleaf未修改，本轮没有额外启动新训练。

**上一批工作：DTD/Aircraft固定三种子验证已完成。** 正式18次训练/84次独立重载全部通过，另有6次单轮训练/28次重载单列保存；4个阶段和最终总审计均通过，141条命令无失败或重跑。模型代码和固定协议不变。完整压缩包及24个模型均已下载到本机并逐个校验。见[完整实验报告](reports/experiments/crossdataset-study-20260911-55424/README.md)及[备份收据](reports/experiments/crossdataset-study-20260911-55424/backup-receipt.json)。

**上一批结论：整体方法提高HM，但层级独立优势未被验证。** DTD原方法/基线/参数匹配平面的HM为69.51/63.79/69.76；Aircraft为36.22/34.26/36.63。原方法相对基线在两个数据集各3/3种子的HM提高，伴随Base下降、Novel上升。层级−平面的平均配对HM分别为−0.29（1/3正向）和−0.42（0/3正向）。结合UCF101仅+0.26、2/3正向，目前不能把层级优于平面作为稳固贡献。后续简单融合、KD配对及ADK公式适配现已完成，结论见顶部。

**上批 GPU 完成时状态（历史入口）：bjb1:55424。** RTX5090 32GB、驱动595.71.05、torch2.8.0+cu128，队列墙钟约42.65分钟。PID1576为已完成队列的历史记录，恢复时先核对真实进程及status.json，不可重复启动。实验目录为`/root/autodl-tmp/textconsistency/runs/crossdataset-study-20260910-v1`。最终实测GPU无计算进程、利用率0%、显存2MiB；服务器电源由用户控制。

**上一批次：结构对照已完成、审计通过、完整模型已备份。** 共15次正式训练/90次独立重载，另外5次单轮验证/30次重载不计入正式结果。原层级融合HM **80.66**，参数匹配双头平面 **80.38**；原层级平均配对优势仅 **+0.26个百分点（SD1.18，2/3种子提升）**，层级结构的独立收益尚不稳定。交错、随机17、随机29的HM分别为79.86、81.20、80.49，全部保留，不能挑选最佳Novel成绩改称最终方法。原结构全部3seed/3分支的18个Base/Novel准确率精确复现历史结果。详见 [完整结构对照报告](reports/experiments/ucf101-structure-study-20260910/README.md)。本批没有仍在运行的训练；后续DTD/Aircraft验证也已完成，结论见顶部；不继续用Novel调参。

结构批次开卡时入口 `connect.bjb2.seetacloud.com:27946`；当时RTX5090 32GB、90GiB内存、驱动580.76.05。当前GPU批次见顶部。实验目录 `/root/autodl-tmp/textconsistency/runs/structure-study-20260910-27946-v2`；PID1553/1574/3154仅为历史记录。队列完成时GPU显存/利用率均为0。原始status.json的`awaiting_analysis_and_backup`是执行器结束时快照；后续分析与本机备份均已完成，以本批analysis.json和backup-receipt.json为证。服务器电源仍由用户控制。

**本批 GPU 基线与 KD 对照已完成并备份。** 共15次正式训练、78次独立重载评估，完整审计通过。官方 CoOp matched20 与旧基线的三个种子准确率完全一致；本轮KD1也逐种子复现历史结果。完整方法KD0→KD1的融合HM为79.50→80.66，3/3种子提高；普通层级为79.91→80.19，仅2/3种子提高。详见 [完整报告](reports/experiments/ucf101-kd-study-20260910/README.md)。这是历史批次；DTD/Aircraft正式验证现已完成，见顶部。

## 目标与当前授权

- 目标：完成可复现、有实际贡献的期刊投稿，争取在 2027 年春节前获得录用；尚未确认目标期刊和学校认可条件，未承诺录用。
- 当前优先代码和实验，暂不修改论文/Overleaf。
- 最新用户授权直接实施文本侧残差三组小实验并通过AutoDL开卡/训练/备份/关机，现已全部完成。代码和结果推送本仓库；没有自动追加训练。研究主线仍为模型改进，论文/Overleaf暂不修改。
- 此前“我开机了你开始吧”授权的H1 GPU固定实验已执行完：4次训练/4次导出，独立复核和完整备份通过，三个数值门槛全未通过，已停止。电源/模式由用户切换，已告知该批可以关卡。
- 上一阶段用户“你开始吧”授权执行H1本机实现和CPU核验，随后“你继续”恢复中断的审计修复；该阶段代码与CPU检查已完成，没有登录服务器、启动GPU或修改论文。准备完成不能当作H1效果或创新性得到验证。
- 用户2026-09-12开启GPU并授权旧模型验证集推理；沿用bjb1:42527，整批已完成并备份，零新增训练。GPU结束时显存和利用率均为0，已告知用户可以关卡；电源与模式由用户操作。
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
| H1结束后的选题筛查 | 历史记录；实验研究路线已取消 | 用户选择模型改进；旧 8/24 次矩阵不得启动，不再等待学校形式确认 |
| H1独立队列及服务器无卡准备 | 已完成，并已执行GPU计划 | efcd3db；各193测试/四份prepare；后续4次训练均完成，不能再用原准备入口重跑 |
| 修复训练/保存/评估与记录 | 已完成 | 原 23 项测试 + 三种子审计 |
| UCF101 三种子实验 | 已完成 | 上述报告；禁止当成所有数据集结论 |
| 服务器连接与环境检查 | 最新GPU批次完成 | westc:26022，RTX4090 24GB/90GiB；结束时GPU空闲，完整模型已备份，电源由用户操作 |
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
| AAPE / ADK 接近方法比较 | ADK公式适配GPU评估及审计已完成 | 三数据集全三种子/六分支保留；适配不等于官方复现，AAPE未取得作者实现 |
| 简单融合与跨数据集KD贡献验证 | 已完成、审计通过并完整备份 | 136条命令全部成功；10新模型/835实验文件本机哈希一致；简单融合胜DTD/Aircraft完整框架，KD融合增益分别−0.49/−1.16 |
| 关卡后错误诊断与相关工作核对 | 已完成 | 126组预测转移、54组融合分解、2232行全类别统计、6对KD张量核对；7项接近工作及新颖性边界，见diagnostics报告 |
| 旧模型交叉融合工具与准备 | 实际GPU推理、本机复算和备份已完成 | 19条服务器命令、10次本机复算；9组/27旧模型/1500记录/216方法种子行；候选三个数值筛查均失败，停止扩量 |
| KD退步的本机归因与文献核对 | 已完成，独立核验通过并归档 | 1500条记录、375行类别、36组正确性组合、36组置信度分箱；18组历史loss梯度核验，零新增训练；见kd-failure-20260912 |
| H1：删除目标二元KL、保留加权非目标项 | CPU/GPU核验完成，停止扩量 | seed1 SN+M为83.82/72.92/29.00；三个条件全未通过；不是新算法有效性证据 |
| H1服务器部署与独立Base验证队列 | 真实实验及独立审计、备份已完成 | 4训练/4导出；anchor最大差0，29组独立指标复算；73文件/4模型逐SHA验证，study为stopped |
| 视觉适配＋描述监督原型 | 实现、配置、CPU核验和独立复审完成 | 205项测试；8组旧行为132张量完全一致；未跑真实数据或GPU；旧8/24次草案取消 |
| Aircraft 视觉适配 A/B/C 先导准备与运行 | 已完成、核验、备份 | RTX4090固定1a38307，3训练/3导出；C−A -1.50pp、C−B -1.50pp，未通过继续扩量的门槛 |
| 投稿实验定稿及论文修改 | 待上述证据 | 用户当前要求暂不处理论文 |

## 实验顺序和判断规则

KD、结构、DTD/Aircraft及最新描述融合/KD贡献验证批次均已完成，模型和评估齐全，已备份到本机；恢复时先阅读报告和备份收据，避免重复启动。结构代码的固定协议见 [结构协议](docs/STRUCTURE_CONTROLS.md)，最新贡献验证见 [固定计划及完成记录](docs/plans/2026-09-11-attribution.md)。下一步基于完整正负证据决定精简或改进假设，不重复已完成批次，不先扩量再寻找贡献。

1. 已完成matched20基线核验：本仓库基线与官方 CoOp 的模型计算是否一致；配置是否一致。这与“官方论文指标已复现”不同。
2. 已完成UCF101 KD=0/1 成对实验：同一个模型结构、teacher CE、seed/采样、训练轮数、融合权重，只改变 KD。分别报告 student、teacher、ensemble。
3. 已完成UCF101层级结构核验：参数匹配后的层级优势较小且非所有seed正向，分组敏感性存在；不能据此认定层级贡献稳定，不将顺序分组称为语义分组。
4. 已完成DTD、Aircraft三种子：整体方法相对基线的HM均提升；层级对参数匹配平面在两个数据集都没有平均优势。最新简单融合、固定ADK公式适配和跨数据集KD对照也已完成，均不支持当前复杂模块的稳定独立收益；不是ADK官方复现。
5. 层级/KD跨数据集独立收益不稳定。旧模型交叉融合和随后H1固定损失消融均已完成，均未通过各自的事前数值条件；停止扩量。H1仅有seed1 Base验证证据，没有Novel/HM；不盲目叠模块，不按新分数改融合规则或门槛。后续已形成描述监督视觉适配候选并完成本机准备，见顶部。

本轮三个数据集用于研究决策，并不是任何期刊通用的“足够发表”门槛。不根据 Novel 测试结果选择模型、超参数或删掉不好的 seed。训练轮数更长不保证 Novel/HM 更高，基线需按明确协议判断。

## 接近工作及创新边界

- [AAPE，NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/file/93e67adb632d636bb2bcedf4eef1a6fe-Paper-Conference.pdf)：已有图像对齐的描述聚合及表示蒸馏；与当前 logit KD / 双分支推理不同，区别的效果需验证。
- [ADK，2025 预印本](https://arxiv.org/pdf/2512.17313)：已有描述均值与图像引导的无参数描述选择。
- [最新蒸馏修补文献核对](reports/diagnostics/kd-failure-20260912/LITERATURE.md)：DKD已有目标/非目标分解，logit标准化已有尺度约束分析，ProGrad和可靠性加权也已有先例。仅用已知NCKD分解消融不构成新算法；端点logit导数不构成参数梯度或训练因果证据。
- 不能把“多描述 + 注意力 + 蒸馏”整体作为未经对照证实的独有创新。参数匹配、简单融合和ADK公式适配对照均已完成：跨数据集证据尚不支持两级聚合的稳定独立价值；ADK适配不等同官方复现。当前描述监督视觉适配假设已单独固定协议，不依据 Novel 挑分组、KD 或分支。

## 恢复工作时必读

1. 读本文件，再读最近 preparation / experiments 报告和对应 JSON；以实际日志和产物为准。
2. 最新 GPU 队列已完成：先读 reports/experiments/text-residual-20260914 的 README、status.json、verification.json、power-control.json。PID1239已结束，实例已关机，不重复运行 `text-residual-20260914-d5152e1`。结果后续决策见顶部；旧描述纠错、视觉适配、H1、8/24矩阵均无待执行队列。
3. 当前实例为 westc:26022 的 autodl-container-a5904787c3-600682b1，根目录 `/root/autodl-tmp/textconsistency`。本次迁移 Aircraft、CLIP 和源代码；旧实例的 UCF/DTD/历史 runs 未迁移，完整历史备份仍在本机。切换入口或实例后须重新确认，不能凭相同目录名假设磁盘相同。
4. 新代码必须部署到独立 release 目录，运行使用新输出目录；不覆盖旧结果或伪装复用了已跑数据。
5. GitHub 保存源码、配置、计划、汇总、审计；数据集和 CLIP/训练权重另存。ImageNet-1K 尚未准备，不属于当前无卡阶段。
6. 每次完成后更新本文件：实际完成项、结果/失败、证据路径、下一条命令。训练中如需跨会话跟进，应先记录运行目录与状态，不靠聊天记忆。

## 历史贡献验证与完整备份（2026-09-11）

- [完整报告](reports/experiments/attribution-study-20260911-42527/README.md)、[机器可读分析](reports/experiments/attribution-study-20260911-42527/analysis.json)、[完成状态](reports/experiments/attribution-study-20260911-42527/completed-status.json)、[本机备份收据](reports/experiments/attribution-study-20260911-42527/backup-receipt.json)。
- 本机完整备份目录：工作区 `outputs/textconsistency-attribution-study-20260911-42527/`，含完整压缩包、解压副本及逐文件验证收据。
- 压缩包68,345,250字节，SHA256为`a1214f69edf0c033fbf4bfc5506f4d420c3dcc5653bc21af6f17064bd451c5bf`；835个实验文件和10个新增模型均已核对。
- 本批不重复打包数据集、冻结CLIP或历史18个参考模型；历史完整模型备份仍保留，准备收据固定其SHA与来源。GitHub保存全部分支、逐种子CSV、审计和非权重元数据，权重保存在服务器及本机。
- 该历史study状态passed，禁止重新运行历史启动器。本批之后的旧模型验证及H1对照也已完成，见顶部最新报告。

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
