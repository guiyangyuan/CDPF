# DTD 与 Aircraft 固定配置验证

## Spec / Global Constraints

用户已授权执行下一步并持续保存进度到其 GitHub。首次服务器接入失败已解决：用户随后明确无卡开机，同一入口现已连接；本轮完成CPU准备，GPU训练等待用户开卡。不得关闭实例、保存凭据、覆盖或删除旧模型/实验。不开额外的调参批次。

准备边界澄清：原图、固定划分元数据及描述文件保持不变；数据加载器按固定seed生成或复用`split_fewshot`派生缓存属于已授权的数据准备，允许并应记录。prepare-only把协议和审计证据写入明确的准备目录，不创建训练study目录。该限制不禁止标准数据加载器的派生缓存，也不要求改写数据加载器。

本批只运行DTD和FGVC Aircraft，使用已验证的原训练源码，SHA256为`9bd4a7f248f09c47be17c1bb64921bd9a29d9f3b46e03e8caac9b934b2a95bba`，对应训练代码commit`13b34978eeb8b23532b53abd1066af4719a0f909`。不修改train.py、trainers、datasets、clip、Dassl或原数据/训练配置。允许新增独立的实验编排/审计脚本。

正式比较baseline、hierarchical_adapter、flat_dual_adapter，固定seed1/2/3、16-shot、ViT-B/16、FP32、20epochs、batch16、N_CTX4、N_VIEWS20、N_GROUPS4、KD1（baseline有效KD0）、KD温度2、teacher CE1、diversity0、contrastive0、alpha0.5。主分支baseline为student，其余为ensemble；full模型仍报告student/teacher/ensemble全部分支。原连续分组不变。不选择Novel成绩最好的分组、seed、epoch或超参数。

baseline是本仓库CoOp实现，已在UCF101核验与固定上游CoOp等价；本批不能声称重新执行了官方上游DTD/Aircraft。层级和双头平面参数均790018，baseline2048。

执行顺序固定：smoke_dtd、smoke_fgvc_aircraft（各3变体×seed1×1epoch），全部通过后formal_dtd、formal_fgvc_aircraft（各3变体×3seed×20epochs）。短测试6次训练/28次重载不纳入正式统计；正式18次训练/84次重载。新目录、无自动重试、不覆盖现有输出。任何失败记录状态和日志，停止后续阶段；保留已完成结果。服务器接入不可用只阻塞部署/训练，不阻塞本地准备。

## Task 1: 实验编排和完整审计

Ownership: 新增`scripts/repro/cross_dataset_study.py`、`scripts/repro/cross_dataset_audit.py`和`tests/repro/test_cross_dataset.py`。无需改旧模型或run_pilot/summarize入口。不创建通用任务平台，不实现自动续跑。

1. 提供reviewable、无需GPU或数据访问且无输出目录副作用的plan模式；源目录、数据根目录、全新输出根目录作为参数。复用源目录的run_pilot.build_commands/CLI，不复制训练命令逻辑。CLI模式至少plan-only、prepare-only、execute、audit-only，互斥；prepare-only只在明确证据目录写CPU预检及协议，不创建训练study目录。
2. 运行前固定核对模型source_digest、相关执行器/配置/templates文件SHA、完整CLIP基础权重SHA（`5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f`）、数据预检、CPU/GPU环境和空闲GPU。所有子进程CPU线程设4，CUBLAS_WORKSPACE_CONFIG=:4096:8，去掉继承的TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD。prepare允许CPU，execute无CUDA直接拒绝，不能降级为CPU长训练。
3. 复用preflight.check对两个数据集检查固定划分、每类16shot、描述覆盖。记录期望每seed train/val采样哈希、Base/Novel test哈希和实际数量、类别列表，来自真实dataset loader；用于后续独立核对，不写死Aircraft测试数量。核对训练时使用的preflight与准备证据一致。
4. 状态status.json原子写入stage/state/PID/每阶段命令开始结束码/完成列表。stdout/stderr写分阶段log；无凭据。执行每阶段后audit gate，通过才进入下一阶段。每数据集正式结果独立存储，不能在合并时混数据集。
5. 审计先调用既有summarize.collect，再补它仅在structure_study=True时具备的检查：固定协议身份；实际checkpoint SHA/sidecar/epoch/provenance/signature；可信自产checkpoint在CPU显式weights_only=False加载，参数计数排除token_prefix/token_suffix缓冲区，所有参数有限；每分支重载指向相同指定模型；split必须test；train/val/test哈希与准备数据一致；精确Base/Novel类别集合及描述顺序（baseline无描述）；同seed跨变体训练样本一致；配置固定值完整且层级/双头平面只允许aggregation及运行路径差异，不能允许无关优化参数漂移。基线与两full模型的通用训练配置仅允许明确的结构/KD/分支差异。比对eval日志correct/total与metrics的准确率及真实测试数量。短测试和正式分开审计。
6. analysis JSON包含每数据集全部21个seed/branch行、7个summary行、原层级−平面以及原方法−baseline的逐seed Base/Novel/HM差、均值和样本SD；主比较不能跨分支挑最高。全部正式84项独立eval完整才写最终passed结果；未齐全必须拒绝。无需此任务写人为结论或备份打包（控制者接手）。
7. 有针对性的测试覆盖无副作用计划及确切阶段/命令计数、拒绝现有study目录/无GPU执行、拒绝不完整结果、篡改checkpoint/test哈希/配置/计数、dataset和baseline参数量差异正确处理。复用已有fixture思路或构造小型真实JSON/checkpoint，不能依赖GPU和下载。运行有意义的focused tests和既有repro tests一次，记录结果；不重跑GPU。使用工作区Python：`/Users/bytedance/Documents/Codex/2026-09-08/https-github-com-zhengli97-awesome-prompt/work/tc-venv/bin/python`。

## Task 2: 固定协议、接入、执行和保存

控制者负责协议文档、进度文档、CPU准备证据、部署、GPU运行、完整结果分析/备份、GitHub推送；源码实现和测试由Task1负责。Task1通过审查后才部署。若用户当前入口仍不可用，提交并推送实际准备状态，不宣称已训练；待接入后完成已授权的同一批实验。

新的正式study目录拟为`/root/autodl-tmp/textconsistency/runs/crossdataset-study-20260910-v1`；启动前核实不存在。source根目录沿用只读`releases/structure-20260910-v1`，操作脚本单独部署，禁止把脚本命名为queue.py以免遮蔽Python标准库。

Task1和Task2接口为计划JSON、CPU准备证据、分阶段status、最终analysis；必须记录执行过的操作脚本SHA，版本快照和日志随备份保留。旧source model SHA保持不变，操作脚本版本另记录。

## 2026-09-11 实际阶段状态

Task1已实现并通过针对性审查；Task2本轮CPU范围已完成：最终操作代码6ce9c6e独立部署，服务器59项测试通过，本机与服务器实际loader的样本/类别证据一致，无CUDA拒绝训练检查通过。新study目录未创建；开卡脚本已保存、仅做语法检查。最终审查发现的共享配置漂移漏报已修复：准备时用原train.setup_cfg冻结全部136条命令的完整有效配置，审计训练/评估时逐项对照并独立核对seed；最终复审通过。GPU短测试、正式训练、结果分析及完整模型备份留待用户开卡后执行，不能将本轮准备证据当作训练结果。详见[准备报告](../../reports/preparation/crossdataset-20260910/README.md)。

## 2026-09-11 GPU执行完成

用户开卡并提供bjb1:55424后，固定四阶段及总审计全部通过。18正式训练/84重载、6短训练/28重载全部完成，无失败或重跑；模型及科学协议不变。完整24个模型和原始日志已下载到本机，压缩包和逐模型SHA均验证。Task2的执行、分析、备份及报告已完成，见[完整实验报告](../../reports/experiments/crossdataset-study-20260911-55424/README.md)。本计划不授权据Novel选择新分组或继续扩大实验。
