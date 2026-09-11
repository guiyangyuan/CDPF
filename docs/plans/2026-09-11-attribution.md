# 描述融合与蒸馏贡献验证

## Spec / Global Constraints

用户在评估后授权开始下一步：简单融合对照、跨数据集KD开关、接近方法比较；代码/进度/报告持续推送原GitHub，暂不改论文。2026-09-11实测bjb1:55424仍可接入，但为无卡模式、cgroup内存2GB。本轮先完成可审查的无卡准备；GPU执行只在实际可用时进行，禁止CPU长训练，电源由用户控制。

不覆盖历史源码、数据、模型、输出目录；不保存凭据；不基于Novel选择分组、alpha、种子、轮数、模块或最优分支。所有新程序位于scripts/repro，测试位于tests/repro；train.py、trainers、datasets、clip、Dassl和旧配置保持不变。旧模型使用对应的不可变源码重载，操作脚本独立版本化。三数据集UCF101/DTD/FGVCAircraft，ViT-B/16、FP32、16shot、seed1/2/3、20epochs、batch16、N_CTX4、N_VIEWS20、N_GROUPS4，沿用既有优化器/增强/描述及类别划分。基线2048参数；完整层级+MLP790018参数，teacher CE1、温度2、diversity0、contrastive0、融合alpha0.5。KD是唯一训练处理变量。

接近方法先核对作者代码与论文，必须区分官方实现复现、论文公式的受控适配及普通启发式对照；不把跨协议文献分数当作直接对比。若作者代码不可获得，明确记录，不自行声称官方复现。

## Task 1: 复用CoOp的固定描述均值融合评估

Ownership: 新增scripts/repro/evaluate_frozen_fusion.py及tests/repro/test_frozen_fusion.py。你不独占工作区，不回退他人修改，不生成子代理，不修改旧训练源码或配置。工作Python为/Users/bytedance/Documents/Codex/2026-09-08/https-github-com-zhengli97-awesome-prompt/work/tc-venv/bin/python；包式测试命令 `PYTHONPATH=.:Dassl.pytorch <python> -m unittest discover -s tests/repro -t . -q`。

提供可导入的纯张量融合函数及独立GPU评估CLI。CLI需source-root、baseline-dir、dataset、seed、split(base/new)、data-root、output-dir、expected-source-sha256；固定epoch20和alpha0.5，不提供搜索或训练功能。允许额外prepare/validate-only模式检查元数据，不加载CLIP模型；无CUDA执行在写输出前拒绝。输出目录必须全新且不在baseline/source/data下，不改写旧metrics。

从指定source-root的原run_pilot.build_commands及train.setup_cfg构造baseline对应split实际配置，使用原baseline trainer、原load_model完成严格重载；保留provenance/signature/seed/shot/code SHA检查，不临时伪造签名绕过。UCF101使用cpu-prep-20260910源码（model digest d74a9770d0d814c56269486e2492327881633d98314281d0f13a584ee279f17f）配历史本仓库baseline，DTD/Aircraft使用structure-20260910-v1（digest 9bd4a7f248f09c47be17c1bb64921bd9a29d9f3b46e03e8caac9b934b2a95bba）。可用显式source-root/digest逐次新进程运行，避免Python模块缓存跨版本。不得修改或注入旧源码。

同一batch计算student基线、description_mean、simple_fusion三个分支。描述来自原CachedMultiViewTextEncoder，保留原文本去重、token去重、mask、前20条、类别和顺序；描述向量先逐条单位化，在有效描述上均值，再单位化；图像特征单位化，teacher logits使用同一CLIP logit_scale.exp()。simple_fusion=(student_logits+description_mean_logits)/2，不先softmax，不训练teacher，不用完整模型蒸馏过的student替代原CoOp。description_mean和simple_fusion没有额外可训练参数。可复用原ViewAggregator mean计算并用独立数学fixture验证。

每次实际baseline student准确率/正确数必须与该checkpoint已保存的eval_{split}_student/metrics.json及log.txt相符，否则失败；对照其样本哈希、dataset、seed、checkpoint实际SHA和sidecar、epoch20、训练2048参数及有限性。跨机器拷贝可以重定位绝对目录，但不得更改checkpoint内容；历史路径仅是来源证据，校验实际SHA与历史SHA相同。当前loader train/val/test样本哈希应与baseline历史记录相同；实际测试数从loader获取。prepared/validate-only无需模型加载，可读YAML/JSON和哈希；完整张量/参数/签名验证在GPU执行时完成。

结果写metrics.json，含各分支correct/total/accuracy、实际checkpoint/hash、source/脚本/模板/CLIP SHA、完整有效cfg、描述有效数和顺序、数据采样hash、环境、固定公式、类别顺序。保留每张测试图片相对data-root路径、真值和三分支预测的JSONL及其SHA；核对count与accuracy，输出全部分支不挑选。错误记录状态，未完成不能标passed。源代码和数据在完成前后核对，禁止覆盖旧输入。

有意义的CPU测试：mask/归一化/缩放和logit融合数学fixture，非有限输入/空有效描述拒绝；拒绝错seed、错数据/样本、被篡改checkpoint/sidecar、旧基线不是student或非20epoch；已有输出与无CUDA时不写训练输出；元数据验证不能导入/构造完整CLIP。测试不下载权重。实现后跑focused及59项既有测试，提交仅所拥有文件，写实施报告。

## Task 2: DTD/Aircraft KD开关固定队列及独立审计

Ownership: scripts/repro/attribution_study.py、scripts/repro/attribution_audit.py、tests/repro/test_attribution.py。只新增操作代码；借用旧run_pilot.build_commands、cross_dataset_study和cross_dataset_audit的可复用函数，不修改既有模型。你不独占工作区，不回退他人修改，不生成子代理。工作Python为/Users/bytedance/Documents/Codex/2026-09-08/https-github-com-zhengli97-awesome-prompt/work/tc-venv/bin/python。

输入接口：--source-root指定旧structure源码，--data-root、--output-root、--input-receipt指定父代理已核实的JSON。receipt.fusion_baselines为9行(dataset,seed,baseline_dir,source_root,source_sha256,checkpoint_sha256,historical)，historical含base/new的metrics_sha256/test_sha256/accuracy；receipt.historical_kd1为6行(dataset,seed,model_dir,checkpoint_sha256,source_sha256)。所有checkpoint固定epoch20，source digest与原manifest一致。UCF source旧d74版本，DTD/Aircraft新9bd版本，必须逐进程加载避免模块缓存混用。receipt还含恢复UCF压缩包SHA等普通收据，不改变训练语义。

CLI互斥plan-only / prepare-only / execute / audit-only，--preparation-dir单独目录。plan不需GPU/数据，不能创建study；prepare只创建新准备目录、运行真实loader和CPU元数据核验，不加载完整CLIP；execute无CUDA先拒绝且不创建study。GPU必须空闲，不控制电源。固定source digest9bd4a7f248f09c47be17c1bb64921bd9a29d9f3b46e03e8caac9b934b2a95bba、CLIP SHA5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f，原train.py/trainers/datasets/clip/Dassl/configs只读。新增操作代码与所依赖旧编排/审计函数均hash冻结，prepare后变更要拒绝。

固定执行阶段（各阶段审计通过才继续，任一失败停止、不自动重试）：
1. smoke_dtd_kd0、smoke_fgvc_aircraft_kd0，各完整hierarchical_adapter/seed1/KD0/1epoch训练+6个Base/Novel三分支重载；短测试不计正式结果。
2. anchor_dtd_kd1、anchor_fgvc_aircraft_kd1，各同模型seed1/KD1/20epoch训练+6重载。各分支Base/Novel正确数与历史seed1必须完全相同，否则停止调查，不能悄悄放宽阈值或改称新基准。
3. fusion_ucf101、fusion_dtd、fusion_fgvc_aircraft，各原baseline三个seed×base/new共6次Task1独立评估；只做评估，不训练，全部三个分支都记录，主要对照为固定simple_fusion。
4. formal_dtd_kd0、formal_fgvc_aircraft_kd0，各完整hierarchical_adapter三个seed/KD0/20epoch训练+18次重载。
5. historical_dtd_kd1、historical_fgvc_aircraft_kd1，各历史3seed×Base/Novel×student/teacher/ensemble共18次独立重载到新输出目录，不能覆盖旧目录；与历史正确数/测试哈希/模型SHA核对。
合计2次短训练/12短重载、8次正式训练(含2anchor)/84个完整模型正式重载(含12anchor)、18次融合独立评估；旧模型不修改。训练/评估普通命令总数124；阶段审计及最终审计另计。新正式处理模型为6个KD0，anchor不是额外统计种子，旧KD1仍用于配对。KD只改权重0/1，其他训练cfg完全相同。

复用run_pilot.build_commands构造训练/普通评估，通过实际旧train.setup_cfg冻结每条完整cfg；重用历史模型eval时准确重定位--model-dir而非训练目录。prepare记录真实loader的train/val/test采样hash及计数/类别、模板sha和顺序、当前图像inventory、历史config/manifest/metrics/log/checkpoint及sidecar文件hash、操作脚本和源依赖sha、完整CLIPsha。不能只互相比较同轮cfg而漏过共同漂移。执行前后核对immutable输入，缓存split_fewshot属于允许的派生输出。

状态原子写status.json，含stage/state/PID、每条命令开始结束码、已完成阶段；raw stdout/stderr独立保存；训练开始前保存protocol/data_preflight，未齐全不能passed。所有子进程设置PYTHONPATH使用各自源码、CPU线程4、CUBLAS_WORKSPACE_CONFIG=:4096:8，去掉TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD。无卡prepare线程1。所有新输出必须与source/data/baseline/历史model隔离；输入若缺失、错seed/错数据/重复条目/路径混用均拒绝。GPU要求仅在execute验证。

独立audit检查普通重载metrics与actualcfg、实际checkpoint SHA/sidecar/epoch/signature/provenance/参数790018有限性；日志correct/total与accuracy、真实样本数/采样哈希、类别及描述顺序一致；同seed KD0/1完整训练cfg仅允许KD_WEIGHT和输出位置不同；三分支和两split同checkpoint。所有正式KD0/KD1×2datasets×3seeds×3branches共36行seed-branch，Base/Novel合为一行。融合3datasets×3seeds×3branches共27行，均校验Task1指标和逐样本预测文件SHA及计数。anchor独立报告不进入正式配对。最终分析分别给出全部seed/branch Base/Novel/HM、均值和样本SD、HMofmeans与meanSeedHM分开；KD1-KD0逐seed差，simple_fusion-baseline逐seed差，以及完整原方法ensemble-simple_fusion（DTD/Aircraft用旧KD1，UCF由父代理后续补已有完整结果来源）。未齐全或任何失败拒绝最终passed，禁止选择最优分支。

有针对性的CPU测试：124条命令/阶段结构；无副作用plan；拒绝无GPU与已有目录；冻结cfg和seed错误/统一漂移；输入checkpoint或模板SHA篡改；不完整/重复结果；错误correct/total、KD方向、anchor差异；CLI import/plan不需完整CLIP。小型真实JSON/checkpoint fixture优先，不能下载权重或跑GPU。Task1已经测试过的细节无需重复测试。focused和既有repro tests一次，记录输出并仅提交拥有文件。控制者负责真实服务器验证和备份。

## Task 3: 接近方法核对、部署和记录（控制者）

控制者负责原始模型复用清单、接近方法研究报告和适配边界、协议/进度文件、版本部署、无卡准备、启动脚本和GitHub推送。研究及实现审查可并行，但不同时派两个实现代理。所有正式结果待GPU真实运行才能报告；无卡准备不能等同完成训练。若需要接近方法新实现，先在本计划追加精确公式/约束及单独任务，经审查后部署。

Task1输出独立评估metrics/预测文件，Task2编排调用并统一审计；控制者提供复用模型目录映射和实际SHA，不由实现者猜路径。任务接口前置冻结，源代码SHA不变。准备、训练、审计、备份分别记录。
