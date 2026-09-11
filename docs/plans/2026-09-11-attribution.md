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

Ownership: scripts/repro/attribution_study.py、scripts/repro/attribution_audit.py、tests/repro/test_attribution.py。只新增操作代码；借用旧run_pilot.build_commands、cross_dataset_study和cross_dataset_audit的可复用纯函数，不修改既有模型。具体任务在Task1接口审查完成后补充冻结。

计划使用历史六个KD1模型对照新训练六个KD0完整层级+MLP模型；固定额外DTD/Aircraft seed1各一次KD1重复训练作为跨批次重现检查，重现不符先停查原因。所有训练使用同一旧source及固定配置；短测试和正式结果分开。完整模型报告student/teacher/ensemble。prepare核对旧输入和真实loader、冻结完整命令实际cfg；execute无GPU拒绝、全新目录、逐阶段失败即停，不自动重试。Task1的18次baseline重载融合评估与KD阶段分离保存。

## Task 3: 接近方法核对、部署和记录（控制者）

控制者负责原始模型复用清单、接近方法研究报告和适配边界、协议/进度文件、版本部署、无卡准备、启动脚本和GitHub推送。研究及实现审查可并行，但不同时派两个实现代理。所有正式结果待GPU真实运行才能报告；无卡准备不能等同完成训练。若需要接近方法新实现，先在本计划追加精确公式/约束及单独任务，经审查后部署。

Task1输出独立评估metrics/预测文件，Task2编排调用并统一审计；控制者提供复用模型目录映射和实际SHA，不由实现者猜路径。任务接口前置冻结，源代码SHA不变。准备、训练、审计、备份分别记录。
