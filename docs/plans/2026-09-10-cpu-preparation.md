# 无卡阶段：基线与消融准备

用户要求：先完成无卡环境、数据、实验脚本准备，再由用户开卡；将计划及当前进度长期保存到 GitHub。当前不改论文，不启动正式 CPU/GPU 训练，不覆盖已有实验/源代码快照，不提交密钥、数据集、权重。

## Task 1: 准备受控 KD 对照和多数据集运行入口

工作目录是当前 worktree。你拥有 scripts/repro/run_pilot.py、scripts/repro/summarize.py、scripts/repro/preflight.py、configs/repro/dtd.yaml、configs/repro/fgvc_aircraft.yaml，以及 tests/repro/test_runner.py 和新增 tests/repro/test_preflight.py。不修改模型数学实现、trainers、原 UCF101 配置和其他文件。不要自行 spawn 子代理或推送/提交。

保持原 UCF101 默认行为和旧实验协议兼容。添加 --dataset ucf101|dtd|fgvc_aircraft（默认 ucf101）；数据集显示名与训练 loader 一致。新两个配置完全复制原统一配置，不按已观察的测试分数调参。

增加 --kd-weight（默认 1.0，有限非负数）。对所有非 baseline 变体只覆盖 KD_WEIGHT，不改变 teacher CE、结构、推理分支、融合系数。baseline 明确 KD_WEIGHT=0。protocol 保存 dataset、kd_weight、核心固定设置和每条命令。新 KD=0 与 KD=1 必须从相同 seeds / 训练样本和新的输出目录成对重跑；不自动复用旧代码 hash 结果。

预检泛化到三个数据集：必须存在固定 Zhou split JSON（UCF/DTD）或 Aircraft 官方 train/val/test metadata，禁止缺失时随机生成划分；验证所有图像存在、原始划分无路径交叉、每类别描述覆盖、Base/Novel 类别不交叉、16-shot 样本数、1/2/3 不同 seed 的样本哈希。检查返回数据集名与输入对应。允许 dataset loader 自行建立合法 few-shot 缓存。

汇总器读取 protocol 的数据集而不写死 UCF101；检查训练与评估 manifest 的 dataset/seed/shots/code/model_signature/指定 epoch/checkpoint 以及训练数据身份，防止不同实验被混成一组。对新协议验证 KD 等设置与实际配置匹配，但保持旧 protocol 可读。Base/Novel 评估的 class-dependent signature 项如有设计原因不同应沿用已验证规则，不盲目整对象比较导致错误。详细研究现有保存格式后做最小修改。

添加行为测试：不同数据集命令、KD 0/1 唯一受控改变且 baseline 无 KD、拒绝负值/NaN、缺失固定划分被拒绝、汇总拒绝数据集/KD/源码混用及错标 seed。先写关键失败用例再实现。使用 PYTHONPATH=.:Dassl.pytorch ../../tc-venv/bin/python -B -m unittest discover -s tests -v。CPU 小型模型测试允许，禁止完整 benchmark 训练。

完成后写完整报告到本计划 SDD workspace 的 task-1-report.md，说明文件、测试、限制。返回简短摘要。主代理负责官方基线核查、真实数据部署和进度文档，其他人也在此代码库工作，不要撤销他人的改动。

## 后续主代理工作

- 连接用户指定的无卡服务器，确认历史数据和结果保留；准备 DTD / Aircraft 数据与固定 split，核验哈希。
- 查官方 CoOp 上游源代码及 base-to-new 默认配置，记录与当前实现不同处，准备独立 baseline 实验而不冒称已复现。
- 对新增 runner 进行独立审查及 CPU 检查；部署新 release 目录，保留旧 src；生成计划和 readiness 证据。
- 新建 PROJECT_STATUS.md 作为持续更新入口，保存历史三种子结果及阶段表、后续实验、命令、路径、判定标准。
- 经检查后将代码、报告、状态文档快进推送到用户已授权仓库 main；开卡与正式训练等待用户后续通知。
