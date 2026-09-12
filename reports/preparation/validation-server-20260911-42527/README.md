# 旧模型验证集推理：服务器无卡准备完成

2026-09-12 核实完成。**服务器九组 CPU 准备、128 项测试、下载备份与本机独立核对全部通过；GPU 推理尚未执行，没有新增方法成绩。** 用户开启无卡实例后，沿用已授权的 bjb1:42527 连接。本报告承接[本机准备](../validation-probe-20260911/README.md)，研究协议仍为[固定筛查与停止规则](../../diagnostics/error-analysis-20260911/MINIMAL_NEXT_STEP.md)。

## 已完成与证据

- [最终服务器状态](ready.json)、[本机独立审计](local-audit.json)、[逐组清单](cases.csv)。实际操作代码为 `de57efd7452e4feab3fd361ef9acac27a6dee4ef`；两份推理/分析工具的 SHA 与已审查版本完全一致。
- 服务器 Python 3.12.3、torch 2.8.0+cu128、torchvision 0.23.0+cu128、NumPy 2.2.6；无卡 cgroup 内存 2 GiB。准备时 CUDA 不可用，`nvidia-smi` 返回 `No devices were found`。真实执行环境见状态收据。
- 128 项测试通过，测试报告耗时 22.957 秒，子进程墙钟 32.221 秒。九组真实 loader/图像解码完成；无 CUDA 时导出明确拒绝，未生成模型输出。
- 27 个 epoch20 模型及其 manifest、配置和 sidecar，共 108 个文件核对通过，并复制到独立输入布局。两版历史模型源码、CLIP、描述模板均与本机固定输入一致，未重训。
- 三份前后快照一致：源码、非图像文件内容哈希、数据路径/大小/mtime、原始与复制后的 checkpoint 文件无变化。另对全部验证图片逐张核对 SHA：1,500 条记录、1,280 张唯一图片、179,110,246 字节，服务器与本机全部相同。
- 九组类别顺序、train/val/test 样本身份与标签、固定协议均与本机准备一致。九对 KD0/KD1 教师张量相同；保持全部对照。

| 数据集 | seeds | 每 seed Base 类别 | 每 seed 训练图片 | 每 seed 验证图片 | 状态 |
|---|---|---:|---:|---:|---|
| UCF101 | 1/2/3 | 51 | 816 | 204 | 3/3 通过 |
| DTD | 1/2/3 | 24 | 384 | 96 | 3/3 通过 |
| Aircraft | 1/2/3 | 50 | 800 | 200 | 3/3 通过 |

## 中断和修正记录

首次运行的 128 项测试有两项 CLI 测试因硬编码本机 Python 路径而报错。提交 `de57efd` 仅把两处测试解释器改为 `Path(sys.executable)`，本机两项复测通过，服务器完整 128 项随后通过；未改模型、方法或数据处理逻辑。

随后 Aircraft seed2 两次收到 SIGKILL（返回 −9），原日志和失败记录均保留。服务器 cgroup 的 OOM/oom_kill 计数为零，不能断言是内核 OOM。当时约 1.26 GB 为文件缓存；通过只读文件描述符对本任务模型和归档文件执行 `POSIX_FADV_DONTNEED` 缓存建议后，同一固定命令在新目录成功（24.014 秒），seed3 随后成功（21.827 秒）。运行增加分配器环境设置，未改 batch、算法或输入，也未删除文件或修改内存限制。成功的七组未重复执行。

最终有效执行记录包括完整测试、九组 prepare 和一次预期无 CUDA 拒绝，共 11 条。两次 SIGKILL 和首轮路径错误单独保存在完整证据包中，不将其描述成首次全部成功。

## 备份位置

- [完整证据包](evidence/validation-server-prep-20260911-de57efd.tar.gz)及[逐文件备份收据](backup-receipt.json)：1,934,701 字节，44 个 JSON/日志文件；本机逐文件大小与 SHA 均核对通过。SHA256：`0af066a01428856eadba0de9b6da1936485884ffe6a2ac32b731f0301feb1a5f`。
- 包中保留五个准备目录：原始失败、v2、resume、cache、final；包含所有成功 prepared.json、命令/日志、前后快照、图像哈希和未执行的 GPU 计划。包内没有数据图片或模型权重，历史权重备份仍保留。
- 服务器包：`/root/autodl-tmp/textconsistency/backups/validation-server-prep-20260911-de57efd.tar.gz`。
- 本机完整包与解压副本：工作区 `work/validation-server-20260911-42527/` 下的同名压缩包和 `unpacked/`。操作脚本原始副本保存在 [execution/](execution/)，用于追溯，不应在已有输出上重复运行。

## 下一步与恢复入口

1. 用户切换为有卡模式后，先核实当前 SSH、GPU、内存、原输入和输出目录状态。服务器根目录为 `/root/autodl-tmp/textconsistency`；当前 release 为 `releases/validation-ops-20260911-de57efd`，解释器为 `.venv/bin/python`。
2. 读取 `setup-logs/validation-prep-20260911-42527-final/ready.json` 和 [gpu-plan.json](gpu-plan.json)。九组 prepared 收据分布在 `-v2`（七组）、`-cache`（Aircraft seed2）、`-final`（Aircraft seed3），以计划中的绝对路径为准，不能覆盖原始收据。
3. 本次检查时预留 GPU 输出 `runs/validation-probe-20260911-v1` 不存在，计划状态 `not_executed`。开卡时再次确认，若已存在先读状态，不能直接重复运行。
4. 使用 27 个旧模型完成九组 Base 验证集导出，再做九组 CPU 分析及一次全组汇总。按照计划指定的 cwd/PYTHONPATH 执行；严格加载和共享特征核对须在实际 CUDA 导出中通过。不得把 prepare 成功当成 CUDA 前向成功。
5. 唯一主要候选仍为 `S1+M_logits`、alpha=0.5，同时对照 `C+M_logits` 与 `C+M_probability`，保存六个原始分支和全部 18 个固定融合。不开新训练、不拟合温度或融合权重、不根据 Novel 选设置。下载原始 logits 并核验后，才判断是否满足固定筛查规则。

Base 验证是已见类别上的初步筛查，不能替代 Novel/HM 或独立确认数据集，也不证明创新或录用条件成立。跨机器预处理摘要包含绝对路径，不要求该摘要相等；本轮比较的是图像内容、类别/样本顺序、变换定义、张量形状与类型。全数据集图像快照仍以路径/大小/mtime 为依据，逐图内容哈希覆盖的是全部验证图片。
