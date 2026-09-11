# 旧模型验证集交叉检查：本机准备

2026-09-11。当前状态：**本机工具和九组真实 CPU 预检完成，独立审查及全部修复复审通过；未运行 GPU 推理，没有新增方法成绩。** 用户已关闭服务器，本阶段完全在本机完成。

## 本轮范围

从已保存的 27 个模型读取 CoOp、KD0、KD1，在 UCF101、DTD、Aircraft 的 seed1/2/3 上准备验证集推理。模型仍是原第20轮权重。唯一主要候选是 KD1 学生与固定描述均值的等权 logits 融合；所有学生/教师组合及等权概率对照全部保留，不搜索权重或挑选数据集设置。

原训练入口会创建优化器，且验证集缺失时可能退回测试集。本轮新增独立入口，直接构建历史模型类并严格加载权重；验证集必须真实存在。KD0、KD1 教师均保存，权重相同也不隐去重复对照。训练代码和历史指标保持原样。

## 固定输入和协议

- [诊断与科学假设](../../diagnostics/error-analysis-20260911/README.md)、[固定筛查与停止规则](../../diagnostics/error-analysis-20260911/MINIMAL_NEXT_STEP.md)。
- [原模型输入清单](../../diagnostics/error-analysis-20260911/evidence/proposed-probe-inputs.json)：27个 checkpoint 及 manifest 哈希、9组训练/验证样本哈希。
- [实现计划](../../../docs/plans/2026-09-11-validation-probe.md)、[机器可读范围](protocol.json)。
- UCF 模型源码摘要 `d74a9770d0d814c56269486e2492327881633d98314281d0f13a584ee279f17f`；DTD/Aircraft 为 `9bd4a7f248f09c47be17c1bb64921bd9a29d9f3b46e03e8caac9b934b2a95bba`。本机已定位并核对两份源码。
- CLIP ViT-B/16 权重为350,837,078字节，SHA256 `5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f`；没有重新下载。

## 完成记录

- 原有109项复现测试通过；当前实现与修复提交为`76922fa`、`3888aa3`、`62771b4`，报告提交`ce31832`。18项针对性测试通过，完整128项测试通过（11.9秒墙钟）。合并到main后再次[完整验证](evidence/final-integration-tests.json)：128项通过，测试用时7.025秒；[原始日志](evidence/final-integration-tests.log)已保存。完整测试从`tests/`运行，基线109项从`tests/repro/`运行，统计范围不同。
- 本机数据已找到：UCF有13,320张、DTD有5,640张、Aircraft有10,000张JPEG。图片数量和文件存在不代替真实loader验证。
- [独立历史采样核对](evidence/historical-sampling-check.json)通过：九份本机已保存采样缓存的SHA先匹配本轮输入快照，再读取并检查训练/验证样本哈希。每seed验证图片数为UCF204、DTD96、Aircraft200；合计1,500条记录、1,280张不同图片。训练/验证真实路径不重叠。该检查不等于新增导出工具实际运行通过。
- `ce31832`的[九组独立CPU预检](evidence/ce31832-local-audit.json)全部通过，总子进程时间41.89秒。每组在独立进程使用原始源码读取、采样和预处理，没有构建CLIP。
- [无CUDA拒绝检查](evidence/ce31832-cuda-unavailable-check.json)通过：导出返回1，明确记录CUDA必需，没有生成NPZ或成功的export.json。这是预期拒绝，不是失败的训练。
- [输入不变与归档收据](evidence/ce31832-evidence-receipt.json)通过：源码、数据路径/大小/mtime、非图片元数据SHA及108个checkpoint相关文件均与运行前一致。图片内容没有全部逐张计算SHA，因此这里只证明上述记录不变。
- [完整预检证据包](evidence/cpu-prepared-ce31832.tar.gz)包含九组prepared.json、实际命令/日志、核对结果及无CUDA拒绝记录，不含图片或模型权重。

| 数据集 | 每seed Base类别 | 每seed训练图片 | 每seed验证图片 | seeds | 实际解码总数 | 教师KD0/KD1参数相同 |
|---|---:|---:|---:|---|---:|---|
| UCF101 | 51 | 816 | 204 | 1/2/3 | 612 | 3/3 |
| DTD | 24 | 384 | 96 | 1/2/3 | 288 | 3/3 |
| Aircraft | 50 | 800 | 200 | 1/2/3 | 600 | 3/3 |

全部1,500条验证记录实际解码为float32的`[3,224,224]`张量，记录了顺序、样本及变换后张量摘要。这里的张量是图片预处理结果，不是模型输出；教师权重相同也不证明学生与教师的错误重合度。

## 操作入口和下一阶段边界

新增工具为 [validation_probe.py](../../../scripts/repro/validation_probe.py) 和 [validation_analysis.py](../../../scripts/repro/validation_analysis.py)。准备命令在CPU执行，示例：

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:Dassl.pytorch python -B scripts/repro/validation_probe.py prepare --inventory reports/diagnostics/error-analysis-20260911/evidence/proposed-probe-inputs.json --workspace-root /path/to/task-workspace --source-root /path/to/frozen-ucf-source --data-root /path/to/datasets --dataset ucf101 --seed 1 --output-dir /path/to/new-preparation/ucf101-seed1
```

`--workspace-root`是模型清单中`outputs/...`路径的共同根目录，不是默认的Git仓库根目录。源码目录需有对应版本的configs/templates和已核对的`clip/ViT-B-16.pt`。其余八组替换数据集、seed和匹配源码，不能混用历史版本。每次输出必须是全新目录。

换到服务器后必须按其真实路径重新运行prepare；不能修改本机prepared.json里的路径冒充服务器预检。可以在一个新的输入目录下保留模型清单要求的相对布局，模型本体、sidecar、run_manifest.json、effective_config.yaml都须提供。数据和CLIP可复用服务器已有且SHA匹配的副本，无需重训或重新下载。

之后的GPU阶段只对旧模型做推理：先验证模型加载、设备、输出一致性，再生成九组logits；export内含严格加载和共享特征核对。CPU分析读取这些真实输出，计算全部24个分支/融合的指标，最后完整九组一起执行固定筛查规则。命令依次为：

```bash
python -B scripts/repro/validation_probe.py export --prepared /path/to/prepared.json --output-dir /path/to/new-export
python -B scripts/repro/validation_analysis.py analyze --export /path/to/new-export/export.json --output-dir /path/to/new-analysis
python -B scripts/repro/validation_analysis.py aggregate --analyses /path/to/nine/analysis.json/files --output-dir /path/to/new-aggregate
```

最后一条的占位位置应传入九个实际analysis.json路径。 汇总会核验每份analysis引用的export.json文件哈希、NPZ和样本身份，并重新计算完整指标；修改分数、预测或计数都会拒绝。下载产物后保留每组export.json与其同名NPZ在一起，重新运行analyze生成本机路径的报告，再aggregate；不改写原始收据里的哈希。原prepared路径不可用时可依靠固定清单绑定的实际行摘要核验，若可用则额外核验其内容和文件哈希。以上export/analyze/aggregate命令尚未对本项目真实模型输出执行；合成测试不当作实验分数。当前阶段没有连接服务器。下一步可先无卡启动：部署本仓库工具和历史输入，按服务器路径重做九组prepare并核验环境；这些通过后再开卡执行旧模型推理。

## 执行中发现和修正的事项

- 自查补上共享CLIP迁移设备的正确顺序，以及图片批次送入模型设备的步骤；增加对应回归测试。
- 概率融合的NLL使用稳定的log-softmax/log-add-exp，避免用截断概率掩盖极端置信度。
- 本机核对脚本最初误把checkpoint部分当作只有三个角色键；实际还记录了教师相等性及张量键列表。核对在写结果前拒绝，按完整schema修正后通过，未修改实验输入或降低SHA要求。
- 首轮独立审查发现四项问题：概率对照误换主要候选、配置绑定不完整、实际前向输入缺少再核对、聚合仅检查输入相互一致。现已修复；[逐项复审](review/task-1-fix1-review.md)确认4/4解决，没有新增回归。[最终全分支审查](review/final-review.md)另发现三项边界问题：归一化列表未固定、导出行未绑定真实采样哈希、汇总未核验原始输出。已通过一轮修复补上：锁定历史transform、逐行重算采样摘要、由已校验logits重算全部指标/预测/计数；[最终修复复审](review/final-fix-review.md)确认3/3解决，没有新的严重问题。
- 旧版本`04a9142`、`1406ca3`的预检证据保留为修复前历史，不作为当前代码通过的依据。当前[状态收据](ready.json)指向`ce31832`的九组预检与工具SHA；[实现记录](review/task-1-report.md)包含测试命令及红绿结果。

## 阅读结果时的限制

Base验证集来自训练见过的类别，并沿用原训练的每类4张验证采样。它是有限预算的初步筛查，样本较少；即使通过也不是Novel/HM提高的证据。本轮不重训、不拟合温度或融合权重、不用Novel测试标签选方法。真实CUDA推理及模型在CUDA上的一致性仍须等用户开启服务器后单独验证，不能从CPU测试推断通过。
