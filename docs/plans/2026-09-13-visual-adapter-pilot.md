# Aircraft 视觉适配 A/B/C：固定先导实验

2026-09-13，读取本轮任何新准确率前冻结。承接[模型改进方案](2026-09-13-model-improvement.md)。用户要求先完成无卡准备，再告知何时开卡。本机数据准备与执行器验证见[本轮记录](../../reports/preparation/visual-adapter-pilot-20260913/README.md)；服务器部署状态以该记录为准。本文不是训练完成报告。

后续完成：用户更换到 westc:26022；最终本机/新服务器各 235 项 CPU 测试、实际输入检查、固定计划与备份通过，待用户开卡，尚未训练。最终源码 1a38307 和命令见[新实例迁移记录](../../reports/preparation/visual-adapter-westc-20260913/README.md)。执行器退出修复不改变本协议的模型、数据或数值门槛；旧服务器和本机计划均失效。

## 研究问题与对照

验证“固定类别描述监督视觉残差”是否比当前简单融合和同参数普通视觉适配更有用。三组均使用同一 CoOp、固定描述均值、图像/文本骨干和最终融合规则。

| 组 | 视觉分类适配器 | 描述 CE 权重 | 可训练参数 |
|---|---|---:|---:|
| A `simple_fusion` | 关闭 | 0 | 2,048 |
| B `visual_only` | 64 维共享残差 MLP，系数 0.2 | 0 | 67,584 |
| C `visual_description` | 与 B 相同 | 1 | 67,584 |

保持新原型训练源码 SHA256 `38d9afc3cdc0440695cb5eca5c59069f73ace1aba4478d5afc5b656050911840`。本轮只补准备、执行与导出工具，不改模型或训练实现。A 也重新训练，不将历史 checkpoint 混入本次配对。

## 固定范围与预算

- 仅 FGVCAircraft，seed 1、Base 50 类、每类 16-shot；800 张训练图、200 张 Base 验证图。历史 Aircraft 结果已经看过，因此这是开发实验，不是独立确认集。
- ViT-B/16、FP32、4 个上下文 token、最多 20 条描述（token 截断后去重，实际有效条数逐类记录）、batch 16、SGD lr 0.002 / momentum 0.9 / weight decay 0.0005、cosine、1 轮 1e-5 warmup。
- A/B/C 各从冻结 CLIP 起点训练，总计 20 轮；只取最后第 20 轮，不按验证成绩挑 checkpoint。
- 顺序固定为 A 训练/导出、B 训练/导出、C 训练/导出：**3 次新训练、3 次独立重载导出**。各组内 CoOp 与适配器联合训练；没有额外预训练阶段。
- 初始 CPU 输入校验完成后，整条队列（包含训练、导出、阶段校验和分析）最多 3,600 秒。不是保证一小时能跑完；超时即停止并保留现场。该时限也不控制云平台计费或关机。
- 失败、超时、手动终止均停止后续阶段；不重试、不续跑、不自动扩大到其他种子/数据集。已存在输出目录拒绝使用。

## 指标和继续条件

保存每张验证图的标签、样本 ID、类别顺序及三个 FP32 logits：S=CoOp 分支、D=描述分支、F=0.5S+0.5D。B/C 的两个分支均取实际适配后分类特征，不能使用旧 H1/交叉融合导出器假设的冻结图像特征。

主比较使用最终 F 的正确数：**C−A ≥ 1.0 个百分点且 C−B ≥ 1.0 个百分点**才记为继续信号。在同一 200 张图上，两项分别要求净多对至少 2 张，用整数正确数判断，避免四舍五入越过阈值。同时完整报告 B−A、C−A、C−B 在 S/D/F 上的准确率、改对数、改错数。

该条件是提前固定的低成本开发筛查，不是显著性检验或发表标准。三组会按完整计划执行，不因中间成绩好坏临时删组。若 C 不过线，停止扩大这个描述监督候选；B 的表现仍完整保留，但不得把 B 的成功改称 C 的成功。若过线，先据完整结果另行冻结独立/多种子确认和相近方法对照，再考虑投入更大预算。

本轮不计算 Novel/test 指标或 HM。官方 test 元数据只用于检查划分边界；prepare 校验训练/验证图像，验证阶段只推理 200 张 Base val。Base 改善不能证明 Novel 泛化。没有新准确率前，不宣称涨点或创新成立。

## 无卡准备与以后启动

在服务器独立源码 release 中使用已有兼容环境；下列绝对路径应替换为实际路径。保留 Python 虚拟环境入口本身，不将其符号链接解析为环境外的解释器。准备不构建真实 CLIP，适合先在无卡模式核对数据与生成计划；旧无卡实例仅 2 GiB 内存，不用 `train.py --no-train` 代替此准备命令。

```bash
PYTHONPATH=.:Dassl.pytorch /absolute/venv/bin/python scripts/repro/visual_adapter_validation.py prepare \
  --data-root /absolute/data \
  --clip-path /absolute/ViT-B-16.pt \
  --output-dir /absolute/fresh-prepared

PYTHONPATH=.:Dassl.pytorch /absolute/venv/bin/python scripts/repro/visual_adapter_study.py plan \
  --prepared /absolute/fresh-prepared/prepared.json \
  --plan-dir /absolute/fresh-plan \
  --output-root /absolute/fresh-run \
  --python /absolute/venv/bin/python
```

prepare 校验固定源码/CLIP/描述/划分元数据，重建 seed 1 样本并核对缓存，哈希并解码全部 1,000 张训练/验证图，核对验证变换，保存三组完整配置。plan 绑定源码、操作脚本、配置、输入与命令，不创建训练目录。生成目录、训练目录和源数据彼此分开，不覆盖任何旧结果。Mac 上生成的计划绑定 Mac 路径，不能直接复制成服务器可执行计划；服务器须重新 prepare/plan，并比较样本、资产和模型身份。

仅在服务器准备通过、用户开卡后执行：

```bash
PYTHONPATH=.:Dassl.pytorch /absolute/venv/bin/python scripts/repro/visual_adapter_study.py run \
  --plan /absolute/fresh-plan/plan.json
```

无 CUDA 时，该命令在创建训练输出及子进程前拒绝。运行中保存 status.json、每阶段日志、三组完整 checkpoint、三份 export.json/logits.npz 和最终 analysis.json。完成后独立复算，下载原始模型及导出并逐文件核对，再通知用户关卡；脚本本身不控制电源。GitHub 保存代码/配置/计划/核验与指标，图像和大权重另行备份。
