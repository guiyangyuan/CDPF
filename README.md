# TextConsistency：复现修订版

2026-09-10。本版本从当前工作文件复制并修订；原目录 `/Users/bytedance/Documents/textConsistency` 保留。已完成 UCF101 三种子、18 次训练、96 次独立评估。完整方法 HM 80.66，当前 CoOp 风格基线 74.42；其他数据集、KD 独立贡献和官方配置基线尚未验证。旧论文数字未被自动验证。

**后续接手、查看完成进度和下一步，请先读 [PROJECT_STATUS.md](PROJECT_STATUS.md)。**

已完成本地正确性检查及 RTX 5090 上的 UCF101 三种子实验，详见 [实验报告](reports/experiments/TextConsistency-UCF101三种子实验报告.md)。无卡准备和开卡后的完整模型/GPU 短训练检查已通过，已完成并审计[官方基线与 KD 三种子对照](reports/experiments/ucf101-kd-study-20260910/README.md)。

## GitHub 文件索引

这是 `guiyangyuan/textconsistency_v2_tune` 中持续更新的复现代码与研究记录。当前完成情况以 [项目进度](PROJECT_STATUS.md) 为准。

- [项目计划、当前进度和接手说明](PROJECT_STATUS.md)
- [UCF101 三种子真实实验](reports/experiments/TextConsistency-UCF101三种子实验报告.md)
- [官方 CoOp 基线核查](reports/preparation/CoOp基线核查.md)
- [无卡准备及后续运行说明](docs/CPU_PREPARATION.md)
- [执行报告、代码核查及论文评审](reports/README.md)
- [本地验证记录](verification/README.md)
- [方法与实验表述边界](REPRO_METHOD.md)
- [原始交付源码包](artifacts/TextConsistency-repro-source.zip)、[校验信息](artifacts/TextConsistency-源码包校验.json)、[修订补丁](artifacts/TextConsistency-修订.patch)

源码包是上传前的本地交付快照，GitHub 版本另补充了文件索引和报告。二者训练源码一致。历史报告/JSON 中的本机绝对路径用于溯源，换服务器后按下文重新创建环境、准备数据并生成运行命令。

数据集、预训练权重和实验模型没有提交到 Git；获取脚本及官方权重校验信息已包含在项目中。

## 当前修复

- Base 与 Novel 使用同一个已保存模型、同一预测分支规则；分别记录 student、teacher、ensemble。
- 保存 soft prompt 与整个视图聚合器，检查实际加载文件、epoch、代码、模板、CLIP、seed 与采样信息。缺失、旧式不完整或配置不匹配的模型直接报错。
- mean / flat / hierarchical 聚合统一实现，MLP 独立开关。非整除分组不丢弃尾部描述；重复文本、token 截断后重复文本与 padding 不再被当作额外视图。
- 训练可学习的聚合器使用 adaptive CE；KD 对 teacher 停止梯度。拒绝启用旧的冻结特征对比损失。
- 删除按数据集暗改损失或预测分支的路径；记录实际参数与机器可读指标。
- 修复缺失的 `imagenet_templates.py` 依赖和 PyTorch 2.8 学习率预热 API 兼容问题。

`docs/history/README-original.md`、原有中文论文说明、旧结果文件和旧 shell 脚本用于历史追溯。此次实验入口是 `scripts/repro/`。旧的 `CoOp_TextConsistency`、`CoOp_TextConsistency_Avg`、`CoOp_TextConsistency_V2` 已被主入口拒绝，避免误用未经修复的训练实现。

## 环境

本地验证环境：Python 3.12.14、PyTorch 2.8.0、torchvision 0.23.0、FP32、CPU。当前环境未检测到可用 CUDA/MPS。

在服务器上先进入本项目目录，创建独立环境。下面 CUDA 12.6 命令来自 [PyTorch 官方历史版本安装说明](https://pytorch.org/get-started/previous-versions/)，需要服务器驱动支持；此前 RTX 5090 实测使用的是 torch 2.8.0+cu128 / torchvision 0.23.0+cu128，当前无卡服务器保留该环境；cu126 仅为其他机器的安装示例。

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r requirements.lock.txt
python -m pip check
export PYTHONPATH="$PWD:$PWD/Dassl.pytorch"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
python -c 'import torch; print(torch.__version__, torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU only")'
```

`requirements.lock.txt` 是本地验证环境的完整版本快照，不是跨平台二进制锁文件。先安装目标平台的 torch/torchvision，再安装其余固定版本；如平台需要调整依赖，保存新的 `pip freeze` 并重新验证。`requirements.txt` 另列直接运行依赖。不要额外安装 vendored Dassl 的旧 requirements，它含有不同版本的依赖。

macOS CPU 验证可将上面的 PyTorch 安装命令换为：

```bash
python -m pip install torch==2.8.0 torchvision==0.23.0
```

本次已有环境位于工作区 `work/tc-venv`。从本项目目录可使用 `../../work/tc-venv/bin/python`，无需再次安装。

## 准备数据与权重

在本项目根目录运行，`/absolute/...` 替换为真实服务器路径：

```bash
python scripts/repro/prepare_assets.py --ucf-zip /absolute/datasets/ucf101.zip --data-root /absolute/datasets-prepared --download-clip
python scripts/repro/preflight.py --data-root /absolute/datasets-prepared --output verification/server-data-preflight.json
```

脚本只接受已核验的作者 UCF101 midframes ZIP，校验 SHA-256 后解压；官方 CLIP 权重校验后保存到 `clip/ViT-B-16.pt`。已有文件校验失败时不会覆盖。UCF101 是动作中间帧分类数据，不是完整视频训练集。

本机数据已准备在：

`/Users/bytedance/Documents/Codex/2026-09-08/https-github-com-zhengli97-awesome-prompt/outputs/datasets-prepared`

其原始 train / val / test 分别有 7639 / 1898 / 3783 张图像。当前 Base/Novel 为 51/50 类，16-shot Base 训练每个 seed 为 816 张图像。seed 1、2、3 的实际采样哈希互不相同；检查报告是 `verification/ucf101-preflight.json`。文件路径无交叉已核验，尚未额外做图像内容级重复检测。

## 先验证，再跑 pilot

```bash
python -B -m unittest discover -s tests -v
python -B scripts/repro/smoke_real.py --data-root /absolute/datasets-prepared --output verification/server-real-smoke --stage train
python -B scripts/repro/smoke_real.py --data-root /absolute/datasets-prepared --output verification/server-real-smoke --stage reload
python -B scripts/repro/smoke_cli.py --data-root /absolute/datasets-prepared --output verification/server-cli-smoke
```

两个 smoke 脚本使用 CPU：第一个以真实 CLIP/图像检查梯度及独立进程加载后 logits 一致；第二个构造明确标注的 4 类、1-shot 验证划分，调用正常训练/评估命令，检查训练→保存→Base→Novel 完整流程。它们的准确率均不能填写进论文。每次使用新的 output 路径。

先打印正式 pilot 将执行的全部命令，随后启动单种子 GPU 实验：

```bash
python scripts/repro/run_pilot.py --data-root /absolute/datasets-prepared --output-root runs/ucf101-pilot-s1 --seeds 1 --plan-only
python scripts/repro/run_pilot.py --data-root /absolute/datasets-prepared --output-root runs/ucf101-pilot-s1 --seeds 1
```

默认 6 个变体、20 epochs、1 个 seed，共 6 次训练和 32 次独立评估。默认要求 CUDA，未检测到 GPU 会停止。服务器首次正式训练若显存不足，应记录后统一调整所有变体的 batch size 并重新生成实验协议；当前 5090 已验证该配置；未在当前 24GB 4090 上验证。

| 变体 | 描述聚合 | 全局图像 MLP 查询适配 | KD |
|---|---|---|---|
| baseline | 无多描述分支 | 无 | 无 |
| mean | 均值 | 无 | 有 |
| flat | 单层注意力 | 无 | 有 |
| flat_adapter | 单层注意力 | 有 | 有 |
| hierarchical | 顺序分组的两层注意力 | 无 | 有 |
| hierarchical_adapter | 顺序分组的两层注意力 | 有 | 有 |

这是一组受控的 CoOp 实现比较；baseline 是本仓库共享训练路径下的 CoOp 风格基线，不能声称已经复现官方论文的原始协议或数字。CoCoOp 修订通过小型模型的行为测试，尚未开展真实 CLIP 的完整训练。

固定项：同一描述文件、最多 20 条有效描述、同一数据划分与实际 seed、16-shot、ViT-B/16、4 个 context token、FP32、SGD 0.002、20 epochs、最后一个 epoch、KD 温度 2、KD 权重 1、adaptive CE 权重 1、对比/负熵权重 0、ensemble alpha 0.5。Base 和 Novel 都遵守同一预测规则。各变体的 student、teacher、ensemble 单独报告，baseline 只有 student。

单种子结果只用于流程、耗时与早期可行性检查，不作为投稿结论。根据 Base 验证集检查或选择超参数，不能用 Novel 测试结果选模型/调参；需要在最终测试前固定正式协议。

确认首轮可行后，用全新目录跑三种子：

```bash
python scripts/repro/run_pilot.py --data-root /absolute/datasets-prepared --output-root runs/ucf101-s123 --seeds 1 2 3
python scripts/repro/summarize.py --root runs/ucf101-s123
```

汇总器要求协议列出的运行齐全，拒绝缺失 seed、伪装的种子目录、非 16-shot、不同 Base/Novel checkpoint，以及不一致的代码/模型签名。输出 `results.csv` 与 `results.json`，记录 Base/Novel 均值、样本标准差、均值的调和平均及每 seed HM 的均值。一个 seed 时标准差为空。

每个运行目录保存 `effective_config.yaml`、`run_manifest.json`、完整 checkpoint、SHA-256 及 `metrics.json`。`train_samples_sha256` 是有序路径/标签/类名的哈希，不是逐张图像内容哈希。代码哈希覆盖训练入口、trainers、datasets、CLIP 和 Dassl Python 源码；脚本与配置另行保存在交付包及实验协议中。

本版本明确拒绝自动恢复旧训练目录。中断后保留已有产物，在新目录重跑尚需验证的实验；汇总必须对应一套完整协议。只加载自己保存且信任的实验 checkpoint。

## 接下来要完成的投稿工作

1. 已完成 UCF101 首轮和三种子实验；下一步执行 [当前计划](PROJECT_STATUS.md) 中的官方基线/KD 对照。
2. 检查完整模块相对 mean/flat 的贡献，以及 teacher 与 student 各自效果。若仅 ensemble 变好，不能将其解释为 student 蒸馏能力改善。
3. 有稳定贡献后，扩大到代表性数据集并运行 3 个 seed，补真正独立的 KD/MLP/层级消融；不要把当前 6 组误称为所有损失项的完整消融。
4. 新结果替换论文中的占位或无法溯源数字，方法表述按 `REPRO_METHOD.md` 修订。
5. 确認学校认可的期刊要求与投稿费用上限后，再选择刊物和倒排投稿时间；当前没有承诺录用结果。
