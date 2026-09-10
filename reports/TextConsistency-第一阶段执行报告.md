# TextConsistency 第一阶段执行报告

2026-09-10。已完成代码修订、本地环境搭建、数据预检及真实模型的小样本流程验证。尚未完成 GPU 正式实验，不能据此确认论文方法有效或替换论文性能表。

## 保存位置

- 修订项目：`outputs/textConsistency-repro/`，入口说明是 `README.md`。
- 可迁移源码：`outputs/TextConsistency-repro-source.zip`，包含代码、模板、依赖版本、脚本和文本验证证据。
- 改动补丁：`outputs/TextConsistency-修订.patch`；文件清单：`outputs/TextConsistency-改动清单.json`。
- 原始代码仍在 `/Users/bytedance/Documents/textConsistency`，300 个既有文件逐项哈希核对未变。原来的未提交修改也保留。
- 当前运行环境：工作区 `work/tc-venv/`；UCF101 解压在 `outputs/datasets-prepared/ucf101/`；官方 CLIP ViT-B/16 权重在修订项目 `clip/ViT-B-16.pt`。

源码包不包含数据集、预训练权重、实验 checkpoint 或 Python 虚拟环境。数据准备脚本与 README 提供恢复步骤；换服务器后需重新生成带路径的实验计划。

## 解决的主要问题

1. **评估身份不可靠**：Base/Novel 不再暗换预测分支；同一个完整 checkpoint 分别评估 student、teacher、ensemble。指定 epoch 不存在或模型不完整时停止。
2. **部分模块没有被训练或保存**：保存 soft prompt 和整个视图聚合器；adaptive CE 为可学习的聚合器提供梯度，KD 对 teacher 停止梯度。旧冻结特征对比损失被禁用。
3. **消融因素混在一起**：mean / flat / hierarchical 使用共享实现，全局 MLP 查询适配独立开关；新脚本生成六组可比较的配置。
4. **文本视图处理有偏差**：描述去重、token 截断后去重、显式 padding mask，分组不再丢弃尾部；按顺序分组如实标注，不能再声称已实现四类语义层次。
5. **无法追溯实验**：输出实际 seed、采样哈希、代码和模板哈希、完整模型哈希、有效配置与机器可读指标。汇总拒绝缺失种子、错误标签和不匹配的 Base/Novel 模型。
6. **运行环境不能直接启动**：恢复缺失的模板依赖，修复 PyTorch 2.8 预热学习率 API，记录完整环境版本。

这次修订改变了方法实现，旧日志中的准确率不能自动沿用。原论文方法部分也需要同步调整，具体在项目 `REPRO_METHOD.md`；本次未直接改 PDF 或 Overleaf。

## 已执行的验证

| 检查 | 实际结果 | 适用范围 |
|---|---|---|
| 行为与集成测试 | 23 项全部通过 | 小型真实 PyTorch 模型、梯度、保存/加载、运行脚本等 |
| 官方 CLIP + UCF101 真图像 | soft prompt 与 MLP 一步更新，图像编码器保持冻结 | 两类 smoke，不能当性能实验 |
| 独立进程恢复 | logits 最大差 0.0 | 最终代码与同一组实际输入 |
| 正常训练入口 | 训练→保存→Base→Novel 全流程通过 | 专用四类、1-shot、1-epoch fixture |
| 完整 UCF101 数据预检 | 7639/1898/3783 个 train/val/test 路径存在、无跨 split 路径重叠 | 未做内容级去重 |
| 三个种子的采样 | 每个 seed 的 Base 51 类、816 张；Novel 50 类；采样哈希互不相同 | 16-shot 数据加载 |
| 环境依赖 | pip check 无依赖冲突 | Python 3.12.14 / torch 2.8.0 / torchvision 0.23.0 / CPU |
| 正式 GPU 入口 | 无 CUDA 时训练前停止 | 防止把本机 CPU 验证误当正式 GPU 运行 |

最终训练源码 SHA-256：`d74a9770d0d814c56269486e2492327881633d98314281d0f13a584ee279f17f`。证据索引是 `textConsistency-repro/verification/README.md`。

## 已准备的下一阶段

单种子 UCF101 pilot：baseline、mean、flat、flat_adapter、hierarchical、hierarchical_adapter；统一 16-shot、20 epochs，6 次训练、32 次评估。三种子版本会执行 18 次训练、96 次评估。两份计划都已生成，正式命令尚未启动。

首轮需要测出实际显存和耗时，并检查 teacher/student/ensemble 的真实变化。随后再决定是否扩大到多个数据集及补 KD 对照。当前 baseline 是共享实现下的 CoOp 风格基线，还没有独立复现官方基线协议；投稿前必须说明协议并补齐必要比较。

继续沿用当前方向的决定需要真实实验支持：若层级结构或 MLP 没有稳定贡献，就简化方法或调整贡献表述；不能靠补写未经验证的数字维持原来的故事。

## 当前缺少的条件

原 RTX 4090 服务器不可用，本机环境没有 CUDA/MPS，因此完整 GPU 训练、三种子结果、扩展数据集验证仍未完成。需要新的可连接 GPU 服务器，或先确认租用方案与预算后准备算力；本次没有租用、付费或启动任何付费资源。

针对春节前取得录用的目标，建议争取在 10 月底至 11 月初完成第一轮投稿，为返修留出时间。是否能按时录用取决于真实实验进展和期刊审稿，不能保证。选择期刊前还需要明确学校认可范围、是否要求检索、以及版面费上限。
