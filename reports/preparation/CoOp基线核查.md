# 官方 CoOp 基线核查（2026-09-10）

已完成源码/配置核查、两张真实训练图像的CPU数值诊断，以及官方matched20三种子GPU复现；三种子的Base/Novel准确率与历史本仓库基线完全一致。官方200轮参考配置尚未执行。

## 固定来源

- [作者仓库固定提交](https://github.com/KaiyangZhou/CoOp/tree/ff61507c790454bce7c5052c3ac39e60772f1f89)
- [官方 CoOp trainer](https://github.com/KaiyangZhou/CoOp/blob/ff61507c790454bce7c5052c3ac39e60772f1f89/trainers/coop.py)
- [官方 vit_b16_ctxv1 配置](https://github.com/KaiyangZhou/CoOp/blob/ff61507c790454bce7c5052c3ac39e60772f1f89/configs/trainers/CoOp/vit_b16_ctxv1.yaml)
- [CoCoOp 论文，训练细节与 Base-to-New 任务](https://arxiv.org/pdf/2203.05557)

论文训练细节明确 CoOp/CoCoOp 均使用 ViT-B/16、4 个上下文 token 和 “a photo of a” 词向量初始化。仓库的 vit_b16_ctxv1.yaml 设为 200 epochs、train batch 32、test batch 100；这些是该文件的明确设置，不能仅凭此文件断言所有论文/任务均采用同一训练日程。上游当前只有 scripts/coop/main.sh 通用入口，没有 scripts/coop/base2new_train.sh；不编造该文件。

## 与现有 UCF101 实验的差异

| 项目 | 现有受控实验 | 官方来源/准备的参考配置 |
|---|---|---|
| 模型 | 冻结 ViT-B/16，学习共享上下文 | 相同基本形式 |
| 上下文长度 | 4 | 4（词语初始化决定实际长度） |
| 初始化 | N(0, 0.02)，随机 X token | a photo of a 词向量 |
| 训练轮数 | 20 | 上述官方 YAML：200 |
| train/test batch | 16/16 | 上述官方 YAML：32/100 |
| 精度 | FP32 | 上游默认 FP16；本次拟用 FP32，明确作为适配差异 |
| 图像划分 loader | UCF101/DTD/Aircraft | 三个文件逐字与上游相同 |
| 预测与 checkpoint | 强制记录 SHA 和指定 epoch | 官方只学/存 prompt，需要外部记录来源和严格指定加载 |

当前模型初始化差异并不说明原有实验无效：六组仍是同协议的受控比较；但不能据此标成“官方协议已复现”。训练更久也不保证新类表现更好，不能只挑最好 Novel epoch。

## 已执行的数值验证

`audit_coop_equivalence.py` 在两个独立 Python 进程中加载上游/当前实现，共用同一官方权重和相同两张 UCF101 训练图像；固定 4 个随机上下文 token，FP32。只进行一次 SGD 更新。

- prompt 初值、logits、CE、prompt 梯度、更新后 prompt 的最大绝对差均为 **0**。
- 除 prompt 外所有参数冻结且没有梯度。
- [证据 JSON](coop-equivalence-local.json)。这不是准确率实验，也不能代替完整训练比较。
- 服务器无卡模式重复该诊断时，official worker 被 SIGKILL；容器 memory.max=2147483648（2 GiB）。未记录为通过，也不直接断言是 OOM；保留失败日志，开卡后在更充足资源下复验。本机通过结果仍有效。

## 两个独立比较的状态

1. 已完成 `matched20`：上游模型代码，20 epochs / batch16 / FP32 / 4 随机 token，与当前受控基线的设置对齐，验证完整训练路径。
2. 待执行 `reference200`：上游 vit_b16_ctxv1 配置，200 epochs / batch32 / a photo of a 初始化，FP32，Base-to-New；作为更接近官方配置的参考，明确这不是 20-epoch 同预算消融。

两者均 seed1/2/3、16-shot、固定最后一个 epoch，Base/Novel 独立重载同一模型；禁止基于 Novel 测试选 epoch。先在 UCF101 执行，再决定后续扩展预算。运行上游时使用本项目 vendored Dassl（含 torch2.8 学习率兼容修复），记录依赖源码哈希；不宣称完整原始软件环境复现。

## 2026-09-10 开卡复验更新

内存恢复到120GiB后，服务器相同完整模型 CPU 诊断通过，五项最大绝对差仍为0。随后 GPU 真实短训练、保存、重载评估通过，matched20正式三种子核验已完成，逐种子准确率与历史基线完全一致。[运行记录与证据](../experiments/ucf101-kd-study-20260910/README.md)。此前无卡 SIGKILL 作为历史失败保留，不再是待处理的阻塞项。
