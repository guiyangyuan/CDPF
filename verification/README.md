# 本地验证记录（2026-09-10）

本目录只证明修订代码的局部正确性与运行流程，不是论文 benchmark 结果。

| 文件 | 证据 |
|---|---|
| `validation-summary.json` | 本地阶段汇总、代码哈希、完成/未完成项 |
| `behavior-tests.txt` | 23 项测试通过：梯度、mask、分支、完整 checkpoint、种子、汇总、学习率兼容等 |
| `ucf101-preflight.json` | 完整 UCF101 split 路径检查、Base/Novel 类别分离、三个种子实际采样哈希 |
| `real-clip-smoke-final/train_check.json` | 官方 ViT-B/16、两张实际训练图像，一步更新 soft prompt/MLP，冻结图像编码器 |
| `real-clip-smoke-final/reload_check.json` | 独立进程恢复，logits 最大绝对差为 0 |
| `cli-smoke/verification.json` | 普通 train.py：训练、保存、Base 与 Novel 独立进程评估通过，加载同一 checkpoint |
| `gpu-unavailable-guard.txt` | 本机没有 CUDA，正式 pilot 在训练前明确停止 |
| `pilot-plan.json` / `three-seed-plan.json` | 只生成的命令清单，尚未执行正式实验；绝对路径对应本机，换服务器请重新生成 |
| `original-preservation.json` | 原目录 300 个既有文件哈希均未变化 |

CLI smoke 使用专用 4 类、1-shot、1-epoch fixture。真实图像两类 smoke 同样不是标准测试。上述 smoke 文件中的任何 accuracy 都不能抄入论文表格。中途失败的兼容性诊断和旧版本 smoke 已放到工作区 `work/code-audit/`，不作为最终通过证据。

源码交付 ZIP 保留文本/JSON 证据，不附带模型 checkpoint、CLIP 权重或图像。当前工作目录中仍保留最终 smoke 的 checkpoint，服务器可通过脚本重新生成。
