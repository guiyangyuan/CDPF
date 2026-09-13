# westc 新实例：Aircraft A/B/C 准备完成

> 后续GPU计划已执行、复核和备份完成，见[实验结果](../../experiments/visual-adapter-westc-20260914/README.md)。下文为准备阶段记录，启动命令不可重复执行。

2026-09-13 发起迁移；完成时间（北京时间）：2026-09-14T00:01:01+08:00。**代码、依赖、数据、固定运行计划和无卡预检已完成，可以开卡。** 没有 GPU 训练或新准确率。最新状态见 [handoff-ready.json](handoff-ready.json)。

## 已准备内容

- 新入口 `connect.westc.seetacloud.com:26022`，实例 `autodl-container-a5904787c3-600682b1`。无卡时半核 CPU、2 GiB 内存；Python 3.12.3、torch 2.8.0+cu128、torchvision 0.23.0+cu128。独立 `.venv` 安装同一依赖锁，`pip check` 通过；数据盘剩余约 44 GiB。
- 完整 Aircraft 10,000 张图片及元数据，传输包 2,755,127,692 字节。包 SHA256 `baf805b7b07092f5398f306f8020dc8d4ef0df62fc7e129a0223d32a65a62ab2` 验证通过。
- CLIP ViT-B/16 权重 SHA256 `5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f` 验证通过。
- 在新实例按官方划分重建 seed 1 缓存，Base 50 类、800 train、200 val，与本机样本、标签和顺序一致。1,007 个输入逐项检查，其中 1,006 个非缓存文件与本机逐 SHA 相同；缓存因绝对路径不同分别校验。200 张验证图实际 FP32 张量、标签和相对路径摘要与本机精确一致。
- 本机和服务器各 **235 项 CPU 测试通过**，本机另有 20 项执行器专项测试通过。服务器六个阶段全部符合预期：环境、测试、真实 prepare、plan、无 CUDA 拒绝、实际图像变换。预期拒绝 exit 1 不代表失败；训练输出目录始终未创建。
- 已下载并逐文件验证 37 个服务器证据文件（包含初次失败记录）。证据包 SHA256 `a950cc03bdf19185c5bcfad7b07d2f91a72f593fec4fd193df28ec02b52e301b`。训练、验证和执行器的计划绑定哈希与本机固定源码一致。

仅迁移当前先导实验所需 Aircraft 和 CLIP；其他历史数据集/模型仍以既有本机备份为准，不声称它们已经迁到此实例。数据集和大权重留在服务器及本机，GitHub 保存源代码、配置、计划、环境清单、日志和核验。

## 本次预检修复

半核 CPU 暴露了一项短时测试的启动速度问题，检查发现两项测试存在该假设，并复现执行器在取消 timer 前再次收到 SIGALRM、跳过异常清理的问题。新增确定性回归在原 153953a 上失败，修复后通过；信号在异常清理期间延后处理，两项真实计时测试等待目标阶段/孙进程就绪后才触发短 alarm。详细根因与前后证据见 [cleanup-investigation.md](cleanup-investigation.md)。初次失败仍保存在原始证据包的 `previous-attempt-153953a/`，没有被改写为通过。

最终运行源码 `1a3830754dbffa87f06348e842e68a07fb4054c7`（分支 `codex/visual-adapter-pilot`；main 对应实现提交 `e59753a`），源码包 SHA256 `591cd4b0b75f093965d7454ee4d741b10847887f7f5ebf3c5a7850ab1484192e`。只修改执行器与相关测试。模型训练源码仍为 `38d9afc3cdc0440695cb5eca5c59069f73ace1aba4478d5afc5b656050911840`。

## 开卡后的唯一当前入口

- release：`/root/autodl-tmp/textconsistency/releases/visual-adapter-pilot-westc-20260913-1a38307`
- 准备记录：`/root/autodl-tmp/textconsistency/setup-logs/visual-adapter-pilot-westc-20260913-1a38307`
- 计划：`/root/autodl-tmp/textconsistency/setup-logs/visual-adapter-pilot-westc-20260913-1a38307/future-plan/plan.json`
- 预留输出（准备完成时不存在）：`/root/autodl-tmp/textconsistency/runs/visual-adapter-pilot-westc-20260913-1a38307`
- 计划 SHA256：`8f2c65b4130021d62b66e0c33920c1110b0e8af293c58d69167bdd2928eebc5c`

用户开卡后，先核对实例/GPU、计划哈希和预留输出仍未使用，再执行：

```bash
cd /root/autodl-tmp/textconsistency/releases/visual-adapter-pilot-westc-20260913-1a38307
PYTHONPATH=.:Dassl.pytorch:tests OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
/root/autodl-tmp/textconsistency/.venv/bin/python scripts/repro/visual_adapter_study.py run \
  --plan /root/autodl-tmp/textconsistency/setup-logs/visual-adapter-pilot-westc-20260913-1a38307/future-plan/plan.json
```

继续遵守[事前固定协议](../../../docs/plans/2026-09-13-visual-adapter-pilot.md)：Aircraft Base，seed 1，16-shot，A 简单融合/B 普通视觉适配/C 描述监督视觉适配各 20 轮，3 次训练＋3 次导出，整队列 3,600 秒。C 的固定融合相对 A、B 均至少 +1.0pp 才作为继续确认信号；无自动重试或扩量，不使用 Novel/test 选参数。该准备不证明模型涨点或达到发表条件。

旧 bjb1:42527 当前连接被拒绝；旧 bjb1 153953a、westc 初次 153953a 以及本机旧 plan 均不是当前启动入口。H1、旧 8/24 次矩阵仍停止，不重复已完成的历史实验。旧实例数据和历史备份未删除。

## 证据入口

[交接状态](handoff-ready.json)、[服务器计划](server-plan.json)、[六阶段状态](server-cpu-status.json)、[最终远端核对](remote-final-check.json)、[跨机器比对](cross-machine-verification.json)、[备份收据](backup-receipt.json)、[原始证据包](server-evidence.tar.gz)。

原始 server-ready 的 pending-local-comparison 字段是打包时快照；之后另存交接与比对收据，并上传服务器，未改写原始证据包。本机完整记录：工作区 `work/visual-adapter-westc-20260913/`。
