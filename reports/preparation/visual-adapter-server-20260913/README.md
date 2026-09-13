# Aircraft A/B/C：服务器无卡准备

2026-09-13。**本机和服务器无卡准备均已完成，可以通知用户开卡。** 最终服务器 234 项 CPU 测试、真实资源检查、固定计划、无 CUDA 拒绝及跨机器比对均通过，证据已下载并逐文件核验。本轮没有真实模型训练或新准确率。最新恢复状态以[handoff-ready.json](handoff-ready.json)为准。

## 准备结果

- 同一无卡实例，内存上限 2 GiB，数据盘剩余约 37.1 GiB；Python 3.12.3、torch 2.8.0+cu128、torchvision 0.23.0+cu128。没有改变电源或运行模式。
- Aircraft 50 个 Base 类、800 张训练图和 200 张验证图；1,007 个输入文件各自在两台机器上完整校验。图像、CLIP、描述和元数据共 1,006 个文件逐 SHA 相同；缓存 pickle 含机器路径而字节不同，但其样本/标签/顺序分别与同一官方重建划分一致。
- 两台机器全部 1,000 张训练/验证图内容及解码信息一致；200 张验证图的实际 FP32 张量、标签和相对路径摘要也精确一致。没有用包含绝对路径的旧摘要直接跨机器比较。
- 最终服务器六个 CPU 阶段通过：环境、234 项测试、prepare、plan、预期无 CUDA 拒绝、实际变换比对。无 CUDA 拒绝返回 1 是预期结果，训练输出目录未创建。
- 原始服务器证据包 137,640 字节，SHA256 `6a5c69830551abd52df395ac0febf92b0c957223383f9b2e5f7eac9b7a2e22b5`。下载后核对包哈希、18 个清单文件及清单本身，共 19 个文件。
- 服务器计划 SHA256 `89833a45d30d44bec2f104f9edeb5ec8455aa30840d8ff92d0ebbc5ed69b1ec1`；计划绑定的训练、导出和执行脚本与本机最终源码一致。

固定实验是 Aircraft Base、seed 1、16-shot、三组各 20 轮，A 简单融合、B 普通视觉适配、C 描述监督视觉适配；各组独立保存最终 checkpoint 和 S/D/F。仅当 C 的固定融合分别比 A、B 高至少 1.0pp，才作为扩大确认实验的信号。三次训练、三次导出、整条队列 3,600 秒，无自动重试或扩量；不计算 Novel/test 指标。详见[事前固定协议](../../../docs/plans/2026-09-13-visual-adapter-pilot.md)。

## 本轮修复

首次在服务器执行原版 233 项 CPU 测试，232 项通过，1 项计时测试失败。该测试假定 0.55 秒内能启动至少两个 Python 子进程，在无卡服务器上不可靠。改为在两个真实子进程完成后确定性地推进执行器时钟，直接检查累计预算；保留另外的真实超时/进程组清理测试。首次失败日志及状态保留，没有将其记为通过。

本机复核另捕获了一个实际退出问题：SIGALRM 恰好发生在 CPython 获得 `_waitpid_lock` 后、进入释放锁的 try/finally 前，异常会使锁未释放，随后清理中的 `wait()` 卡住。新执行器在创建、等待、清理子进程时暂缓抛出信号异常，每次等待最多 0.2 秒，返回后检查信号；3,600 秒总预算不变。新增回归测试在真实锁获取后注入 SIGALRM，修复前稳定失败，修复后验证锁完整释放。

本机完整 **234 项 CPU 测试通过**；独立复审 **19 项执行器测试通过**，无剩余阻断问题。真实模型、数据划分、损失和训练参数都未改动；模型训练源码 SHA256 仍为 `38d9afc3cdc0440695cb5eca5c59069f73ace1aba4478d5afc5b656050911840`。

## 最终部署与恢复入口

最终运行源码 commit `153953a32a74e7dbf1c8784465b8b41cc769a5e8`，源码包 SHA256 `4ccb228b85554e840547d405d8b4426d6462efe262a4c3846b01dfe9fcb95ad9`。

- 实例：`autodl-container-a03b4fb96c-09b042f2`，入口 `connect.bjb1.seetacloud.com:42527`。
- release：`/root/autodl-tmp/textconsistency/releases/visual-adapter-pilot-20260913-153953a`。
- 准备与日志：`/root/autodl-tmp/textconsistency/setup-logs/visual-adapter-pilot-20260913-153953a`。
- 计划：上述准备目录下 `future-plan/plan.json`。
- 预留输出：`/root/autodl-tmp/textconsistency/runs/visual-adapter-pilot-20260913-153953a`。

旧 `9919345` 服务器目录保留首次 CPU 测试失败证据；`d9e1f7e` 目录仅解压过源码，未执行准备或训练。两者均被最终版本替代。旧本机 plan-local-v2 因执行器更新失效；不要重跑这些旧入口或历史 H1/8/24 次矩阵。

服务器准备完成并由用户开卡后，先核对当前实例/GPU和输出目录仍未使用，再执行：

```bash
cd /root/autodl-tmp/textconsistency/releases/visual-adapter-pilot-20260913-153953a
PYTHONPATH=.:Dassl.pytorch:tests OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
/root/autodl-tmp/textconsistency/.venv/bin/python scripts/repro/visual_adapter_study.py run \
  --plan /root/autodl-tmp/textconsistency/setup-logs/visual-adapter-pilot-20260913-153953a/future-plan/plan.json
```

本轮该入口仅在无卡环境中验证了拒绝启动，没有执行 GPU 训练。开卡会话中也不得覆盖已有输出或绕过计划身份检查。准备成功不表示模型已经涨点；完成实验并核对、备份后才给出结果。

## 完成证据

- [最终交接状态](handoff-ready.json)、[固定服务器计划](server-plan.json)、[六阶段状态](server-cpu-status.json)。
- [跨机器核对](cross-machine-verification.json)、[备份核验](backup-receipt.json)、[服务器原始证据包](server-evidence.tar.gz)、[本机 234 项日志](local-tests.txt)。
- [首次失败与锁等待记录](initial-failure-evidence.tar.gz)、[包哈希](initial-failure-sha256.json)。本机解包最初误用了不支持 tar filter 的系统 Python 3.9，随后使用项目 Python 3.12 完成并核验；没有影响服务器实验或原始包。

原始 server-ready 快照记录的跨机器检查 pending 是打包时状态；本机比对完成后另存 handoff-ready.json，并已上传服务器，未改写原始证据包。本机原始记录位于工作区 `work/visual-adapter-server-20260913/`。GitHub 保存代码、计划与审计，不保存图像或大权重。
