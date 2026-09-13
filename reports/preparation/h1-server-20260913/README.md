# H1 无卡准备与独立验证队列

2026-09-13：**本机与服务器准备均已完成；没有新增真实数据训练，尚无 H1 准确率。**

此前已经实现 `weighted_nckd` 训练损失。本轮新增可核验的训练配置、独立导出、固定分析和有次数上限的执行队列，部署了已审查的版本 `efcd3db`。模型结构、在线教师训练和训练源码未改变。改过损失不等于已经涨点，也不代表创新或录用条件得到验证。

## 已完成的验证

- 本机与服务器完整 CPU 测试各 **193/193** 通过；新队列针对性测试 **24/24** 通过。
- 本机、服务器各四次实际数据 CPU prepare 成功。2,516 个输入文件、686,637,734 字节逐项核验，部署前后未改变。
- 三数据集共 2,000 张训练图、500 张验证图；四份准备收据含重复 DTD 配对，共核对 2,980 条图像记录。相对样本顺序、标签、图像内容、描述/CLIP、训练配置和变换规范在两台机器上一致。
- 不跨机器比较绝对路径、mtime 或包含绝对路径的变换摘要；每台机器内部仍严格验证完整收据。
- 实际 CLI 无 CUDA 拒绝测试通过，训练子进程数为 0。服务器只执行环境检查、测试、四份 prepare 和 plan，共七条成功命令。
- 服务器计划固定 13 个阶段、最多四次训练，预留训练目录尚不存在。准备没有加载真实 CLIP 模型训练，也没有执行 GPU 导出。
- 服务器 24 个证据文件已下载，在解包前后逐文件核对 SHA256。原始服务器证据包为 1,076,780 字节，SHA256 `05f8d9a02ccff45f4a1cbc2fcc381a88bbf3ec92718189ac30ceb1993d6072f3`。

| 准备案例 | 损失 | train / val | 服务器 prepare 耗时 |
|---|---|---:|---:|
| DTD anchor，seed1 | full_kl | 384 / 96 | 20.89秒 |
| UCF101，seed1 | weighted_nckd | 816 / 204 | 27.31秒 |
| DTD，seed1 | weighted_nckd | 384 / 96 | 17.68秒 |
| Aircraft，seed1 | weighted_nckd | 800 / 200 | 49.20秒 |

上述时间只是 CPU 准备耗时，不是训练速度或方法指标。使用旧 S1 的 500 条历史 logits 另做了独立算术 QA，计数和筛查结论精确一致；这些旧分数没有作为 H1 结果。

## 当前服务器与恢复入口

实际实例 `autodl-container-a03b4fb96c-09b042f2`，本次核实时无 GPU，cgroup 内存上限 2 GiB，数据盘剩余约 37.5 GiB。Python 3.12.3、torch 2.8.0+cu128、torchvision 0.23.0+cu128。未操作服务器电源。

- release：`/root/autodl-tmp/textconsistency/releases/h1-validation-20260913-efcd3db`
- 准备/日志：`/root/autodl-tmp/textconsistency/setup-logs/h1-validation-20260913-efcd3db`
- 待运行计划：`/root/autodl-tmp/textconsistency/setup-logs/h1-validation-20260913-efcd3db/future-plan/plan.json`
- 预留输出：`/root/autodl-tmp/textconsistency/runs/h1-validation-20260913-efcd3db`
- 计划文件 SHA256：`684ae3b6c6649cec536efb048e116443b0d8df136806569c18a20108b9a2dde6`
- 训练源码 SHA256：`387ea6e46aa0fa6cba5109e64baab94a548b35c200a1945267d392bc35a36677`
- 源码包 SHA256：`ab2cb0b7c0eb0338b0e151596ff96bc6bbcee60f7340c35dbea6fc824e28fab7`

本机原始材料保存在工作区 `work/h1-prep-20260913/`，下载副本在其中的 `server-evidence-efcd3db/`。Git 保存源码、计划、准备记录和轻量证据包，不重复放数据集或权重。

## 下一步：用户开卡后执行固定小规模对照

先重新确认当前 SSH 对应同一实例、GPU 可用，以及预留输出和锁不存在。如已有输出，先读状态，不重跑。使用服务器本地计划，不复制本机绝对路径计划。

```bash
cd /root/autodl-tmp/textconsistency/releases/h1-validation-20260913-efcd3db
PYTHONPATH=.:Dassl.pytorch:tests OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
/root/autodl-tmp/textconsistency/.venv/bin/python scripts/repro/h1_study.py run \
  --plan /root/autodl-tmp/textconsistency/setup-logs/h1-validation-20260913-efcd3db/future-plan/plan.json
```

此命令**本轮未执行**。固定顺序为一次 DTD full_kl 20 轮 anchor；只有 S/T/M 预测完全一致、logits 满足事前固定 `rtol=1e-5, atol=1e-4`，才运行 UCF101、DTD、Aircraft 各 seed1 一次 weighted_nckd。固定16-shot、ViT-B/16、fp32、T2、KD1、teacher CE1、alpha=.5，不重试、不改温度/权重/融合规则挽回。

主候选为 SN+M logits，三个数据集等权。相对两种 C+M 对照均要求宏平均差≥0且每数据集≥−1个百分点；相对 S0+M 要求宏平均差>0。门槛用正确数/总数的有理数计算，避免浮点零误判。失败即停，成功也只是探索性门槛；没有 Novel/HM 或三种子稳定性结论。

## 审查、修复与保留记录

两轮任务审查和最终整体验证通过。修复了导出目录可落入输入树、阶段间文件未完整绑定、简化“passed”标记可被接受，以及严格宏平均门槛的浮点零误判。当前执行器在每一步前后核对真实文件，并独立重算完整分析。

历史版本的准备和计划、预期失败测试、无 CUDA 拒绝记录均保留；旧计划不重新标记为新版本。一次本机清单脚本解释器版本错误、一次 QA 脚本导入路径错误已修正，失败日志保留，均未训练模型。服务器本次七条命令全部首次成功，无重跑。

留待下次修改的一项非运行问题：`stage_validator` 类型注解仍写两个参数，实际实现和调用均为三个；已记录审查意见，不影响本次执行。测试日志末尾的三行数据集提示是既有信息输出，不是失败。

本轮执行决定：沿用隔离工作树并使用新的服务器 release；保持历史训练设置，仅限制自动评估为 Base val；固定锚点容差，不按结果调整；候选固定描述分支也必须匹配历史输出。对应代价分别是保留额外材料、暂不提供 Novel 结论，以及不匹配时停止并诊断；不以放宽检查代替复现。

## 可核对材料

- [最终核验 JSON](final-verification.json)
- [服务器准备完成收据](server-ready.json)、[固定服务器计划](server-plan.json)
- [跨机器复核](cross-machine-verification.json)
- [完整证据包](evidence/h1-no-gpu-evidence-20260913.tar.gz)、[逐文件备份收据](backup-receipt.json)
- [工具用法](../../../docs/H1_VALIDATION_STUDY.md)、[此前损失实现](../h1-cpu-20260913/README.md)
