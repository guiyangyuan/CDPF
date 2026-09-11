# DTD / Aircraft固定三种子实验：完成、审计通过、完整模型已备份

2026-09-11（北京时间）。正式18次训练/84次独立重载全部完成；另有6次单轮短训练/28次重载，全部单列保存，不进入正式分数。4次阶段审计及最终总审计通过，141条命令均返回0。队列约42.65分钟；没有失败重跑，没有变更配置或筛选种子。

**结论：完整方法在两个数据集的三个种子上均提高了相对本仓库CoOp基线的HM，收益伴随Base下降和Novel上升；层级结构相对参数匹配平面没有表现出稳定独立优势。** DTD层级−平面仅1/3种子正向，Aircraft为0/3。结合此前UCF101仅+0.26平均配对HM、2/3正向的结果，当前不能把层级聚合稳定优于平面作为已验证的核心贡献。

## 主结果

主分支固定为baseline的student、两个完整结构的ensemble（alpha=0.5）。数值单位为百分比；下表HM由Base/Novel均值计算，标准差为三个种子的样本标准差。

| 数据集 | 方法 | Base均值±SD | Novel均值±SD | 均值的HM |
|---|---|---:|---:|---:|
| DTD | 本仓库CoOp基线 | 77.78±0.92 | 54.07±0.77 | 63.79 |
| DTD | 原层级+MLP | 75.77±0.35 | 64.21±4.17 | 69.51 |
| DTD | 参数匹配双头平面+MLP | 74.50±0.82 | 65.58±0.53 | 69.76 |
| Aircraft | 本仓库CoOp基线 | 37.54±0.92 | 31.51±1.84 | 34.26 |
| Aircraft | 原层级+MLP | 34.85±1.03 | 37.69±0.86 | 36.22 |
| Aircraft | 参数匹配双头平面+MLP | 34.61±0.52 | 38.89±0.78 | 36.63 |

## 配对差值与研究判断

下表先对每个seed计算HM差，再汇总均值和SD；与上表两个“均值的HM”相减是不同统计量，不能混用。

| 数据集 | 比较 | seed1 ΔHM | seed2 ΔHM | seed3 ΔHM | 平均ΔHM±SD | 正向种子数 |
|---|---|---:|---:|---:|---:|---:|
| DTD | 原层级−参数匹配平面 | +2.0347 | -0.1806 | -2.7323 | -0.2927±2.3855 | 1/3 |
| DTD | 原方法−基线 | +8.1390 | +6.3743 | +2.4963 | +5.6699±2.8865 | 3/3 |
| Aircraft | 原层级−参数匹配平面 | -0.1738 | -0.1606 | -0.9154 | -0.4166±0.4320 | 0/3 |
| Aircraft | 原方法−基线 | +1.7404 | +3.0565 | +1.1001 | +1.9657±0.9975 | 3/3 |

- DTD：原方法相对基线Base均值下降2.01个百分点，Novel上升10.14；原层级相比平面的平均配对HM为−0.29，且种子波动较大。
- Aircraft：原方法相对基线Base均值下降2.68个百分点，Novel上升6.18；原层级相比平面的平均配对HM为−0.42，三个种子都低于平面。
- 这些结果支持整体框架的Base/Novel权衡改善，不能将全部涨点归因于层级或KD。本轮未做两个数据集的KD开/关，也未跑接近方法的官方实现，尚不能声称SOTA或达到录用条件。
- 下一步建议：保留平面模型作为简化候选，降低层级贡献主张，优先完成接近方法的公平比较。不得根据本轮Novel结果更换分组、挑种子或宣称新的最优方法；此建议尚未开始执行。

## 完整分支汇总

下表保留全部分支；不从分支中选最高值替换预先规定的主结果。完整逐seed结果见[JSON](analysis.json)和[CSV](seed-branch-results.csv)。

| 数据集 | 变体 | 分支 | Base均值 | Novel均值 | 均值的HM | 每seed HM均值±SD |
|---|---|---|---:|---:|---:|---:|
| DTD | baseline | student | 77.7778 | 54.0660 | 63.7897 | 63.7884±0.7723 |
| DTD | hierarchical_adapter | student | 74.2284 | 57.0853 | 64.5379 | 64.5326±0.8679 |
| DTD | hierarchical_adapter | teacher | 69.7531 | 60.8696 | 65.0092 | 64.9451±2.2135 |
| DTD | hierarchical_adapter | ensemble | 75.7716 | 64.2110 | 69.5139 | 69.4583±2.3001 |
| DTD | flat_dual_adapter | student | 73.9198 | 57.5684 | 64.7274 | 64.7194±0.7299 |
| DTD | flat_dual_adapter | teacher | 69.8688 | 63.4863 | 66.5248 | 66.5050±0.9681 |
| DTD | flat_dual_adapter | ensemble | 74.4985 | 65.5797 | 69.7552 | 69.7510±0.1011 |
| Aircraft | baseline | student | 37.5350 | 31.5137 | 34.2618 | 34.2460±1.3144 |
| Aircraft | hierarchical_adapter | student | 35.9144 | 34.9930 | 35.4477 | 35.4405±1.1676 |
| Aircraft | hierarchical_adapter | teacher | 32.6531 | 39.0122 | 35.5505 | 35.5435±0.0723 |
| Aircraft | hierarchical_adapter | ensemble | 34.8539 | 37.6925 | 36.2177 | 36.2117±0.7791 |
| Aircraft | flat_dual_adapter | student | 36.2545 | 34.2531 | 35.2254 | 35.2108±0.7262 |
| Aircraft | flat_dual_adapter | teacher | 32.7531 | 39.1522 | 35.6679 | 35.6626±0.8769 |
| Aircraft | flat_dual_adapter | ensemble | 34.6138 | 38.8922 | 36.6285 | 36.6283±0.6326 |

## 固定协议与验证证据

- ViT-B/16、FP32、16-shot、seed1/2/3、20epochs、batch16；N_CTX4、N_VIEWS20、N_GROUPS4；两完整模型KD1、teacher CE1、温度2、diversity/contrastive0、融合alpha0.5；baseline有效KD0。层级连续分组不变，无超参数或分组搜索。
- 本仓库CoOp实现已在此前UCF101核验上游数值等价性，本轮DTD/Aircraft没有重新跑官方上游代码，不应称为“官方指标复现”。层级和平面均790018参数，baseline2048；相同参数量不代表所有计算性质相同。
- 服务器RTX5090 32GB，驱动595.71.05，torch2.8.0+cu128；GPU前后向通过。实际训练源码SHA256始终为`9bd4a7f248f09c47be17c1bb64921bd9a29d9f3b46e03e8caac9b934b2a95bba`，模型代码commit`13b34978eeb8b23532b53abd1066af4719a0f909`，操作代码commit`6ce9c6e2770c6359dcd1f8dfea326e3755aaa79c`。
- 准备时由原train.setup_cfg冻结136条实际命令的完整有效配置，训练/评估逐项核对；测试样本哈希、类名/描述顺序、16-shot采样、参数数目/有限性、checkpoint实际SHA/sidecar/epoch/来源均通过。每次重载的correct/total与accuracy及真实测试数量一致。
- DTD每seed训练384张，Base/Novel测试864/828张；Aircraft每seed训练800张，测试1666/1667张。使用固定划分；派生fewshot缓存允许生成/复用，原图/划分/描述不修改。图像清单为路径/大小/mtime校验，不是逐图内容SHA；基础CLIP、代码和相关元数据有SHA验证。
- [完成状态](completed-status.json)、[固定命令协议](protocol.json)、[完整分析](analysis.json)、[正式DTD审计](audit/formal_dtd.json)、[正式Aircraft审计](audit/formal_fgvc_aircraft.json)、[DTD短测试](audit/smoke_dtd.json)、[Aircraft短测试](audit/smoke_fgvc_aircraft.json)。
- [实际配置/指标/模型哈希证据](evidence/)、[GPU启动证据](start-evidence/start-evidence.json)、[启动收据](start-evidence/launch.json)、[准备报告](../../preparation/crossdataset-20260910/README.md)、[研究协议](../../../docs/CROSS_DATASET_VALIDATION.md)。启动收据中的pending状态和PID1576是初始历史快照，以completed-status.json的passed为最终执行状态。
- [耗时记录](runtime.json)统计独立命令的墙钟时间，含解释器、模型/数据加载等开销，不能当作纯推理延迟或FLOPs比较。

## 完整备份和恢复

- 服务器实验目录：`/root/autodl-tmp/textconsistency/runs/crossdataset-study-20260910-v1`。队列全部完成，禁止重复启动或覆盖目录。
- 本机目录：工作区`outputs/textconsistency-crossdataset-study-20260911-55424/`，包含完整压缩包及解压副本。
- 完整压缩包`crossdataset-study-20260911-55424.tar.gz`，112,686,945字节；SHA256：`4a681854c299562d8d4dca771584abf09cf167daf919de89ff2f631187e9e34e`。本机压缩包及全部24个checkpoint逐个校验通过。
- 同一压缩包位于服务器`/root/autodl-tmp/textconsistency/backups/crossdataset-study-20260911-55424.tar.gz`，含18个正式/6个短测试模型、全部原始日志/配置/指标/审计、启动与准备证据及操作源码包，不含数据集及冻结CLIP基础权重。
- [完整备份收据](backup-receipt.json)、[提前完成的DTD子集备份收据](dtd-partial-backup-receipt.json)、[打包脚本快照](execution/backup_completed.py)。GitHub仅保存轻量代码、报告和证据；大模型/数据不入Git。服务器电源仍由用户控制。

恢复入口：[PROJECT_STATUS.md](../../../PROJECT_STATUS.md)。

[最后运行检查](final-runtime-check.json)：最终状态passed，GPU计算进程为空，利用率0%、显存2MiB；代理未开关服务器。
