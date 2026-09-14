# 文本侧残差与近邻中心化：真实GPU结果

2026-09-14。**未通过预先固定的小实验继续门槛，停止此配置扩量。** 此处是Aircraft seed1、200张Base验证图上的开发结果，每多对一张等于0.5个百分点；不是Novel/HM、统计显著性或发表结论。

## 改了什么

新增CoOp_TextResidual。学生继续按原CoOp CE训练；共享512→64→512文本映射产生delta，R=CLIP尺度×原图像特征×delta转置，最终Z=S+R，纠错CE使用stopgrad(S)。候选contrast在映射前减去最近5个其他类别名文本的均值，目的是检验相近类别参照是否有助于描述纠错。description对照只移除该中心化；classname用单条模板类别名。三组均67,584可训练参数，无按类别学习的参数表，无KD，无权重搜索。[完整事前协议](../../../docs/TEXT_RESIDUAL_PILOT.md)在训练前已推送。

旧图像侧描述纠错32.5%未通过上一轮门槛，此结果和停止决定保留。本轮同时改变了残差作用的位置与输入处理，不能将与旧模型的差异全部归因于中心化；中心化效应应比较本轮contrast和description。

## 固定训练与结果

代码[d5152e1](https://github.com/guiyangyuan/textconsistency_v2_tune/commit/d5152e1920ff8255e8960a1afacaea52f0667eab)，RTX4090，16-shot（800训练/50Base类），ViT-B/16 FP32，seed1，batch16，SGD lr0.002，20轮。三组均重新训练；恰好3train+3独立最终checkpoint重载导出，六阶段通过，总用时15.63分钟。只用第20轮，未重试、未搜参、未评估新Novel/test。

| 模型 | 学生S % | 固定描述M % | 原简单融合F % | 文本纠错Z % |
|---|---:|---:|---:|---:|
| classname | 32.0 | 24.0 | 30.5 | 32.0 |
| description | 32.0 | 24.0 | 30.5 | 32.0 |
| contrast | 32.0 | 24.0 | 30.5 | 32.5 |

S及F均由同一模型导出，三组学生逐图logits和最终提示参数完全一致（独立核验最大差均为0）。因此S是受保护学生的配对基线，没有额外增加第四次学生训练。F=.5S+.5M，只作固定参照。

| 候选contrast.Z减去参照 | 差值pp | 改对图数 | 改错图数 | 至少+1pp |
|---|---:|---:|---:|---|
| contrast.Z - classname.S | +0.5 | 1 | 0 | 未通过 |
| contrast.Z - classname.F | +2.0 | 16 | 12 | 通过 |
| contrast.Z - classname.Z | +0.5 | 1 | 0 | 未通过 |
| contrast.Z - description.Z | +0.5 | 1 | 0 | 未通过 |

事前要求四个对照全部至少+1pp，实际continuation_signal=false。全部分数和差值见[analysis.json](analysis.json)，逐图原始分数在三个子目录中。不存在验证后调权或改门槛。

相对各自学生，预测标签改变张数：{"classname": 1, "description": 2, "contrast": 5}。标签改变并不等同改对，必须结合上表的改对/改错看。

## 验证与完整保存

实现前后只运行两项针对性短CPU检查（测试体0.962秒）：两步训练、梯度隔离、骨干冻结、零残差起点、近邻排除本类、保存重载、类别文本重建、输出公式；服务器未重复。备份核验仅检查实际产物的完整性和指标，不是额外训练测试。

三个完整checkpoint（包括可训练模块、optimizer/scheduler）、全部日志、逐图NPZ、固定配置、源代码压缩包均已下载。42个文件完成大小/SHA核对，指标及四项配对差值独立重算，checkpoint确认20轮、正确参数及非零文本残差权重。详见[verification.json](verification.json)及[备份清单](backup-manifest.json)。

- 本机备份：工作区outputs/textconsistency-text-residual-20260914（含results.tar.gz和extracted）。
- 完整归档SHA256：`cafd916ae8dc4a2b1ea4d519dd651911ea6fb3a62d2562414272fce36d4c84bd`。
- 服务器run：`/root/autodl-tmp/textconsistency/runs/text-residual-20260914-d5152e1`。PID1239为已结束历史进程，不重新启动。
- GitHub保存代码、协议、原始小日志/分数、报告与校验收据；数据集和CLIP/训练权重留在完整备份，不塞入Git历史。本文不包含登录凭证。

## 判断范围与下一步

未通过预先固定的小实验继续门槛，停止此配置扩量。 单种子200张、且该验证集已多次参与开发，不能据此认定方法达到发表条件。仅在固定方案有足够信号并获得后续工作安排时，才补多种子、Base/Novel/HM和未用于开发的新数据集；不重新命名已失败模块继续无限搜索。

近邻描述判别已有[DisCLIP（AAAI2025，作者实现）](https://github.com/CompVis/DisCLIP)等先例；残差适配已有[CLIP-Adapter](https://arxiv.org/abs/2110.04544)。本轮是候选改进的实现及受控实验，不能把文本残差或近邻思想本身直接认定为原创。论文/Overleaf未修改。

## 实例已关闭

完整备份核验通过后，通过AutoDL网页关闭同一实例，页面最终确认“已关机”。[电源记录](power-control.json)保存本次实际观察；实例未删除。期间SSH连接中断，重连后确认六阶段完成；上面的15.63分钟是训练/导出队列用时，并非整个租用或计费用时。没有后台实验等待执行。
