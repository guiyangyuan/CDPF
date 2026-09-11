# DTD / Aircraft 固定三种子实验：正式阶段运行中

启动时间：2026-09-11 10:45:27 +08:00。用户开卡并提供新入口后，在`connect.bjb1.seetacloud.com:55424`启动固定队列。两套短测试审计已通过，DTD正式阶段已完成并通过审计，Aircraft正式阶段已启动。当前是运行记录，**尚未宣称正式实验完成或方法有效**。

- 实测RTX5090 32GB，驱动595.71.05，torch2.8.0+cu128；CUDA前后向有限性检查通过。数据/准备目录保留，操作代码和启动脚本SHA与无卡准备逐项一致。
- 队列PID`1576`；启动时新study目录不存在，GPU无其他计算进程。
- 操作代码commit`6ce9c6e2770c6359dcd1f8dfea326e3755aaa79c`，模型代码commit`13b34978eeb8b23532b53abd1066af4719a0f909`。没有修改科学协议或训练模型。
- 固定6次单轮短测试/28次独立重载，两个数据集均通过审计后，再做18次正式训练/84次独立重载。正式seed1/2/3、20轮、16-shot，三变体为本仓库CoOp、原层级+MLP、参数匹配双头平面+MLP。
- 服务器study：`/root/autodl-tmp/textconsistency/runs/crossdataset-study-20260910-v1`；恢复时先读`status.json`及`logs/`，不得重复运行启动脚本或覆盖目录。
- 启动日志：`/root/autodl-tmp/textconsistency/setup-logs/crossdataset-gpu-start-20260911-55424/queue.stdout.log`和`queue.stderr.log`。各阶段日志在study下；阶段审计失败会停止后续。
- [GPU环境证据](start-evidence/start-evidence.json)、[启动收据](start-evidence/launch.json)、[已完成准备](../../preparation/crossdataset-20260910/README.md)、[固定协议](../../../docs/CROSS_DATASET_VALIDATION.md)。

全部正式结果通过后再给出跨数据集结论，并备份原始日志及全部模型；服务器电源由用户控制。

短测试证据：[DTD](audit/smoke_dtd.json)、[Aircraft](audit/smoke_fgvc_aircraft.json)。其单轮分数不纳入正式统计。

## DTD阶段完整结果（Aircraft仍在运行）

DTD9次正式训练/42次独立重载已通过[完整审计](audit/formal_dtd.json)。主分支baseline=student，原层级和平面=ensemble，三个seed全部保留。

| 方法 | Base均值 | Novel均值 | 均值的HM |
|---|---:|---:|---:|
| 本仓库CoOp基线 | 77.78 | 54.07 | 63.79 |
| 原层级+MLP | 75.77 | 64.21 | 69.51 |
| 参数匹配双头平面+MLP | 74.50 | 65.58 | 69.76 |

原方法−基线的逐seed HM差为+8.1390、+6.3743、+2.4963，平均+5.6699，样本SD2.8865，3/3正向。原层级−平面的逐seed HM差为+2.0347、−0.1806、−2.7323，平均−0.2927，样本SD2.3855，只有1/3正向。DTD支持完整方法相对本仓库基线的收益，尚不支持层级结构独立优于参数匹配平面；不依据这些结果更换分组、超参或删seed。等待Aircraft完整结果后再作跨数据集判断。
