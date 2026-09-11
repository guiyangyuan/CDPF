# DTD / Aircraft 固定三种子实验：已启动

启动时间：2026-09-11 10:45:27 +08:00。用户开卡并提供新入口后，在`connect.bjb1.seetacloud.com:55424`启动固定队列。当前是运行记录，**尚未宣称正式实验完成或方法有效**。

- 实测RTX5090 32GB，驱动595.71.05，torch2.8.0+cu128；CUDA前后向有限性检查通过。数据/准备目录保留，操作代码和启动脚本SHA与无卡准备逐项一致。
- 队列PID`1576`；启动时新study目录不存在，GPU无其他计算进程。
- 操作代码commit`6ce9c6e2770c6359dcd1f8dfea326e3755aaa79c`，模型代码commit`13b34978eeb8b23532b53abd1066af4719a0f909`。没有修改科学协议或训练模型。
- 固定6次单轮短测试/28次独立重载，两个数据集均通过审计后，再做18次正式训练/84次独立重载。正式seed1/2/3、20轮、16-shot，三变体为本仓库CoOp、原层级+MLP、参数匹配双头平面+MLP。
- 服务器study：`/root/autodl-tmp/textconsistency/runs/crossdataset-study-20260910-v1`；恢复时先读`status.json`及`logs/`，不得重复运行启动脚本或覆盖目录。
- 启动日志：`/root/autodl-tmp/textconsistency/setup-logs/crossdataset-gpu-start-20260911-55424/queue.stdout.log`和`queue.stderr.log`。各阶段日志在study下；阶段审计失败会停止后续。
- [GPU环境证据](start-evidence/start-evidence.json)、[启动收据](start-evidence/launch.json)、[已完成准备](../../preparation/crossdataset-20260910/README.md)、[固定协议](../../../docs/CROSS_DATASET_VALIDATION.md)。

全部正式结果通过后再给出跨数据集结论，并备份原始日志及全部模型；服务器电源由用户控制。
