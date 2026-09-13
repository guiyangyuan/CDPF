# Aircraft A/B/C：GPU 实验进行中

2026-09-14，用户开卡授权后启动。RTX 4090 24GB，实例 `autodl-container-a5904787c3-600682b1`，入口 westc:26022，内存 90GiB。

固定源码 `1a3830754dbffa87f06348e842e68a07fb4054c7`，计划 SHA `8f2c65b4130021d62b66e0c33920c1110b0e8af293c58d69167bdd2928eebc5c`。实际启动时间 UTC 2026-09-13T16:06:09（北京时间 00:06:09），执行器 PID 1269。启动前 CUDA 运算、实例、操作代码、计划及未使用输出检查通过；GPU 无其他计算进程。

运行目录 `/root/autodl-tmp/textconsistency/runs/visual-adapter-pilot-westc-20260913-1a38307`。只监控现有 status.json，**不得再次执行启动命令**。一次性计划含 A/B/C 各20轮训练与各一次 S/D/F 导出，整队列3600秒，不重试/不扩量。当前还没有完整验证结果。

见[固定协议](../../../docs/plans/2026-09-13-visual-adapter-pilot.md)、[准备与输入核验](../../preparation/visual-adapter-westc-20260913/README.md)、[实际启动收据](launch.json)。完成后核对全部日志、模型和 logits，下载逐文件核验，再更新结果与下一步。电源由用户操作。
