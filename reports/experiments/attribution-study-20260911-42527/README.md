# 描述融合与跨数据集KD贡献验证：运行中

2026-09-11。用户开卡入口 bjb1:42527，实际 RTX5090 32GB、驱动580.76.05、torch2.8.0+cu128、90GB容器内存。CUDA前后向检查通过，启动前没有其他计算进程。

固定操作代码43e537c，冻结准备和启动器SHA已核对。队列07:54:22 UTC启动，PID1479，后台日志位于 `/root/autodl-tmp/textconsistency/setup-logs/attribution-gpu-20260911-42527/launcher.log`。实验目录为 `/root/autodl-tmp/textconsistency/runs/attribution-study-20260911-v1`。

当前为运行中，不能据短测试判断方法有效，也不能把不完整分支作为正式结果。先两数据集短测试，再两次KD1历史精确重现，之后三数据集描述融合和DTD/Aircraft的KD0/1配对。全部种子和分支均报告，不根据Novel调整实验。

恢复时先读取服务器study/status.json和本目录启动收据；已有study禁止重复启动或覆盖。完成后独立审计、完整模型备份、更新报告与GitHub。服务器电源由用户操作。
