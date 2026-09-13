# H1 固定 Base 验证实验

2026-09-13：用户开卡并授权开始，队列已启动，正在运行。尚无新损失有效性结论。

- 服务器：bjb1:42527，实例 autodl-container-a03b4fb96c-09b042f2；RTX5090 32GB、90GiB内存。
- 固定代码：efcd3db；计划SHA256：684ae3b6c6649cec536efb048e116443b0d8df136806569c18a20108b9a2dde6。
- 启动：北京时间13:31:27，PID1531；PID只能结合start-record中的进程开始标记和真实状态核对，不能据此直接重跑。
- study：`/root/autodl-tmp/textconsistency/runs/h1-validation-20260913-efcd3db`。
- 启动日志：`/root/autodl-tmp/textconsistency/setup-logs/h1-gpu-start-20260913-efcd3db/runner.log`。
- 固定顺序：DTD seed1 full_kl 20轮anchor → 只有复现通过才跑UCF101/DTD/Aircraft各seed1 weighted_nckd。最多4次训练；不重试、调参或跑Novel。
- 本机运行材料：工作区`work/h1-gpu-20260913-42527/`。

先读study的status.json与logs，不重复启动队列。训练结束后还须完成严格导出/分析、停止条件检查及完整模型和证据备份，随后更新本报告。

[GPU预检](preflight.json) / [启动记录](start-record.json) / [此前无卡准备与固定协议](../../preparation/h1-server-20260913/README.md)
