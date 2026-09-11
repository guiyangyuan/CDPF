# 描述融合与KD贡献验证：准备记录

2026-09-11。用户授权在前一轮评估后开始下一步；本批服务器bjb1:55424实际为无卡模式，cgroup内存上限2,147,483,648字节。本文件当前记录已验证输入，不代表新训练或评估已完成。

- [预先固定的实验计划](../../../docs/plans/2026-09-11-attribution.md)：复用CoOp做描述均值融合，补DTD/Aircraft的KD0，核对两次KD1重现，再做明确标注的ADK公式受控适配。没有Novel调参或种子选择。接近方法的作者发布证据和公式适配边界见[核对记录](closest-methods.md)。
- [模型输入收据](input-receipt.json)：九个CoOp基线、六个DTD/Aircraft历史KD1、三个UCF101完整方法参考均已核对实际checkpoint SHA和sidecar。
- [UCF101恢复收据](ucf101-baselines-receipt.json)：三个历史基线从本机完整备份恢复到服务器全新目录`reused-inputs/attribution-20260911/ucf101/baseline`，原checkpoint没有修改。恢复压缩包1,442,120字节，SHA256 `a18262644ab7970d41e8ce22f2b2ccd35e11ca8f9dedad1305dd218664b0f9f5`。权重不进入Git。
- [真实数据预检](input-preflight.json)：UCF101、DTD、Aircraft固定三种子loader、描述覆盖和样本检查通过；记录原图路径/大小/mtime清单哈希，未逐图做内容SHA。
- [跨机器模型一致性](local-backup-correspondence.json)：服务器全部18个参考模型的实际SHA与本机已验证完整备份的模型字节一致。
- [历史划分一致性](historical-sample-check.json)：九个基线的18组Base/Novel测试哈希与当前真实loader一致。
- 本地新worktree基线59项测试通过。首次测试调用少了`-t .`导致相对导入错误，修正调用后通过，未改测试或源码来消除错误。

GPU开关由用户操作。正式新输出计划为`/root/autodl-tmp/textconsistency/runs/attribution-study-20260911-v1`，当前未创建；准备目录为`setup-logs/attribution-prep-20260911-v1`。融合/ADK评估入口已实现，21项新测试与59项既有测试通过，旧UCF seed1的Base/Novel元数据实际验证通过；种子域限制修复后25项focused通过，独立任务审查和复审已通过，见[审查记录](review/task-1-rereview.md)。KD固定队列正在实现，仍待审查，尚未形成可执行就绪结论。所有旧批次结果和备份保持可用。
