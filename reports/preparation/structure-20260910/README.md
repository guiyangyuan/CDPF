# 结构对照无卡准备记录

2026-09-10，本机实现与测试完成，服务器部署验证进行中；尚未产生新正式实验结果。

- 实验预先固定为 [5变体×3seed](../../../docs/STRUCTURE_CONTROLS.md)，15次训练/90次重载评估。
- 本地46项CPU测试通过，包括小尺寸真实CLIP架构前后向和checkpoint重载；[测试记录](local-tests.txt)。
- [旧版数值回归](default-regression.json)：原5种聚合配置×3seed，初始化、输出、梯度、随机数状态一致。512维实测聚合器787970、含prompt790018；平面双头/层级匹配初始化一致。
- 独立审查发现的测试集哈希缺口已复现和修复；汇总强制检查结构批次test split及Base/Novel分别跨seed/variant/branch的测试样本哈希。
- 原始顺序、数据、旧release和旧结果保留；新checkpoint为format2，旧checkpoint需继续用旧release。
- GPU完整模型前后向/冒烟、15次正式训练、90次重载评估均待开卡后执行。本报告不提供新增准确率结论。
