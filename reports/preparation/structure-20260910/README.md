# 结构对照无卡准备记录

2026-09-10，**无卡准备已完成，等待用户开卡**；尚未产生新正式实验结果。

- 实验预先固定为 [5变体×3seed](../../../docs/STRUCTURE_CONTROLS.md)，15次训练/90次重载评估。
- 本地46项CPU测试通过，包括小尺寸真实CLIP架构前后向和checkpoint重载；[测试记录](local-tests.txt)。
- 无卡服务器46项CPU测试通过；[服务器测试记录](server/server-tests.txt)。CUDA=False，torch2.8.0+cu128，Python3.12.3，cgroup内存2GiB。
- UCF101/DTD/Aircraft的固定划分、全部图片存在、描述覆盖、每类16-shot和三seed检查均通过，预检JSON与上一轮逐项完全相同；CLIP权重SHA256再次验证通过。
- [服务器准备收据](server/ready.json) / [本机对照收据](verification-receipt.json)：源码哈希相同、9个关键文件SHA256一致；源代码commit为`13b34978eeb8b23532b53abd1066af4719a0f909`。
- [正式计划](server/formal-plan.json)含105条命令，[单轮GPU冒烟计划](server/smoke-plan.json)含35条命令；均仅生成，未执行。[CUDA保护检查](server/cuda-guard.txt)按预期退出1，且未创建训练输出目录。
- [旧版数值回归](default-regression.json)：原5种聚合配置×3seed，初始化、输出、梯度、随机数状态一致。512维实测聚合器787970、含prompt790018；平面双头/层级匹配初始化一致。
- 独立审查发现的测试集哈希缺口已复现和修复；汇总强制检查结构批次test split及Base/Novel分别跨seed/variant/branch的测试样本哈希。
- [独立审查与修复](code-review.md)：无剩余Critical/Important；复审4项相关检查通过。
- 原始顺序、数据、旧release和旧结果保留；新checkpoint为format2，旧checkpoint需继续用旧release。
- GPU完整模型前后向/冒烟、15次正式训练、90次重载评估均待开卡后执行。本报告不提供新增准确率结论。

新release：`/root/autodl-tmp/textconsistency/releases/structure-20260910-v1`。激活命令：

```bash
source /root/autodl-tmp/textconsistency/activate-structure-v1.sh
```

服务器证据：`/root/autodl-tmp/textconsistency/setup-logs/structure-prep-20260910/`；本机完整证据：工作区 `work/structure-prep-20260910/`。服务器旧激活脚本、旧源码和旧实验目录保留。新增激活脚本将CPU线程限制为1，适合无卡准备；GPU阶段若调整CPU线程数，必须统一全部变体并记录。

[一次性部署/验证脚本](prepare_server.py)是本次操作快照，硬编码了已核查源包哈希和新目录，不应在同一目录再次执行。
