# 独立代码审查与处理

2026-09-10，只读审查，范围为 f30395c 之后的结构对照实现、测试和固定协议。审查期间首次45项测试通过，修复后总计46项。

- Critical：无。
- Important：结构汇总原先仅核对描述类别/数量，没有验证测试图片哈希，也未强制 test split。审查者通过修改 flat_dual_adapter 的 test_samples_sha256 复现汇总仍接受的问题。
- 修复：结构批次 Base/Novel 分别要求64位十六进制测试哈希，跨 seed/variant/branch 一致；Base 还与训练 manifest 一致；split 必须为 test。旧协议行为保留。
- 回归证据：先增加“不同测试图片/缺失哈希/val split”三个反例，在旧实现均未抛错而测试失败；修复后全部通过。独立复审运行新增篡改测试、完整结构汇总及两项旧协议兼容测试，共4项通过。
- Minor（保留）：`--structure-study` 需要显式 `--seeds 1 2 3`。完整命令已写入协议与服务器计划；单独省略 seeds 会拒绝执行，不会悄悄跑单种子。
- 复审结论：可完成无 GPU 阶段准备，无剩余 Critical/Important。新版本的完整预训练模型 GPU 检查、冒烟及正式结果仍待开卡。

同时检查了四投影/两温度参与计算、参数预算、token 去重后置换有效描述、padding/mask保持、独立随机生成器、checkpoint 分组签名隔离；未发现其他阻塞问题。
