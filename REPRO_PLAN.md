# 首阶段复现修订执行范围（历史计划）

> 本计划的修订和 UCF101 三种子实验已完成。最新任务、进度与开卡前准备见 [PROJECT_STATUS.md](PROJECT_STATUS.md)。

本目录由用户当前代码复制而来，原目录保持不变。目标是为一次可靠的小规模验证准备实现与运行环境，所有旧数字仅作为历史记录。

1. 修复缺失 checkpoint 静默评估、分支按 base/new 切换、描述重复和尾部丢弃。
2. 主模型共享 mean/flat/hierarchical 聚合，特征适配独立开关；顺序分组不再声称是四类语义分组。
3. 保留 student CE + adaptive CE + stop-gradient KD + 可选负熵；拒绝使用当前无梯度的 contrastive 项。
4. 统一配置/实际参数，记录 seed、数据 split、文本、代码、CLIP 与 checkpoint 的哈希，独立保存 Base/Novel 和各预测分支指标。
5. 先运行本地真实梯度与保存/加载测试，再做 UCF101 数据准备及受限 smoke；正式多种子完整训练需要 GPU 环境。
6. 生成单数据集 pilot 和扩大验证脚本，默认不覆盖历史输出，不从测试集选超参数。

方法修订不代表已经得到性能提升；论文实验表需由新日志生成。
