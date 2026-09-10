# UCF101 结构对照：运行中

2026-09-10 用户开卡并授权执行。本报告当前是启动记录，正式结果尚未完成。

- 最新SSH：`connect.bjb2.seetacloud.com:27946`，密码未保存。GPU为RTX5090 32GB，驱动580.76.05，cgroup内存90GiB；torch2.8.0+cu128，Python3.12.3。
- [固定协议](../../../docs/STRUCTURE_CONTROLS.md)：原层级、参数匹配双头平面、交错、随机17/29，共5变体×3seed，20epochs/16-shot/KD1。
- 源码commit `13b34978eeb8b23532b53abd1066af4719a0f909`；训练源码SHA256 `9bd4a7f248f09c47be17c1bb64921bd9a29d9f3b46e03e8caac9b934b2a95bba`，启动前源码/关键文件/完整CLIP权重全部通过哈希核对。
- 新GPU与之前实例驱动、CPU/内存配置有所不同；本批5变体使用相同新环境，CPU线程统一设为4。新GPU上的同版原结构也重跑，用于与历史结果对照。

当前队列：`/root/autodl-tmp/textconsistency/runs/structure-study-20260910-27946-v2`，PID1553。先执行`smoke/`（5次单轮训练+30次独立重载），全部通过配置/分组/参数/实际checkpoint哈希审计后，自动执行`formal/`（15次训练+90次独立重载）。这些PID只是本次启动记录，操作前核实实际命令和状态。

进度入口为该目录`status.json`；详细日志为`smoke.log`、`formal.log`；显存/利用率/功率采样为`gpu-samples.jsonl`。结束状态`awaiting_analysis_and_backup`表示训练和初步审计完成，仍需逐种子分析与完整模型备份。

第一次启动目录`structure-study-20260910-27946`在运行环境检查阶段失败：启动器取名`queue.py`遮蔽了Python标准库queue，导致PyTorch导入失败。没有执行任何训练。已保留原失败状态/日志，在独立`execution-v2/structure_queue.py`中修复，并验证PyTorch和标准库导入、真实CUDA前后向成功。训练源码没有因此改动。

[启动证据](start-evidence/)包括第一次失败和修复脚本、运行环境及启动时状态快照。该快照不是实时状态；恢复工作应读取服务器当前status.json，避免重复启动。服务器电源由用户控制，本队列不关闭实例。

全部完成后需要：汇总所有seed和三个分支、配对HM变化、原结构与历史结果差异、核对分组/样本/配置与15个实际checkpoint，备份日志/完整训练权重到本机并校验，随后更新本报告和PROJECT_STATUS.md。不能根据Novel结果选择随机分组或删掉不好的seed。
