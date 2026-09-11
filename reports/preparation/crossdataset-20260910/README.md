# DTD / Aircraft 固定三种子验证：无卡准备完成

2026-09-11（北京时间）：本机及服务器CPU准备已完成，**本批GPU短测试和正式训练均未启动**。用户已无卡启动`connect.bjb2.seetacloud.com:27946`；现有环境和数据可用，无需重新下载到服务器。服务器电源仍由用户控制。当前有效操作版本为`6ce9c6e`，证据位于`final-v2/`；[初版f20795c记录](INITIAL_PREPARATION.md)及其日志保留，开卡时使用下方新版入口。

## 已验证的状态

- 服务器Python3.12.3、torch2.8.0+cu128、torchvision0.23.0+cu128可用，当前0.5CPU/2GiB内存、CUDA=false。原训练源码及10个固定依赖文件SHA与本机一致；旧模型和实验保留。见[环境检查](server-runtime.json)、[接入记录](connection-check.json)。
- 新操作代码已独立部署，服务器**59项测试全部通过**，详见[测试日志](final-v2/server-tests.log)。本机同一实现的14项新增测试及45项原测试也已通过，见[实现记录](review/final-fix-report.md)。
- 完整CPU准备通过；DTD和Aircraft的实际loader逐seed、逐Base/Novel的train/val/test样本哈希、数量、类别与本机一致，见[比对收据](final-v2/verification.json)和[完整准备证据](final-v2/prepared-v2/preparation.json)。
- 无GPU拒绝训练验证通过：显式隐藏CUDA后运行execute，返回预期错误；未创建study目录。见[日志](final-v2/no-cuda-guard.log)。该日志的非零退出是预期防护测试，不是训练失败。
- [ready.json](final-v2/ready.json)为`cpu_preparation_complete`；[driver-status.json](final-v2/driver-status.json)为`completed / awaiting_user_gpu`。PID4151仅为已经结束的准备进程，不能当作训练PID。

## 数据和代码边界

本机保留两个原ZIP并重新校验完整SHA，解压到工作区`work/crossdataset-20260910/data/`，见[数据收据](local-data-receipt.json)。DTD固定Zhou划分，24个Base类、23个Novel类，每seed384张训练图，Base/Novel测试864/828张；Aircraft固定划分，50/50类，每seed800张训练图，测试1666/1667张。图像存在、划分无交集、描述覆盖、每类16-shot、seed1/2/3的不同采样均通过。见[DTD预检](local-dtd-preflight.json)、[Aircraft预检](local-fgvc_aircraft-preflight.json)。

允许固定seed的`split_fewshot`派生缓存生成/复用。原图、划分元数据和描述文件不修改。图像清单核验路径、大小和mtime，**不是逐图内容SHA**；两个ZIP、完整CLIP权重和固定代码/配置另做SHA校验。本机与服务器解压时间不同，图像清单不要求跨机相同；其余数据/实际loader证据逐项一致。

- 模型代码：`13b34978eeb8b23532b53abd1066af4719a0f909`，模型源码SHA256：`9bd4a7f248f09c47be17c1bb64921bd9a29d9f3b46e03e8caac9b934b2a95bba`；使用原`releases/structure-20260910-v1`，见[固定文件](expected-source-files.json)。
- 操作代码：`6ce9c6e2770c6359dcd1f8dfea326e3755aaa79c`；独立release为`/root/autodl-tmp/textconsistency/releases/crossdataset-ops-20260911-6ce9c6e`。压缩包和实际操作脚本SHA逐项验证，见[部署收据](final-v2/deployment.json)。
- CLIP完整权重SHA：`5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f`，见[本机收据](local-clip-receipt.json)。旧源码和新操作release均指向服务器同一验证过的基础权重。
- 初审和复审后，[最终审查](review/final-review.md)发现同一seed所有变体一起漂移超参数可能漏报，已补充由原`train.setup_cfg`生成并冻结全部136条命令的完整有效配置；训练/评估逐项对照并独立核对seed。[最终复审](review/final-rereview.md)确认问题已解决，无阻塞项。原模型源码和固定研究协议不变。

## 开卡后执行

固定协议见[研究协议](../../../docs/CROSS_DATASET_VALIDATION.md)，完整命令见[protocol.json](final-v2/prepared-v2/protocol.json)。本仓库CoOp、原层级+MLP、参数匹配双头平面+MLP，两个数据集、seed1/2/3、20轮、16-shot，正式**18次训练/84次独立重载**。先做6次单轮短测试/28次重载，两数据集均通过审计后再进入正式阶段。短测试不纳入统计；不增加分组、调参或种子。

用户开卡后重新核实当前SSH及同一实例，执行：

```bash
bash /root/autodl-tmp/textconsistency/setup-logs/crossdataset-prep-20260911-v2/run_gpu_study.sh
```

[启动脚本快照](final-v2/execution/run_gpu_study.sh)已保存到服务器，仅做`bash -n`检查。正式执行前会重新检查源码/操作脚本/CLIP/数据证据、CUDA及GPU空闲状况。输出固定为`/root/autodl-tmp/textconsistency/runs/crossdataset-study-20260910-v1`，当前不存在；若后续已经存在，先读状态和日志，不能覆盖或直接重跑。准备目录为同一项目下`setup-logs/crossdataset-prep-20260911-v2/prepared-v2`。

每阶段日志和审计通过才进入下一阶段；失败保留证据并停止。全部正式结果齐全后分别报告两个数据集的原方法−基线、原层级−参数匹配平面逐seed差及均值/标准差，并备份完整模型和原始日志到本机。GitHub保存代码、协议和轻量证据，不保存数据/模型大文件或凭据。

恢复入口：[PROJECT_STATUS.md](../../../PROJECT_STATUS.md)。本次CPU执行脚本快照：[prepare_remote.py](final-v2/execution/prepare_remote.py)。准备完成不代表新方法效果已经获得跨数据集验证。
