# DTD / Aircraft：固定配置验证协议

本轮目标是检验原方法的整体收益和层级结构的独立收益是否能跨数据集重复。协议在本轮任何GPU训练前固定；不根据Novel测试成绩改变模型、超参数、分组、种子或最终epoch。

## 对照和预算

| 变体 | 展示名称 | 参数量 | 有效KD | 主分支 |
|---|---|---:|---:|---|
| baseline | 本仓库CoOp基线 | 2,048 | 0 | student |
| hierarchical_adapter | 原连续分组层级 + MLP | 790,018 | 1 | ensemble |
| flat_dual_adapter | 参数匹配双头平面 + MLP | 790,018 | 1 | ensemble |

三种子固定为1、2、3，使用ViT-B/16、FP32、16-shot、20epochs、batch16；prompt长度4、最多20条描述、层级4组、KD温度2、teacher CE权重1、diversity和contrastive为0、融合alpha=0.5。优化器、学习率、增广和输入处理沿用已有`configs/repro/dtd.yaml`及`fgvc_aircraft.yaml`，不做本轮搜索。基础CLIP权重、模型代码、配置和描述文件均记录SHA256。

基线使用已在UCF101核验过数值等价性的本仓库CoOp实现；本轮并未重新执行官方上游的DTD/Aircraft代码，因此不得写成“官方结果复现”。普通结构对照中仅层级/双头平面聚合方式不同，其余非运行路径设置必须一致。参数量相同不等于FLOPs、优化动态相同。

## 固定执行顺序

| 阶段 | 数据集 | 种子 / 轮数 | 训练数 | 独立重载评估数 |
|---|---|---|---:|---:|
| smoke_dtd | DTD | seed1 / 1epoch | 3 | 14 |
| smoke_fgvc_aircraft | Aircraft | seed1 / 1epoch | 3 | 14 |
| formal_dtd | DTD | seed1/2/3 / 20epochs | 9 | 42 |
| formal_fgvc_aircraft | Aircraft | seed1/2/3 / 20epochs | 9 | 42 |

两套短测试均审计通过后才能开始正式训练；短测试不进入结果表。正式共18个模型、84次独立重载，每个模型只使用第20轮checkpoint。基线报告Base/Novel student；两个完整结构同时保存Base/Novel的student、teacher、ensemble，主比较始终使用ensemble。

## 数据与审计

- DTD使用固定Zhou划分，24个Base类、23个Novel类，每seed训练384张；Aircraft使用固定官方train/val/test划分，50个Base类、50个Novel类，每seed训练800张。训练和评估类按现有数据加载器划分，禁止更换划分。
- CPU准备阶段验证压缩包/分割元数据、实际图像存在、训练/验证/测试不重叠、描述覆盖、每类16-shot以及三个独立采样seed。固定各seed及Base/Novel分别对应的train/val/test样本哈希、实际数量和类别列表；Novel采样不能直接与Base训练记录比较。允许数据加载器生成/复用固定seed的split_fewshot派生缓存，原图、划分元数据和描述文件保持不变。
- 复用`run_pilot.build_commands`生成原训练/重载命令，由本轮独立控制器记录协议、执行及审计：普通模式的`summarize.collect`原本没有强制完整测试哈希和参数匹配核验，不能只凭它成功就认定全部受控。
- 用原train.setup_cfg按真实默认值、扩展、YAML、CLI及override顺序生成每条命令的完整有效配置，准备时冻结，训练/评估时精确比对并独立核对seed；不能仅依赖同seed变体相互一致，以防共享超参数漂移。
- 核对每个实际checkpoint的SHA256、epoch、保存来源、结构签名、实际参数总数和有限性；核对每次独立评估加载同一指定模型，测试样本与准备记录相符，correct/total与accuracy一致。
- 分数据集输出全部种子及分支、Base/Novel均值和标准差、均值的HM、每seed HM统计。主配对分别为原层级−参数匹配平面、原方法−基线，不能把不同数据集或分支混到一个最佳分数中。

## 判断规则

如果原方法相对基线在两个数据集都有可重复收益，继续补接近方法的公平比较；如果层级相对参数匹配平面收益仍较小、负向或种子不一致，应精简模块或降低层级贡献主张。三种子方向一致也不等于统计显著，三个数据集不是期刊通用的录用门槛。

本轮不运行随机17/29新分组、不扩超参数搜索、不启动ImageNet，也不修改论文。下一次研究决策基于本轮完整结果，不能只保留好看的seed。

## 版本、执行和恢复

- 模型代码commit：`13b34978eeb8b23532b53abd1066af4719a0f909`；模型源码SHA256：`9bd4a7f248f09c47be17c1bb64921bd9a29d9f3b46e03e8caac9b934b2a95bba`。
- 独立操作脚本：`scripts/repro/cross_dataset_study.py`与`cross_dataset_audit.py`；其版本和哈希另记录，不改变旧模型源码。
- 准备、执行、审计状态以[本轮准备报告](../reports/preparation/crossdataset-20260910/README.md)及[项目进度](../PROJECT_STATUS.md)为准。计划文件本身不证明已运行。
- 服务器旧源码release及旧runs只读保留。新输出目录固定为`/root/autodl-tmp/textconsistency/runs/crossdataset-study-20260910-v1`；必须确认不存在后启动，失败保留证据，不自动覆盖或重试。
- 完成后完整模型与原始日志备份到本机，GitHub保存源码/固定协议/配置/汇总和审计。服务器电源仍由用户控制。
