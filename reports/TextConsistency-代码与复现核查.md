# TextConsistency：代码、历史结果与投稿准备核查

核查日期：2026-09-10。代码目录：/Users/bytedance/Documents/textConsistency。论文：用户提供的 29 页 Elsevier 模板 PDF。

**建议保留 CLIP 研究方向和现有工程，先修正实现与评估，再用小实验决定是否继续投入。当前论文的整套结果还不能视为已验证，也不适合直接按现有表格投稿。**

用户说明：约一半数字来自已经运行的实验；目前不能确定可复现性；学校期刊级别要求较宽，录用即可；原 RTX 4090 24GB 服务器已不可用，希望春节前录用。暂按 2027 年春节前安排，具体学校认可范围仍应与导师确认。

## 1. 本次实际完成了什么

- 阅读当前 CoOp、CoCoOp 的 TextConsistency 主实现、旧变体、消融实现、配置、训练与评估脚本、数据描述模板及 Dassl 的保存/加载逻辑。
- 对照本地 Git 提交 53d0c171eb05048e5db59c816c09551fee152ee7 与当前未提交修改。
- 从 Git 历史复制出 60 份实验日志文本，建立来源索引；其中包括一份含 77 次运行摘要的 CoCoOp 调参日志。只复制到本次工作区，未恢复原目录中的删除项。
- 运行仓库已有的 10 个轻量测试：全部通过。多数是检查代码字符串/配置结构，不能证明模型训练正确。
- 解析 160 个 Python 文件：未发现 Python 语法错误。
- 从原文件提取实际方法执行小型行为验证：三个主/消融加载器遇到缺失 checkpoint 都静默返回；另验证了两种主实现按 Base/Novel 切换预测分支的行为。

**没有重新训练模型，也没有得到新的准确率。** 本次使用的 Python 环境没有 PyTorch；当前本地是 Apple Silicon macOS，主实现直接调用 CUDA，代码要求的 clip/ViT-B-16.pt 也不在该目录。因此本次是代码与历史证据审查，不是 GPU 复现完成。

原代码目录已有大量删除及修改，本次没有改动它们。主要源文件的核查前后 SHA-256 一致。

## 2. 找到了哪些真实运行记录

下表由历史日志里的测试输出重新提取，HM 由提取值计算。这说明数字有对应运行记录，不等于已经复现、协议正确或对应当前实现。

| 数据集 / 设置 | 日志 Base | 日志 Novel | 计算 HM | 与 PDF 的关系 |
| --- | ---: | ---: | ---: | --- |
| Caltech101 / Baseline | 97.676 | 93.777 | 95.69 | 对应 PDF 表 6 |
| Caltech101 / Full | 97.353 | 95.961 | 96.65 | 对应 PDF 表 6 |
| UCF101 / Baseline | 85.264 | 60.681 | 70.90 | PDF 表 7 写的是 84.69 / 56.05 / 67.46 |
| UCF101 / Full | 73.888 | 79.232 | 76.47 | 对应 PDF 表 7 |
| Aircraft / Baseline | 40.096 | 23.095 | 29.31 | PDF 表 8 写的是 40.45 / 22.30 / 28.75 |
| Aircraft / Full | 29.532 | 40.792 | 34.26 | PDF 表 8 写的是 32.77 / 40.43 / 36.20 |

这些历史消融日志是 seed 1、50 epochs；Full 使用 20 views、KD weight 60、KD temperature 2.0，contrastive/diversity weight 都是 0。当前新消融脚本使用 100 epochs，并为各数据集设置不同的 views、KD、分组及正则权重。**新脚本不是旧消融数字的原样复现脚本。**

CoCoOp 日志还包含 Caltech101 的 T3.9 单次实验 97.611 / 95.852 / 96.72，以及 EuroSAT 的 V42 单次实验 84.762 / 82.615 / 83.67，分别与 PDF 主表相近或吻合。这些目前只能记作可追溯的单次调参结果。

证据见[历史日志证据摘录](audit/历史日志证据摘录.md)、[消融数字 CSV](audit/historical-ablation.csv)和[完整日志索引](audit/historical-log-index.json)。没有在这些日志中找到的表格数字应标为“待提供来源”，不能据此断言从未运行过。

## 3. 需要先解决的关键问题

### 3.1 多随机种子标签与实际 seed 不一致

在恢复的 cocoop_tune_all_new.log 中，18 次摘要标成 SEED1、SEED2、SEED3 的运行，其训练与评估打印出的实际 seed 都是 1。

例如 Food101-SEED2：原日志第 1515 行参数 seed: 1，第 1611 行配置 SEED: 1；评估第 2663、2759 行仍为 1；第 2982 行摘要却标为 SEED2。不能把这些摘要当成三个不同随机种子的均值或方差。

当前若干脚本已明确循环 1/2/3，这只能证明现在的脚本有循环，不能追溯证明旧实验实际使用了三个种子。后续应逐次验证实际启动参数、配置和 few-shot 划分，并分别保存。

### 3.2 已提交旧版本与当前版本的训练目标和权重保存方式不同

Git HEAD 里的两种 V2_FG 主实现只注册保存 prompt_learner，没有保存 view_aggregator。旧损失没有 adaptive branch 的分类 CE，KD 又对 teacher logits 使用 detach；因此在 diversity weight 为 0 时，聚合器没有来自这些损失的训练梯度。contrastive 项也不能为它提供梯度，原因见下一节。

历史消融日志显示 checkpoint 保存到 prompt_learner，且没有 loss_adaptive_ce 输出，与这套旧实现相符。不过日志没有绑定运行时 Git commit，不能声称已精确证明每次实验都用了 HEAD 的全部代码。

当前未提交修改已加入 adaptive CE，并将 prompt_learner 与 view_aggregator 一起注册为 textconsistency。这两项修正值得保留，但它们会改变训练与推理，不能沿用旧数字来宣称当前方法有效。

代码位置：[当前注册与损失](</Users/bytedance/Documents/textConsistency/trainers/coop_textconsistency_v2_fg.py:329>)。旧版本关键片段已保存在证据摘录。

### 3.3 当前 contrastive loss 无法训练模型，也不是论文中的图文对比学习

[CoOp 代码](</Users/bytedance/Documents/textConsistency/trainers/coop_textconsistency_v2_fg.py:361>)与[CoCoOp 代码](</Users/bytedance/Documents/textConsistency/trainers/cocoop_textconsistency_v2_fg.py:481>)计算的是冻结 image_encoder 输出之间的相似度差，输入不经过可训练的投影或提示模块。因此该项没有通向可训练参数的梯度。

CoCoOp 常用 batch size 1，此时排除对角线后没有正样本，代码直接令该损失为 0。即便 batch 增大，冻结特征的问题仍然存在。

PDF 第 14 页公式 23–24 描述的是增强图像与自适应文本之间的 InfoNCE，与当前实现不同。最低成本的处理是先禁用这项并删除相应贡献论断；若保留，应明确正负样本定义，让损失真正连接可训练表示，并重新验证。

### 3.4 Novel 提升不能直接归因于“蒸馏后的提示泛化”

当前评估规则是：

| 实现 | Base 预测 | Novel 预测 |
| --- | --- | --- |
| CoOp + TC | prompt/student 分支 | adaptive/teacher 分支 |
| CoCoOp + TC | prompt/student 分支 | 两分支 logits 各占一半 |

见 [CoOp model_inference](</Users/bytedance/Documents/textConsistency/trainers/coop_textconsistency_v2_fg.py:419>)及 [CoCoOp model_inference](</Users/bytedance/Documents/textConsistency/trainers/cocoop_textconsistency_v2_fg.py:540>)。

已知评估类别集合是 base 还是 novel，本身不是读取每张图片的真实标签；问题是论文必须说明这种分支策略，并公平对照。当前 CoOp 的 Novel 根本不使用蒸馏后的 student 输出；在编码器冻结、两分支参数分离、teacher detach 的设置下，KD 不会通过梯度更新 teacher。因此不能用其 Novel 数字直接证明 KD 改善 student 泛化。

建议对同一个 checkpoint 分别报告 student、teacher、固定融合三种方式，并为主表事先固定推理规则。旧 V2 的评估只返回 student，不能和 V2_FG 的 teacher Novel 数字直接解释成单一组件的增益。

### 3.5 找不到 checkpoint 时仍然可以继续评估

[主加载器](</Users/bytedance/Documents/textConsistency/trainers/coop_textconsistency_v2_fg.py:427>)及另外两个主/消融加载器，在指定目录没有权重时直接 continue。当前代码改用了 textconsistency 子目录；旧输出只有 prompt_learner 子目录时，就可能完全没有载入训练后的模块，仍继续计算准确率。

本次实际执行提取出的 load_model 方法：给出空的现有目录并指定 epoch 42，三个加载器均返回 None，没有报错。未加载真实 checkpoint，因为它不存在。

应改为缺失即报错；显式指定的 epoch 不得悄悄替换为其他 epoch；记录载入路径、epoch、哈希与参数缺失列表。旧的仅 prompt 权重不足以恢复训练过的聚合器。

### 3.6 代码并未实现论文所说的四类语义分组

模板是“类名 → 描述字符串数组”。[当前分组](</Users/bytedance/Documents/textConsistency/trainers/coop_textconsistency_v2_fg.py:150>)按数组顺序等长切块，没有读取 function / appearance / scene / part 标签。抽查 UCF101 和 Aircraft 的描述顺序，也不能支持将每块命名为这四种语义。部分数据集使用 5 组。

此外，不能整除时会截掉末尾描述。例如 22 views / 4 groups 实际仅使用前 20 个。模板少于请求数量时则复制第一条：UCF101 每类 50 条，请求 55 条会补 5 条重复文本；EuroSAT 每类 25 条，请求 40 或 42 条会大量重复。这不等于增加同等数量的独立语义视角。

应选择并如实实现一种方案：明确建立语义标签分组；或者把当前方法描述为顺序分块，并加入随机重排/分组对照。FGE 也是作用于全局图像向量的残差 MLP，没有提取局部 patch 或部件特征，宜表述为全局特征适配。

### 3.7 ImageNet 系列实际上没有用上现有多描述模板

两种 V2_FG 主实现的 TEMPLATE_FILES 缺少 ImageNet 及其变体条目，虽然目录里有 imagenet_prompts_full.json。缺失映射会进入默认分支，把同一条类名提示的向量复制 N 次。

当同一类的所有 view 特征完全相同，任意归一化注意力加权仍得到同一个类向量，自适应多视角聚合在该路径上退化。因此现在不能用这条路径支持 ImageNet 上“多视角语义选择”的结论。

补映射后还应核对类名匹配；现有 ImageNet JSON 有 998 个键，不能只根据数量断定是哪两个类缺失或为何合并，必须与实际 1K 类 ID/类名逐个对应。

### 3.8 CoCoOp 消融参数存在隐藏覆盖，Full 消融也同时改变多个因素

CoCoOp 根据 FG_DATASETS 的硬编码选择 flat/hierarchical 结构、分组数和正则权重。即使命令行设置 CONTRASTIVE_WEIGHT、DIVERSITY_WEIGHT 或 N_GROUPS，它仍可能使用内部字典值。见[权重选择](</Users/bytedance/Documents/textConsistency/trainers/cocoop_textconsistency_v2_fg.py:453>)。仅看打印的输入配置不足以证明损失被关闭。

当前新消融中，v2_no_fg 到 full 同时改变 flat→hierarchical、残差 MLP 和正则项；PDF 把 w/o FG 解释为仅移除残差投影。这不是单变量消融。后续需要独立开关，并打印最终生效配置。

## 4. 对上一次论文评审的修正

**熵正则的实现符号是合理的，当前主要是论文公式需要同步。** 代码使用负的组内熵与组间熵之和，正权重最小化它会提高熵；PDF 却写成正熵并说最小化能防止集中。这不能再笼统描述成“代码熵符号写错”。注意：最大化每次注意力的熵，也不能单独证明实例间具有有意义的差异。

另外，attention 中确实有可训练 query/key 投影和温度，论文直接点积公式需要补全；当前代码也明确具有 adaptive CE 和 teacher detach，可据此重写训练流程。

## 5. 哪些材料保留，哪些结果待验证

| 内容 | 建议 |
| --- | --- |
| 数据加载、CLIP/CoOp 基础、描述文本、当前主要模块 | 保留，作为工程基础 |
| 当前 adaptive CE、完整训练模块保存修改 | 保留并补行为验证 |
| 本次找回的旧日志 | 归档为历史实验，记录配置和局限 |
| 现有主表、消融表数字 | 按单元格建立来源；未验证数字不进入正式结论 |
| 三种子均值/标准差的说法 | 根据真实种子重新运行后生成 |
| 图文对比学习、四类语义分组、局部细粒度恢复的贡献 | 按最终真实实现重写或补足验证 |
| 现有可解释性注意力图与效率表 | 使用最终模型重新生成，记录分支与实际生效参数 |

主表和消融表应该由日志程序生成，不手工挑选不同配置下的有利单元格。超参数在训练/验证数据上选择并冻结，再评估 novel test；当前调参记录包含多次 Novel/HM 比较，需要核实最终选择是否使用了测试结果，不能仅凭日志断言已经发生泄漏。

## 6. 面向毕业目标的最小推进顺序

1. **先固定代码和评估协议。** 确定主实现版本、推理分支、checkpoint 格式；修复缺失权重静默跳过，处理无梯度损失、模板映射、描述重复/截断和隐藏参数覆盖。固定依赖及 CLIP 权重来源，保留 split 与描述文件的哈希。
2. **先做小型正确性验证。** 检查每个声称可训练的模块确有梯度；保存后新进程加载，输出一致；确认 1/2/3 种子确实进入配置并产生对应 few-shot 样本；修改应影响结果的有效开关确实生效。
3. **先在一个小数据集跑一轮。** 建议先选已下载的 UCF101 midframes，1 个种子用于发现问题，不把它作为最终统计。比较相同描述/预算/推理规则下的 fixed average、flat、flat+MLP、hierarchical、hierarchical+MLP，并保留 CoOp 基线。按结果决定是否值得扩展。
4. **核心结论成立再扩展。** 对关键对照使用 3 个真实随机种子，并补 Aircraft、DTD 等数据集；最终覆盖范围按论文论断和目标期刊确定。不要只挑有提升的数据集报告。ImageNet 放到代码正确、预算明确之后。
5. **根据结果收缩论文。** 如果收益主要来自多描述或融合，就围绕被实验支持的问题改写。若层级结构没有稳定独立收益，优先调整当前 CLIP 题目，保留工程，不必立刻完全换领域。

原 4090 24GB 的使用经历不能保证修改后所有配置都适配同样显存。新环境先以小样本实测峰值显存和单轮耗时，再估算费用；当前没有足够证据给出可靠的总训练时长或租用价格。

## 7. 暂定投稿节奏

按 2027 年春节前录用的目标，建议 9 月完成代码纠偏与核心试验，10 月完成主实验和写作，争取 10—11 月首投。这是工作安排，不是对审稿速度或录用的预测。目标期刊应先核实学校是否认可、论文范围是否匹配，再依据官方信息确认投稿要求；本次没有选刊或查询审稿周期。

当前最有价值的投入是建立一套能解释、能重复的实验。只有这一步成立，才值得为这篇安排完整的 GPU 预算和投稿计划。

## 附：可复查产物

- [历史日志证据摘录](audit/历史日志证据摘录.md)
- [历史消融数据](audit/historical-ablation.csv)
- [77 次 CoCoOp 运行摘要与实际种子](audit/historical-cocoop-runs.json)
- [模板数量与样例清单](audit/template-inventory.json)
- [实际加载/推理方法行为验证](audit/source-method-probes.json)
- [来源 commit、源码哈希和环境边界](audit/audit-metadata.json)

历史日志原文保存在本次工作区的 work/code-audit/history，索引中包含完整绝对路径与每份文本的 SHA-256。本报告替代先前评审中有关实现尚未核实的推测；未被本次证据解决的论文问题仍保留。
