# 接近方法及本轮受控适配边界

核对日期2026-09-11。结论：当前未取得AAPE或ADK的可执行作者实现；本轮新增的是ADK公开公式在固定CoOp上的受控适配，不是作者代码复现或论文表格复现。

## 作者发布证据

- AAPE：NeurIPS2024论文清单对开放代码回答No，Apple研究页和论文入口未提供可用实现。[官方论文](https://proceedings.neurips.cc/paper_files/paper/2024/file/93e67adb632d636bb2bcedf4eef1a6fe-Paper-Conference.pdf)、[Apple页面](https://machinelearning.apple.com/research/aggregate-and-adapt)。未找到可冻结的作者代码commit/权重/原始描述文件，不能擅自补出未说明的结构参数再称官方复现。
- ADK：[第一作者仓库](https://github.com/leesb7426/ADK/tree/7d6b77cf17db16ae2b201c5472048cef9fe7eaa7)的唯一commit为`7d6b77cf17db16ae2b201c5472048cef9fe7eaa7`，只有一行README，无实现、权重、描述或license文件。[论文](https://arxiv.org/abs/2512.17313v1)公布方法公式。
- 可用相邻资源：[2SFS官方MIT代码](https://github.com/FarinaMatteo/rethinking_fewshot_vlms/tree/64ac143c7d22803bfeeaadfe42f570220ab29b06)、[CuPL作者描述](https://github.com/sarahpratt/CuPL/tree/719619c829792ff8f37298266e941b7f077c7f5c)。它们分别是协议背景和文本先验来源，不等同于AAPE/ADK作者完整实现；本轮不额外增加其训练或更换已有描述。

## 本轮确定的公式

按[ADK第3.4节公式2–13](https://arxiv.org/html/2512.17313v1#S3.SS4)，将均值描述和图像相关描述分别形成类别概率，再与CoOp类别概率融合。描述、CLIP和原CoOp模型固定，全部评估候选类别与当前Base/Novel一致。

令`s=exp(CLIP.logit_scale)`，图像及描述单位化；有效描述注意力为`softmax(s*cos(image,description))`，按每类描述维归一化。均值/加权描述再单位化以计算余弦。所有类别概率使用同一个`s`。

- `simple_fusion`：原CoOp与描述均值的logits各占0.5。
- `mean_probability_fusion`：两者的类别概率各占0.5，用于隔离概率融合方式的影响。
- `adk_probability_fusion`：CoOp概率0.5，描述均值概率0.25，图像相关描述概率0.25；与论文公式12/13的argmax等价。
- 同时报告student、description_mean、description_instance。六个分支全部保留，不能看分数后修改主比较。

论文公式除以tau，但文字称其为logit scale；本轮固定解释为标准CLIP的温度`tau=1/s`，记录实际s，不调温度。作者仓库为空，不能声称已用作者实现消除这一措辞歧义。

这是对公式的CoOp适配：论文的手工提示分支替换为已训练CoOp，使用本项目同一批固定描述和20epoch基线；没有复现其GPT-4描述、adapter/LoRA/2SFS两阶段训练。此处CLIP和描述全部冻结，描述CE对CoOp上下文梯度为零，因此直接复用原CoOp模型有数学依据；这不适用于论文中会更新编码器的设置。

原始研究工作报告另存本机`work/attribution-20260911/priorwork-research.md`，SHA256 `c17a2c66febb3481e5b29d0b7f4bf0a875d0bc8f15515a9a968065cebee8025a`。GPU未执行，本文不包含新分数或优劣判断。
