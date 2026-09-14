# 文本侧残差：固定小实验（2026-09-14）

**执行已完成：** d5152e1，3train+3export全部通过，15.63分钟；候选32.5%，学生/两种文本对照32.0%，F30.5%。未通过继续门槛，已备份并关机。见[最终报告](../reports/experiments/text-residual-20260914/README.md)。下文保留训练前固定的协议。

用户授权直接完成改进、训练、报告，并使用 AutoDL 现有实例开卡/备份后关机。本轮仅一次固定三组开发实验，最多1 GPU小时，不自动追加种子、数据集或超参数搜索。论文暂不改动。

## 假设与计算

上轮图像侧描述纠错Z仅32.5%，未达到继续门槛。对已归档200张Base验证图的端点诊断显示：描述残差只改变4张最终预测；原学生136个错误中，真实类别相对原预测类别的分差改善63个、变差73个。因此不能仅凭残差小就假定放大权重会有效。

本轮检验新的受限假设：通过共享的文本侧映射产生分类修正，并从描述中减去近邻类别的文本均值，能否学到比原始描述/类别名更有用的区分信息。减均值只是可检验的近似，不能称为已经提取了真实的判别属性。

```
f = normalize(frozen_CLIP_image(x))
S = clip_scale * f @ learned_CoOp_text.T
N(c) = cosine-nearest 5 OTHER classname texts (stable ties; only candidate texts)
u_cj = normalized_description_cj - mean(classname_text[N(c)])
h_c = masked_mean_j(relu(W_down u_cj))
delta_c = 0.2 * W_up h_c
R = clip_scale * f @ delta.T
Z = S + R
loss = CE(S,y) + CE(stopgrad(S)+R,y)
```

512→64→512，无bias，down默认Kaiming、up为零，初始化隔离训练RNG。delta不做归一化，系数固定；R初始精确为零。CoOp只接收原CE，CLIP及logit_scale冻结。共享映射无每类可训练表，Novel时从新候选文本重建描述和近邻；不使用Novel图像或标签训练。真实Novel评估不在本批范围。

## 三组固定对照

| 名称 | 残差映射输入 | 对照目的 |
|---|---|---|
| classname | 每类单条官方模板类别名文本 | 排除只有普通文本适配带来的收益 |
| description | 每类最多20条原始描述；非线性之后按mask均值 | 验证描述信息 |
| contrast（候选） | 同一批描述减去最近5个其他类别名文本均值 | 相对description只改变近邻中心化 |

三组均67,584可训练参数，均重新训练学生和文本残差20轮。S与简单融合F=.5S+.5M在每个模型内同时导出，确认三组学生logits相同后作为配对参照；没有额外独立学生训练。旧图像侧纠错为历史参考，不能将本批与它的差异全部归因于中心化。

Aircraft、seed1、Base、16-shot（800训练/200验证/50类别）、ViT-B/16、FP32、batch16、SGD lr=.002/momentum=.9/wd=.0005、cosine、1轮warmup。数据/模板/CLIP、训练源码及三份配置hash在入口中锁定。训练时NO_TEST=True，只导出第20轮独立重载的Base val。所有结果公开，不筛seed、不改门槛。

**事前继续条件：contrast.Z相对S、F、classname.Z、description.Z四者均至少+1个百分点（200张图净多对至少2张）。** 这是是否值得追加确认的开发门槛，不是统计显著性或发表门槛；即使通过也不自动扩量。此验证集已多轮用于开发，不作独立确认集。

## 实现与短检查

- 模型：`trainers/coop_text_residual.py`；仅给公共runtime增加新模块的训练/保存注册；旧模型默认行为不变。
- 配置：`configs/repro/text_residual/{classname,description,contrast}.yaml`。
- 启动：`PYTHONPATH=.:Dassl.pytorch python scripts/repro/text_residual_pilot.py run --data-root DATA --clip-path CLIP --output-root FRESH_OUTPUT`。仅支持3train+3export，一小时上限；不覆盖、不重试、不恢复旧队列。
- 本机两项针对性CPU检查通过，测试体0.962秒：两步更新、梯度隔离/冻结、三组同参数、零起点、近邻排除本类、完整重载、新类别文本重建、导出公式。未跑全量测试，服务器不重复这套检查。
- 训练源码SHA：`b582a30a733c735126afccf6c77f64c0d1fdec3be9fd4b34d8a227b15e7c39bf`。

## 原创性边界

[CLIP-Adapter](https://arxiv.org/abs/2110.04544)已有特征残差适配，[VDT-Adapter](https://arxiv.org/abs/2307.11661)已有描述和适配器结合；[DisCLIP作者实现（AAAI2025）](https://github.com/CompVis/DisCLIP)已有基于局部标签邻域选择判别描述的无训练方法。本轮检验可训练文本修正与固定近邻中心化的效果，不宣称“近邻判别描述”本身原创，也不是这些作者方法的复现。若有实际信号，后续需对完整公式与公平实验核查贡献边界。
