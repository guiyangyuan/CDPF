# 描述纠错原型与三组小实验

2026-09-14完成更新：**固定三组GPU实验已完成、备份核验通过，AutoDL页面确认已关机。** 原学生/类别名纠错/详细描述纠错为32.0/32.0/32.5%，未通过继续门槛。见[真实实验结果](../reports/experiments/description-correction-20260914/README.md)。下面保留实现及事前协议；启动命令现在仅供复现参考，不可重复启动已完成队列。沿用用户要求的快速迭代方式，没有运行全量测试或多轮审查。

## 实际实现

新入口是 `CoOp_TextConsistency_Correction`，核心代码见 [trainer](../trainers/coop_textconsistency_correction.py)。原训练入口保留；新增入口返回明确命名的分数，避免套用旧的等权融合导出逻辑。

```text
S  = 原始冻结图像特征 × 可学习CoOp文本
D0 = 未适配特征 × 固定文本
Da = 适配后特征 × 同一固定文本
R  = Da - D0
Z  = S + R
loss = CE(S, y) + CE(stop_gradient(S) + R, y)
```

视觉适配器仅服务R，64维瓶颈、残差尺度0.2、无bias、升维矩阵零初始化，纠错系数固定1。CLIP及logit scale冻结，文本特征固定。D0与Da使用相同的归一化操作，使零初始化R严格为零；学生特征仍沿用原始路径。详细描述仍是至多20条有效描述编码归一化后求均值再归一化。

`CORRECTION_TEXT=classname` 使用仓库原有数据集类别名模板（Aircraft为 `a photo of a {}, a type of aircraft.`）；`description` 使用固定详细描述。两者参数量相同。类别文本按当前类别重新构建，适配器跨类别共享，不引入需要Novel标签的类别矩阵。

模型身份记录文本来源、公式、适配位置及启用状态；不同文本条件的checkpoint不能混载。保存仍包含可训练参数、优化器和调度器状态。原CLIP权重单独保存。

## 本次最小检查

命令（使用现有环境，无额外安装）：

```bash
PYTHONPATH=.:Dassl.pytorch python -m unittest tests.repro.test_description_correction -v
```

最后一次实际输出：`Ran 2 tests in 0.934s`，`OK`。这是tiny真实CLIP结构和合成输入的CPU软件检查，不是预训练ViT-B/16在真实数据集上的训练成绩；耗时仅指测试执行，不包含解释器导入及实现工作。

- 两个小批次：旧学生和新版三组的学生参数更新逐元素一致；纠错损失对学生提示无梯度；适配器两层确实更新；冻结参数不变；初始化Z=S；类别名和描述产生不同适配器更新。
- 一次保存后重载：全部分数与原模型完全一致；不同类别数时重建文本且正确加载适配器；不同文本条件混载被拒绝；真实导出函数的Z与模型推理一致。

初次短检查暴露的是对照辅助函数学习率不一致（旧辅助函数0.0003，新配置0.002），保存重载已通过。随后让旧模型和新模型从同一实验配置构造，解决了对照条件不一致；没有放宽逐元素比较。两项检查最终通过。启动脚本的CLI导入/帮助信息也通过；GPU执行路径后续已完成三组实机运行，见顶部结果。

## 固定小实验：开卡后执行

范围固定为Aircraft seed1、Base验证200张/50类、16-shot训练800张，ViT-B/16 FP32、上下文4、SGD/0.002、batch16、20轮。三组在同一总训练轮数内更新学生与纠错器，不增加学生预训练阶段。

| 组别 | 配置 | 训练目标 | ViT-B/16可训练参数（按结构计算） |
|---|---|---|---:|
| 原学生 | [student.yaml](../configs/repro/description_correction/student.yaml) | CE(S) | 2,048 |
| 类别名纠错 | [classname.yaml](../configs/repro/description_correction/classname.yaml) | CE(S)+纠错CE | 67,584 |
| 描述纠错 | [description.yaml](../configs/repro/description_correction/description.yaml) | CE(S)+纠错CE | 67,584 |

入口 [description_correction_pilot.py](../scripts/repro/description_correction_pilot.py) 恰好执行3次训练＋3次独立checkpoint重载导出；不增加独立预热训练。禁用训练后的重复评估，由独立重载一次性导出S、描述均值M、D0、Da、R、Z及原简单融合F=.5S+.5M。每10个batch记录训练情况，异常即停止，不自动重试或扩量。全队列上限3600秒，输出必须是新目录；脚本不操作服务器电源。

```bash
python scripts/repro/description_correction_pilot.py run \
  --data-root "$DATA_ROOT" \
  --clip-path "$CLIP_PATH" \
  --output-root "$FRESH_OUTPUT"
```

上述变量需在实际服务器上核对后设置，输出必须与源码/数据/权重分开。沿用已有Aircraft数据、seed1缓存和CLIP权重；启动前只核对必要资产及小型划分记录，不重新扫描全部图像或重跑CPU套件。训练源码SHA为 `84c7fa79d54a03edf4549cabd128080dfd648d9e0e34cd7a522dd06fe2fd4ddc`，三份配置SHA固化在启动脚本；每阶段核对源码及配置，保存执行命令、日志、身份和原始logits。

**继续信号：描述纠错Z须分别超过原学生S、原简单融合F和类别名纠错Z至少1pp，即200张图各净多对至少2张。** 该规则在新训练前固定。未过则停止此候选扩量；通过也只作为开发信号，需多种子及独立数据上的确认。Aircraft/DTD/UCF已参与开发，不能称作独立确认集。旧A/B/C失败结果和旧队列继续保留，不重跑。

这轮只判断候选是否值得继续，不能据此宣称创新成立或达到录用条件。相关边界和动机见[诊断与建议](../reports/diagnostics/visual-adapter-followup-20260914/README.md)。
