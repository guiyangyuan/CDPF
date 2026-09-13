# 描述监督视觉适配：本机原型

状态：**原型未训练；没有精度、泛化或新颖性结论。** 本文对应[模型改进候选](plans/2026-09-13-model-improvement.md)，解释 A/B/C 模型与配置。后续已冻结 [Aircraft seed 1 先导协议](plans/2026-09-13-visual-adapter-pilot.md)：三次训练、20 轮、Base 验证、C 对 A/B 均至少 +1.0pp 才作为继续信号。专用准备/执行/导出入口已补齐；本机检查与服务器部署状态见[准备记录](../reports/preparation/visual-adapter-pilot-20260913/README.md)。尚未启动 GPU，旧 H1 不再扩展。

## 三组对照

三组均使用 `CoOp_TextConsistency_V2_FG`、固定描述均值（`AGGREGATION=mean`），推理输出为 `0.5 * (CoOp logits + description logits)`。

| 组别 | 配置 | `USE_VISUAL_ADAPTER` | `ADAPTIVE_CE_WEIGHT` | 训练目标 |
|---|---|---:|---:|---|
| A：简单融合 | [simple_fusion.yaml](../configs/repro/visual_adapter/simple_fusion.yaml) | false | 0 | CoOp CE |
| B：普通视觉适配 | [visual_only.yaml](../configs/repro/visual_adapter/visual_only.yaml) | true | 0 | CoOp CE |
| C：描述监督视觉适配 | [visual_description.yaml](../configs/repro/visual_adapter/visual_description.yaml) | true | 1 | CoOp CE + 描述 CE |

B/C 使用类别共享的 64 维瓶颈、残差系数 0.2；固定描述 CE 通过分类 logits 更新视觉适配器。描述编码和 CLIP 图像/文本骨干保持冻结，CoOp 上下文通过 CoOp CE 更新。`USE_FEATURE_ADAPTER=false` 关闭旧的注意力查询适配器，它与新的 `USE_VISUAL_ADAPTER` 是不同开关。三组均显式关闭 KD、对比和多样性损失，保留 `KD_LOSS_MODE=full_kl` 和 `ABLATION_GROUP=full`。

适配器两层线性映射均无 bias：降维层沿用 `nn.Linear` 的 Kaiming 初始化，升维层权重为零，使初始残差为零。初始化通过 `fork_rng` 隔离随机数消耗，避免新增模块改变后续随机序列。宽度、系数、初始化和描述 CE 权重纳入模型身份；B/C 的参数量相同。

输入、优化器和预算沿用 [FGVCAircraft 参考配置](../configs/repro/fgvc_aircraft.yaml)：ViT-B/16、fp32、16-shot、batch 16、4 个上下文 token、20 条描述、SGD（学习率 0.002，momentum 0.9，weight decay 0.0005）、cosine、1 轮固定 1e-5 warmup、总计 20 轮。A/B/C 均从同一冻结 CLIP 开始；B/C 在这 **20 轮总预算**内联合训练 CoOp 和适配器，不先训练 20 轮 CoOp 再追加 20 轮。配对比较须使用相同 Base 图像、描述、种子和推理规则，分别检查 B−A、C−B。

训练只使用 Base 类别的 16-shot 图像及标签。配置默认 `SUBSAMPLE_CLASSES=base`、`TEST.SPLIT=val`、顺序验证采样、0 个 worker、CPU 设备，供本机准备和后续 Base 验证使用。这里没有 Novel 指标；Base 验证结果也不能代替泛化证据。最终评估 Novel 时，应从同一 Base checkpoint 加载共享 CoOp 上下文及视觉适配器，并根据 Novel 类别重新建立文本 token 和固定描述特征，不使用 Novel 标签更新参数。

## CPU 配置检查

在源码根目录使用已安装依赖的 Python；以下检查只合并并校验原型配置，不加载图像、CLIP 权重，不训练或评估。先导实验已在独立协议中固定 seed 1；其准备器生成完整 GPU 配置，原始 YAML 保留 CPU 默认值。

```bash
PYTHONPATH=.:Dassl.pytorch python -B - <<'PY'
from pathlib import Path
from dassl.config import get_cfg_default
from train import extend_cfg
from trainers.textconsistency_runtime import ValidatedTrainer

for path in sorted(Path("configs/repro/visual_adapter").glob("*.yaml")):
    cfg = get_cfg_default()
    extend_cfg(cfg)
    cfg.merge_from_file("configs/datasets/fgvc_aircraft.yaml")
    cfg.merge_from_file(str(path))
    cfg.SEED = 1
    cfg.freeze()
    ValidatedTrainer.check_cfg(None, cfg)
    assert not cfg.USE_CUDA
    assert cfg.TRAINER.TEXTCONSISTENCY.DEVICE == "cpu"
    assert cfg.DATASET.SUBSAMPLE_CLASSES == "base"
    assert cfg.TEST.SPLIT == "val"
    print(path.name, "configuration OK")
PY
```

`train.py` 的 `--no-train` 会构建数据集、加载本地 CLIP 和类别描述并写入 manifest；它不是仅解析配置的 dry-run。无卡准备应优先使用新协议中的 `visual_adapter_validation.py prepare`，不构建完整模型。以下仅为内存充足时的 **CPU 初始化示例**，不用于此前只有 2 GiB 内存的无卡实例。先将数据根目录、CLIP 路径和全新输出目录替换为实际路径；该命令不调用训练，也不产生验证准确率：

```bash
PYTHONPATH=.:Dassl.pytorch python -B train.py \
  --dataset-config-file configs/datasets/fgvc_aircraft.yaml \
  --config-file configs/repro/visual_adapter/visual_description.yaml \
  --root /absolute/path/to/data \
  --output-dir /absolute/path/to/fresh/visual-description-cpu-init \
  --seed 1 --no-train \
  TRAINER.TEXTCONSISTENCY.CLIP_PATH /absolute/path/to/ViT-B-16.pt
```

CLI 按“数据集配置 → 方法配置 → CLI 参数 → 末尾配置覆盖”合并；方法配置已显式提供 trainer、16-shot 和 Base 选择，因此可与 `--dataset-config-file` 组合。每次构建必须使用新的空输出目录。当前准备阶段不删除示例中的 `--no-train` 来启动完整 CPU 训练。

## 特征语义与历史身份

模型 `forward` 的第三个返回张量是**实际用于分类的归一化视觉特征**：A 为冻结 CLIP 特征，B/C 为适配后特征。CoOp 和描述分类分支都必须使用同一个适配后特征；不能把这个张量仍当作原始冻结 CLIP 特征用于交叉推理分析。

`scripts/repro/validation_probe.py` 及 H1 导出/分析链属于历史协议，依赖冻结图像特征等假设，**不得用于本原型的模型导出**。新 `visual_adapter_validation.py` 仅复用其中与模型无关的数据核验帮助函数，直接从真实模型输出保存 S/D/F；`visual_adapter_study.py` 负责固定队列及判断。旧 `run_pilot.py` 和历史 Base/Novel 汇总不是本原型的运行入口。

完整 checkpoint 应保存 CoOp 和视觉适配器的训练参数以及匹配的模型签名、损失、输入和代码身份。启用视觉适配器的 B/C 使用 format 3 签名，包含适配器配置及 `description_ce_weight`。Base→Novel 重载时仅重建类别相关文本，不能丢弃适配器权重或把 B 的 checkpoint 改为 C。现有严格 `code_sha256` 校验继续生效：历史模型仍需各自冻结的源码 release；保留默认旧行为不表示新源码可以加载具有旧代码哈希的 checkpoint。A 的旧结果是否可复用仍须逐项核对结构、输入、预算和身份，不能自动归入本次对照。
