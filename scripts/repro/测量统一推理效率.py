#!/usr/bin/env python3
"""在同一块 GPU、同一检查点和同一批输入上比较 CoOp 与两种固定融合的推理效率。

本脚本只做冻结推理，不训练、不搜索超参。描述文本特征在计时前编码为每类
一个固定向量；模型和数据加载、检查点恢复及文本特征编码都不计入延迟。
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import platform
import sys
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

import numpy as np
import torch
from torch.nn import functional as F

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
import evaluate_frozen_fusion as fusion


METHODS = {
    "coop": "CoOp",
    "logit_fusion": "固定logit均值融合",
    "probability_fusion": "固定均值概率融合",
}


def description_mean_features(features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """复现正式评估中的“逐描述归一化→有效描述均值→再归一化”。"""
    if features.ndim != 3 or mask.ndim != 2 or tuple(mask.shape) != tuple(features.shape[:2]):
        raise ValueError("描述特征必须为 [类别,视角,维度]，掩码必须为 [类别,视角]")
    valid = mask.to(device=features.device, dtype=torch.bool)
    if not valid.any(dim=1).all().item():
        raise ValueError("每个类别至少需要一条有效描述")
    normalized = F.normalize(features.float(), dim=-1)
    weights = valid.to(normalized.dtype) / valid.sum(dim=1, keepdim=True)
    return F.normalize(torch.einsum("cv,cvd->cd", weights, normalized), dim=-1)


def fuse_from_features(
    student: torch.Tensor,
    image_features: torch.Tensor,
    class_features: torch.Tensor,
    logit_scale: torch.Tensor,
    method: str,
) -> torch.Tensor:
    """仅实现论文主结果里的两个预注册固定公式。"""
    if method not in ("logit_fusion", "probability_fusion"):
        raise ValueError(f"未知融合方法: {method}")
    images = F.normalize(image_features.float(), dim=-1)
    descriptions = logit_scale.float() * images @ class_features.float().t()
    descriptions = descriptions.to(student.dtype)
    if method == "logit_fusion":
        return 0.5 * student + 0.5 * descriptions
    return 0.5 * torch.softmax(student, dim=-1) + 0.5 * torch.softmax(descriptions, dim=-1)


def summarize_milliseconds(values: Iterable[float], batch_size: int) -> dict[str, float]:
    samples = np.asarray(list(values), dtype=np.float64)
    if samples.ndim != 1 or samples.size == 0 or not np.isfinite(samples).all() or (samples <= 0).any():
        raise ValueError("计时样本必须是非空、有限的正数")
    mean_ms = float(samples.mean())
    return {
        "迭代数": int(samples.size),
        "批次延迟均值_ms": mean_ms,
        "批次延迟中位数_ms": float(np.median(samples)),
        "批次延迟P95_ms": float(np.percentile(samples, 95)),
        "单图折算延迟_ms": mean_ms / batch_size,
        "吞吐_图每秒": 1000.0 * batch_size / mean_ms,
        "标准差_ms": float(samples.std(ddof=1)) if samples.size > 1 else 0.0,
    }


def _tensor_sha256(value: torch.Tensor) -> str:
    array = value.detach().float().cpu().contiguous().numpy()
    return hashlib.sha256(array.tobytes()).hexdigest()


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def _fixed_images(loader: Any, count: int, device: torch.device) -> torch.Tensor:
    chunks, total = [], 0
    for batch in loader:
        images = batch["img"].detach().cpu()
        chunks.append(images)
        total += len(images)
        if total >= count:
            break
    if total < count:
        raise ValueError(f"测试集只有 {total} 张可用图片，少于需要的 {count} 张")
    images = torch.cat(chunks, dim=0)[:count].contiguous()
    if not torch.isfinite(images).all().item():
        raise ValueError("固定输入中包含非有限像素")
    return images.to(device)


def _measure(
    function: Callable[[torch.Tensor], torch.Tensor],
    images: torch.Tensor,
    warmup: int,
    iterations: int,
) -> tuple[dict[str, float], str]:
    gc.collect()
    torch.cuda.empty_cache()
    with torch.inference_mode():
        output = None
        for _ in range(warmup):
            output = function(images)
        torch.cuda.synchronize()
        del output
        baseline_allocated = torch.cuda.memory_allocated()
        baseline_reserved = torch.cuda.memory_reserved()
        torch.cuda.reset_peak_memory_stats()
        starts = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
        ends = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
        output = None
        for start, end in zip(starts, ends):
            start.record()
            output = function(images)
            end.record()
        torch.cuda.synchronize()
        elapsed = [start.elapsed_time(end) for start, end in zip(starts, ends)]
        digest = _tensor_sha256(output)
        peak_allocated = torch.cuda.max_memory_allocated()
        peak_reserved = torch.cuda.max_memory_reserved()
    result = summarize_milliseconds(elapsed, len(images))
    result.update({
        "计时前已分配显存_MiB": baseline_allocated / 2**20,
        "峰值已分配显存_MiB": peak_allocated / 2**20,
        "增量峰值显存_MiB": (peak_allocated - baseline_allocated) / 2**20,
        "计时前保留显存_MiB": baseline_reserved / 2**20,
        "峰值保留显存_MiB": peak_reserved / 2**20,
    })
    return result, digest


def benchmark(args: argparse.Namespace) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("统一效率测试必须使用 CUDA，禁止 CPU 回退")
    if args.warmup < 1 or args.iterations < 2:
        raise ValueError("预热至少 1 次，正式计时至少 2 次")
    batch_sizes = sorted(set(args.batch_sizes))
    if not batch_sizes or batch_sizes[0] < 1:
        raise ValueError("batch size 必须为正数")

    source_root = args.source_root.resolve()
    baseline_dir = args.baseline_dir.resolve()
    data_root = args.data_root.resolve()
    output_dir = fusion._ensure_disjoint_fresh_output(
        args.output_dir.resolve(), (source_root, baseline_dir, data_root))
    actual_source_sha = fusion.model_source_digest(source_root)
    if actual_source_sha != args.expected_source_sha256:
        raise ValueError("冻结源码 SHA-256 与指定值不一致")

    _, cfg, expected_train, expected_eval, training = fusion.build_source_configs(
        source_root, data_root, output_dir, args.dataset, args.seed, args.split)
    metadata = fusion.validate_baseline_metadata(
        baseline_dir, args.dataset, args.seed, args.split, args.expected_source_sha256,
        expected_train_config=expected_train, expected_eval_config=expected_eval)

    from dassl.utils import set_random_seed
    from trainers.textconsistency_common import CachedMultiViewTextEncoder, load_clip_checkpoint

    set_random_seed(args.seed)
    trainer = training.build_trainer(cfg)
    current = trainer.run_provenance
    historical = metadata["metric"]
    for field in ("dataset", "seed", "shots", "code_sha256", "train_samples_sha256",
                  "val_samples_sha256", "test_samples_sha256", "trainable_parameters", "model_signature"):
        if current.get(field) != historical.get(field):
            raise ValueError(f"当前模型/数据与历史基线不一致: {field}")
    trainer.load_model(baseline_dir, epoch=fusion.EPOCH)
    loaded = getattr(trainer, "loaded_checkpoints", [])
    if len(loaded) != 1 or loaded[0]["sha256"] != metadata["checkpoint_sha256"]:
        raise ValueError("检查点加载收据与冻结基线不一致")
    fusion._validate_checkpoint_tensors(Path(metadata["checkpoint"]), metadata["train_manifest"])
    trainer.set_model_mode("eval")
    device = trainer.device
    images = _fixed_images(trainer.test_loader, max(batch_sizes), device)

    model = trainer.model
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    total = sum(parameter.numel() for parameter in model.parameters())
    if trainable != fusion.TRAINABLE_PARAMETERS:
        raise ValueError(f"可训练参数应为 {fusion.TRAINABLE_PARAMETERS}，实际为 {trainable}")

    def coop_forward(batch: torch.Tensor) -> torch.Tensor:
        student, _, _, _ = model(batch, return_attn=False)
        return student

    results: dict[str, dict[str, Any]] = {name: {} for name in METHODS}
    output_digests: dict[str, dict[str, str]] = {name: {} for name in METHODS}
    for batch_size in batch_sizes:
        metrics, digest = _measure(coop_forward, images[:batch_size], args.warmup, args.iterations)
        results["coop"][str(batch_size)] = metrics
        output_digests["coop"][str(batch_size)] = digest

    clip_model = load_clip_checkpoint(cfg)
    encoder = CachedMultiViewTextEncoder(cfg, trainer.dm.dataset.classnames, clip_model)
    descriptions = encoder.get_view_features()
    view_mask = encoder.view_mask
    class_features_cpu = description_mean_features(descriptions, view_mask).cpu()
    description_counts = list(encoder.description_counts)
    template_sha256 = encoder.template_sha256
    del encoder, clip_model, descriptions
    gc.collect()
    torch.cuda.empty_cache()
    class_features = class_features_cpu.to(device)
    logit_scale = model.logit_scale.exp().detach()

    def forward_with(method: str, batch: torch.Tensor) -> torch.Tensor:
        student, _, features, _ = model(batch, return_attn=False)
        return fuse_from_features(student, features, class_features, logit_scale, method)

    # 用正式评估公式对首批输出做数值级绑定，防止效率脚本测到另一个实现。
    with torch.inference_mode():
        check_images = images[:min(4, len(images))]
        student, _, features, _ = model(check_images, return_attn=False)
        reference = fusion.frozen_fusion_logits(student, features, class_features[:, None, :],
                                                 torch.ones(len(class_features), 1, dtype=torch.bool,
                                                            device=device), logit_scale)
        torch.testing.assert_close(
            forward_with("logit_fusion", check_images), reference["simple_fusion"], rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(
            forward_with("probability_fusion", check_images),
            reference["mean_probability_fusion"], rtol=1e-5, atol=1e-5)

    for method in ("logit_fusion", "probability_fusion"):
        for batch_size in batch_sizes:
            function = lambda batch, selected=method: forward_with(selected, batch)
            metrics, digest = _measure(function, images[:batch_size], args.warmup, args.iterations)
            results[method][str(batch_size)] = metrics
            output_digests[method][str(batch_size)] = digest

    for method in METHODS:
        for batch_size in batch_sizes:
            base = results["coop"][str(batch_size)]["批次延迟均值_ms"]
            current_ms = results[method][str(batch_size)]["批次延迟均值_ms"]
            results[method][str(batch_size)]["相对CoOp延迟倍数"] = current_ms / base
            results[method][str(batch_size)]["相对CoOp延迟增幅_百分比"] = 100.0 * (current_ms / base - 1.0)

    class_buffer_bytes = class_features.numel() * class_features.element_size()
    payload = {
        "状态": "通过",
        "协议": {
            "任务": "冻结推理效率比较",
            "数据集": fusion.DATASETS[args.dataset][0],
            "子集": args.split,
            "种子": args.seed,
            "轮数": fusion.EPOCH,
            "batch_sizes": batch_sizes,
            "预热次数": args.warmup,
            "计时次数": args.iterations,
            "排除项": ["模型加载", "数据加载", "检查点恢复", "描述文本编码"],
            "融合系数": 0.5,
            "描述类向量": "逐描述归一化，对有效描述求均值，再归一化；计时前固定",
            "方法顺序": list(METHODS),
        },
        "模型": {
            "骨干": cfg.MODEL.BACKBONE.NAME,
            "类别数": len(trainer.dm.dataset.classnames),
            "可训练参数": trainable,
            "模型总参数": total,
            "固定描述类向量字节数": class_buffer_bytes,
            "固定描述类向量_MiB": class_buffer_bytes / 2**20,
            "新增可训练参数": {name: 0 for name in METHODS},
            "每类有效描述数": description_counts,
        },
        "结果": results,
        "输出SHA256": output_digests,
        "数值一致性": {
            "状态": "通过",
            "对照": "scripts/repro/evaluate_frozen_fusion.py::frozen_fusion_logits",
            "rtol": 1e-5,
            "atol": 1e-5,
        },
        "来源": {
            "源码根目录": str(source_root),
            "模型源码SHA256": actual_source_sha,
            "检查点": metadata["checkpoint"],
            "检查点SHA256": metadata["checkpoint_sha256"],
            "模板SHA256": template_sha256,
            "基准脚本SHA256": fusion.sha256_file(Path(__file__).resolve()),
        },
        "环境": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
            "cudnn": torch.backends.cudnn.version(),
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
        },
        "限制": [
            "只在一块 GPU 上测量，不表示其他硬件的绝对延迟",
            "计时时重用固定的预处理图像，不包含磁盘读取和 CPU 图像预处理",
            "三个方法按固定顺序测量，绝对数字可受 GPU 频率波动影响",
        ],
    }
    if not all(math.isfinite(value) for method in results.values() for batch in method.values()
               for value in batch.values() if isinstance(value, float)):
        raise ValueError("结果中包含非有限数字")
    # Dassl's logger creates the configured output directory while the model is
    # constructed. The path was verified fresh above, so reusing that same
    # framework-created directory here is expected and safe.
    output_dir.mkdir(parents=True, exist_ok=True)
    _atomic_json(output_dir / "效率实测结果.json", payload)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--dataset", choices=tuple(fusion.DATASETS), required=True)
    parser.add_argument("--seed", type=int, choices=fusion.ALLOWED_SEEDS, required=True)
    parser.add_argument("--split", choices=("base", "new"), required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-source-sha256", required=True)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=(1, 64))
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    return parser


def main(argv: list[str] | None = None) -> int:
    result = benchmark(build_parser().parse_args(argv))
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
