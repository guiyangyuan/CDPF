#!/usr/bin/env python3
"""在CPU上用冻结CLIP导出类别名与多描述的三项文本几何统计。"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from torch.nn import functional as F

from clip import clip


根目录 = Path(__file__).resolve().parents[2]
数据集定义 = {
    "DTD": {
        "模板": "templates/texture_prompts_full.json",
        "类别名模板": "{} texture.",
        "Base清单": "reports/experiments/置信度门控DTD筛选-20260914/原始结果/DTD/均值描述/种子1/基础测试集/导出.json",
        "Novel清单": "reports/experiments/置信度门控DTD筛选-20260914/原始结果/DTD/均值描述/种子1/新类别测试集/导出.json",
    },
    "UCF101": {
        "模板": "templates/ucf_prompts_full.json",
        "类别名模板": "a photo of a person doing {}.",
        "Base清单": "reports/experiments/语义一致性门控UCF101独立验证-20260915/UCF101/种子1/基础测试集/导出.json",
        "Novel清单": "reports/experiments/语义一致性门控UCF101独立验证-20260915/UCF101/种子1/新类别测试集/导出.json",
    },
    "OxfordPets": {
        "模板": "templates/pets_prompts_full.json",
        "类别名模板": "a photo of a {}, a type of pet.",
        "Base清单": "reports/experiments/语义一致性门控OxfordPets独立验证-20260915/基础测试集导出.json",
        "Novel清单": "reports/experiments/语义一致性门控OxfordPets独立验证-20260915/新类别测试集导出.json",
    },
}


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical(text: str) -> str:
    return " ".join(text.replace("_", " ").casefold().split())


def encode_unique(model, texts: list[str], batch_size: int) -> tuple[torch.Tensor, int]:
    tokens = clip.tokenize(texts, truncate=True)
    seen, keep = set(), []
    for index, row in enumerate(tokens):
        key = tuple(row.tolist())
        if key not in seen:
            seen.add(key)
            keep.append(index)
    tokens = tokens[keep]
    encoded = []
    with torch.no_grad():
        for chunk in tokens.split(batch_size):
            encoded.append(F.normalize(model.encode_text(chunk).float(), dim=-1).cpu())
    return torch.cat(encoded), len(keep)


def split_features(model, dataset: str, split: str, spec: dict, batch_size: int) -> dict:
    manifest_path = 根目录 / spec[f"{split}清单"]
    manifest = json.loads(manifest_path.read_text())
    classnames = manifest["class_order"]
    template_path = 根目录 / spec["模板"]
    raw = json.loads(template_path.read_text())
    lookup = {canonical(name): values for name, values in raw.items()}
    name_texts = [spec["类别名模板"].format(name.replace("_", " ")) for name in classnames]
    name_features, name_count = encode_unique(model, name_texts, batch_size)
    if name_count != len(classnames):
        raise ValueError(f"类别名token发生跨类别重复: {dataset} {split}")
    description_centers, rows = [], []
    for index, classname in enumerate(classnames):
        descriptions = lookup.get(canonical(classname))
        if not isinstance(descriptions, list) or not descriptions:
            raise ValueError(f"缺少描述: {dataset} {split} {classname}")
        features, count = encode_unique(model, descriptions, batch_size)
        center = F.normalize(features.mean(0), dim=0)
        description_centers.append(center)
        rows.append({
            "类别序号": index,
            "类别名": classname,
            "有效描述数": count,
            "类内一致性": float((features @ center).mean()),
            "类别名对齐": float(center @ name_features[index]),
        })
    centers = torch.stack(description_centers)
    own = (centers * name_features).sum(1)
    cross = centers @ name_features.T
    cross.fill_diagonal_(-torch.inf)
    margin = own - cross.max(1).values
    for index, row in enumerate(rows):
        row["跨类间隔"] = float(margin[index])
    return {
        "数据集": dataset,
        "划分": split,
        "类别数": len(classnames),
        "类别名模板": spec["类别名模板"],
        "描述模板路径": spec["模板"],
        "描述模板SHA256": sha(template_path),
        "类别清单路径": spec[f"{split}清单"],
        "类别清单SHA256": sha(manifest_path),
        "特征": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clip-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    clip_path = args.clip_path.expanduser().resolve()
    if sha(clip_path) != "5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f":
        raise ValueError("CLIP ViT-B/16权重SHA256不一致")
    if args.batch_size < 1:
        raise ValueError("batch-size必须为正")
    model, _ = clip.load(str(clip_path), device="cpu", jit=False)
    model.eval().float()
    outputs = []
    for dataset, spec in 数据集定义.items():
        for split in ("Base", "Novel"):
            outputs.append(split_features(model, dataset, split, spec, args.batch_size))
    payload = {
        "版本": 1,
        "设备": "cpu",
        "模型": "OpenAI CLIP ViT-B/16",
        "CLIP路径": str(clip_path),
        "CLIPSHA256": sha(clip_path),
        "特征定义": {
            "类内一致性": "mean_v cosine(d_cv, normalize(mean_v d_cv))",
            "类别名对齐": "cosine(normalize(mean_v d_cv), name_c)",
            "跨类间隔": "cosine(desc_center_c,name_c)-max_k!=c cosine(desc_center_c,name_k)",
        },
        "数据": outputs,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"状态": "完成", "输出": str(args.output), "输出SHA256": sha(args.output),
                      "划分数": len(outputs), "类别总数": sum(x["类别数"] for x in outputs)},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
