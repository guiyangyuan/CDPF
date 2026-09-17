#!/usr/bin/env python3
"""核验本仓库的类别描述是否可追溯到 ProText/CuPL 官方资产。"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from collections import Counter
from pathlib import Path


PROTEXT_URL = "https://github.com/muzairkhattak/ProText"
PROTEXT_COMMIT = "f63d96bc971f6fde5d55e8e24bc5d93bb463453f"
CUPL_URL = "https://github.com/sarahpratt/CuPL"
CUPL_COMMIT = "719619c829792ff8f37298266e941b7f077c7f5c"


文件映射 = {
    "airplane_prompts_full.json": "airplane_prompts_full.json",
    "cal_prompts_full.json": "cal_prompts_full.json",
    "cars_prompts_full.json": "cars_prompts_full.json",
    "eurosat_prompts_full.json": "EuroSAT.json",
    "flower_prompts_full.json": "flower_prompts_full.json",
    "food_prompts_full.json": "food_prompts_full.json",
    "imagenet_prompts_full.json": "imagenet_prompts_full.json",
    "pets_prompts_full.json": "pets_prompts_full.json",
    "sun_prompts_full.json": "sun_prompts_full.json",
    "texture_prompts_full.json": "texture_prompts_full.json",
    "ucf_prompts_full.json": "ucf_prompts_full.json",
}


CUPL数据集指令 = {
    "airplane_prompts_full.json": [
        "Describe a(n) {} aircraft", "Describe the {} aircraft",
    ],
    "cal_prompts_full.json": [
        "Describe what a(n) {} looks like:", "Describe a(n) {}:",
        "What are the identifying characteristics of a(n) {}?",
    ],
    "cars_prompts_full.json": [
        "How can you identify a(n) {}?", "Description of a(n) {}, a type of car.",
        "A caption of a photo of a(n) {}:", "What are the primary characteristics of a(n) {}?",
        "Description of the exterior of a(n) {}",
        "What are the identifying characteristics of a(n) {}, a type of car?",
        "Describe an image from the internet of a(n) {}", "What does a(n) {} look like?",
        "Describe what a(n) {}, a type of car, looks like",
    ],
    "flower_prompts_full.json": [
        "Describe how to identify a(n) {}, a type of flower", "What does a(n) {} flower look like?",
    ],
    "food_prompts_full.json": [
        "Describe what {} looks like", "Visually describe {}",
        "How can you tell that the food in this photo is {}?",
    ],
    "imagenet_prompts_full.json": [
        "Describe what a(n) {} looks like", "How can you identify a(n) {}?",
        "What does a(n) {} look like?", "Describe an image from the internet of a(n) {}",
        "A caption of an image of a(n) {}:",
    ],
    "pets_prompts_full.json": [
        "Describe what a pet {} looks like", "Visually describe a(n) '{}', a type of pet.",
    ],
    "sun_prompts_full.json": [
        "Describe what a(n) {} looks like", "How can you identify a(n) {}?",
        "Describe a photo of a(n) {}",
    ],
    "texture_prompts_full.json": [
        'What does "{}" material look like?', 'What does a "{}" surface look like?',
        'What does a "{}" texture look like?', 'What does a "{}" object look like?',
        'What does a "{}" thing look like?', 'What does a "{}" pattern look like?',
    ],
    "ucf_prompts_full.json": [
        "Describe the action of {}", "What does the act of {} look like?",
        "What does a person doing {} look like?", 'Describe "{}"', 'Describe the action "{}"',
    ],
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit(root: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True, capture_output=True, text=True,
    )
    return result.stdout.strip()


def canonical_name(value: str) -> str:
    return " ".join(value.replace("_", " ").casefold().split())


def normalized_map(value: dict) -> dict[str, list[str]]:
    output = {}
    for key, descriptions in value.items():
        normalized = canonical_name(key)
        if normalized in output:
            raise ValueError(f"规范化后类别名重复：{key}")
        if not isinstance(descriptions, list) or any(not isinstance(x, str) for x in descriptions):
            raise ValueError(f"描述列表格式非法：{key}")
        output[normalized] = descriptions
    return output


def compare_file(local_path: Path, official_path: Path) -> dict:
    local = json.loads(local_path.read_text(encoding="utf-8"))
    official = json.loads(official_path.read_text(encoding="utf-8"))
    local_map, official_map = normalized_map(local), normalized_map(official)
    if set(local_map) != set(official_map):
        raise ValueError(f"规范化后类别集不一致：{local_path.name}")

    renamed = sum(1 for key in local if key not in official)
    added, removed = [], []
    for key in sorted(local_map):
        local_counter = Counter(local_map[key])
        official_counter = Counter(official_map[key])
        added.extend({"类别": key, "文本": text, "数量": count}
                     for text, count in (local_counter - official_counter).items())
        removed.extend({"类别": key, "文本": text, "数量": count}
                       for text, count in (official_counter - local_counter).items())

    record = {
        "本地文件": str(local_path),
        "ProText文件": str(official_path.relative_to(official_path.parents[1])),
        "本地SHA256": sha256(local_path),
        "ProText_SHA256": sha256(official_path),
        "字节完全一致": local_path.read_bytes() == official_path.read_bytes(),
        "类别数": len(local),
        "本地描述数": sum(len(value) for value in local.values()),
        "ProText描述数": sum(len(value) for value in official.values()),
        "仅类别键名规范化数": renamed,
        "本地新增文本": added,
        "本地删除文本": removed,
    }
    return record


def validate_known_edits(records: list[dict]) -> None:
    changed = {Path(row["本地文件"]).name: row for row in records if not row["字节完全一致"]}
    if set(changed) != {
        "cal_prompts_full.json", "cars_prompts_full.json", "pets_prompts_full.json",
        "sun_prompts_full.json", "ucf_prompts_full.json",
    }:
        raise ValueError(f"发现未登记的不一致文件：{sorted(changed)}")
    expected_diffs = {
        "cars_prompts_full.json": {
            "added": [],
            "removed": [{"类别": "2012 nissan juke hatchback", "文本": "IMG_1555.", "数量": 1}],
        },
        "sun_prompts_full.json": {
            "added": [{"类别": "creek", "文本": " A creek is usually a small, shallow stream of water.", "数量": 1}],
            "removed": [{"类别": "creek", "文本": "_A creek is usually a small, shallow stream of water.", "数量": 1}],
        },
    }
    for name, row in changed.items():
        if name not in expected_diffs:
            if row["本地新增文本"] or row["本地删除文本"]:
                raise ValueError(f"{name} 不应包含文本改动")
            continue
        expected = expected_diffs[name]
        if row["本地新增文本"] != expected["added"] or row["本地删除文本"] != expected["removed"]:
            raise ValueError(f"{name} 的文本改动超出登记范围")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--protext-root", type=Path, required=True)
    parser.add_argument("--cupl-root", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    if git_commit(args.protext_root) != PROTEXT_COMMIT:
        raise ValueError("ProText checkout 不是已审计的固定提交")
    if args.cupl_root and git_commit(args.cupl_root) != CUPL_COMMIT:
        raise ValueError("CuPL checkout 不是已审计的固定提交")

    records = [
        compare_file(
            args.project_root / "templates" / local_name,
            args.protext_root / "templates" / official_name,
        )
        for local_name, official_name in 文件映射.items()
    ]
    validate_known_edits(records)
    exact = sum(row["字节完全一致"] for row in records)
    payload = {
        "状态": "通过",
        "直接来源": {"URL": PROTEXT_URL, "提交": PROTEXT_COMMIT, "根目录许可证": "MIT"},
        "CuPL上游生成协议": {
            "URL": CUPL_URL, "提交": CUPL_COMMIT,
            "模型": "OpenAI GPT-3 DaVinci-002", "温度": 0.99,
            "每个生成指令返回数": 10, "最大token": 50, "停止符": ".",
            "生成时间": "CuPL论文与代码未公布精确调用日期。",
            "数据集生成指令": CUPL数据集指令,
            "许可说明": "该固定提交的根目录未发现单独LICENSE；本仓库直接对齐的ProText镜像根目录声明MIT。",
        },
        "EuroSAT来源边界": {
            "直接文件": "ProText/templates/EuroSAT.json",
            "每类描述数": 25,
            "可核实事实": "与ProText固定提交字节级一致。",
            "未公开元数据": ["具体LLM版本", "温度", "生成日期", "每指令采样数"],
        },
        "实验实际选择": {
            "每类上限": 20,
            "规则": "按文件原始顺序取前20条非空且文本去重的描述；CLIP分词后再去除token序列完全重复项。",
            "人工筛选": "没有按模型结果或测试集准确率筛选。只包含类别键名匹配规范化和两处已登记的脏文本清理。",
        },
        "文件统计": {"总数": len(records), "字节完全一致": exact, "已登记适配": len(records) - exact},
        "文件": records,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "描述模板来源核验.json"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# 描述模板来源与许可核验", "",
        f"- 核验结果：**{payload['状态']}**。",
        f"- 直接来源：[{PROTEXT_URL}]({PROTEXT_URL})，固定提交 `{PROTEXT_COMMIT}`。",
        f"- 上游生成协议：[{CUPL_URL}]({CUPL_URL})，固定提交 `{CUPL_COMMIT}`。",
        f"- 11个文件中，{exact}个与ProText字节级一致，{len(records)-exact}个只含下述已登记适配。", "",
        "## 已登记的本地适配", "",
        "- Caltech101、OxfordPets、SUN397、UCF101的部分JSON键名调整为数据集类别名格式；规范化后类别集与官方文件完全一致。",
        "- StanfordCars删除了一条非描述文本 `IMG_1555.`。",
        "- SUN397将一条开头误写为 `_A creek ...` 的描述修正为 `A creek ...`。",
        "- 无基于准确率的描述挑选。", "",
        "## 论文可报告的生成设置", "",
        "除EuroSAT外，这些文本资产可追溯到CuPL full prompts。CuPL使用GPT-3 DaVinci-002，温度0.99，最大50 tokens，每条数据集特定的LLM指令采样10个回答，并在句号处提前停止。不同数据集的指令数为2到9，生成的精确日期未公开。本实验不重新调用LLM，而是复用ProText公开文本资产，每类按固定文件顺序最多取20条。", "",
        "EuroSAT文件与ProText字节级一致，每类25条；但ProText未公布该文件的精确LLM版本、温度、生成日期和每指令采样数。论文必须将这些字段标为上游未披露，不得推测填充。", "",
        "| 数据集文件 | CuPL指令数 | 官方公开指令 |",
        "|---|---:|---|",
    ]
    for name, prompts in CUPL数据集指令.items():
        rendered = "<br>".join(f"`{prompt}`" for prompt in prompts)
        lines.append(f"| `{name}` | {len(prompts)} | {rendered} |")
    lines.extend([
        "", "## 许可边界", "",
        "ProText官方仓库根目录提供MIT许可证。CuPL的上述固定提交未找到单独LICENSE，因此论文应同时引用CuPL和ProText，并如实说明直接下载位置，不声称对上游文本的排他权利。", "",
        "## 文件清单", "",
        "| 本地文件 | 类别数 | 描述数 | 字节一致 | SHA256 |",
        "|---|---:|---:|:---:|---|",
    ])
    for row in records:
        lines.append(
            f"| `{Path(row['本地文件']).name}` | {row['类别数']} | {row['本地描述数']} | "
            f"{'\u662f' if row['字节完全一致'] else '\u5426（已登记）'} | `{row['本地SHA256']}` |"
        )
    (args.output_dir / "描述模板来源与许可.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(payload["文件统计"], ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
