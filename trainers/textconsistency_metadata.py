"""与设备无关的描述顺序元数据工具。"""

from __future__ import annotations

import hashlib
import json
import random


def description_permutation(count: int, policy: str = "contiguous", n_groups: int = 4,
                            seed: int = 0) -> list[int]:
    """返回描述视图的固定排列，不依赖训练随机数。"""
    if count < 1 or n_groups < 1 or not isinstance(seed, int) or seed < 0:
        raise ValueError("Invalid description order configuration")
    if policy not in ("contiguous", "interleaved", "shuffle") or (policy != "shuffle" and seed != 0):
        raise ValueError("Use contiguous/interleaved with seed 0, or shuffle with an explicit seed")
    order = list(range(count))
    if policy == "interleaved":
        order.sort(key=lambda index: (index % n_groups, index // n_groups))
    elif policy == "shuffle":
        random.Random(seed).shuffle(order)
    return order


def view_order_manifest(classnames: list[str], counts: list[int], policy: str,
                        n_groups: int, seed: int) -> dict:
    """按类别位置记录描述顺序；显示名称相同的 synset 仍保持独立身份。"""
    if len(classnames) != len(counts):
        raise ValueError("Description ordering requires complete counts")
    duplicate_names = len(set(classnames)) != len(classnames)
    records = []
    for index, (name, count) in enumerate(zip(classnames, counts)):
        record = {
            "classname": name,
            "count": count,
            "order": description_permutation(count, policy, n_groups, seed),
        }
        if duplicate_names:
            record["class_index"] = index
        records.append(record)
    payload = {
        "version": 1,
        "policy": policy,
        "seed": seed,
        "n_groups": n_groups,
        "classes": records,
    }
    payload["sha256"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()
    return payload
