"""Post-hoc CPU diagnostics of archived logits; no fitting or model execution."""
import hashlib
import json
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
SOURCE = HERE.parents[1] / "experiments" / "visual-adapter-westc-20260914"
VARIANTS = {"A": "simple_fusion", "B": "visual_only", "C": "visual_description"}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def probability(x):
    x = x.astype(np.float64)
    e = np.exp(x - x.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def summarize(student, description, labels):
    fusion = 0.5 * student + 0.5 * description
    prob_fusion = 0.5 * probability(student) + 0.5 * probability(description)
    s, d, f = [x.argmax(axis=1) == labels for x in (student, description, fusion)]
    count = lambda x: int(np.count_nonzero(x))
    n = len(labels)
    return {
        "n": n,
        "student_correct": count(s),
        "description_correct": count(d),
        "logit_fusion_correct": count(f),
        "probability_fusion_correct": count(prob_fusion.argmax(axis=1) == labels),
        "student_accuracy_pct": 100 * count(s) / n,
        "description_accuracy_pct": 100 * count(d) / n,
        "logit_fusion_accuracy_pct": 100 * count(f) / n,
        "probability_fusion_accuracy_pct": 100 * count(prob_fusion.argmax(axis=1) == labels) / n,
        "only_student_correct": count(s & ~d),
        "only_description_correct": count(~s & d),
        "both_correct": count(s & d),
        "both_wrong": count(~s & ~d),
        "fusion_corrects_student": count(~s & f),
        "fusion_harms_student": count(s & ~f),
    }


def main():
    arrays, metadata, inputs = {}, {}, {}
    for key, variant in VARIANTS.items():
        folder = SOURCE / variant
        meta_path, npz_path = folder / "export.json", folder / "logits.npz"
        meta = json.loads(meta_path.read_text())
        assert sha(npz_path) == meta["logits_sha256"], variant
        with np.load(npz_path, allow_pickle=False) as data:
            arrays[key] = {name: data[name].copy() for name in data.files}
        arr = arrays[key]
        assert arr["labels"].shape == (200,)
        assert len(meta["rows"]) == 200 and len(meta["class_order"]) == 50
        np.testing.assert_array_equal(arr["labels"], [row["label"] for row in meta["rows"]])
        for branch in ("S", "D", "F"):
            assert arr[branch].shape == (200, 50) and np.isfinite(arr[branch]).all()
        np.testing.assert_array_equal(arr["F"], 0.5 * arr["S"] + 0.5 * arr["D"])
        metadata[key] = meta
        inputs[key] = {
            "variant": variant,
            "logits_sha256": sha(npz_path),
            "export_json_sha256": sha(meta_path),
            "source_sha256": meta["source_sha256"],
        }
    for key in ("B", "C"):
        for field in ("rows", "class_order", "protocol", "source_sha256", "decoded_val"):
            assert metadata[key][field] == metadata["A"][field], (key, field)
        np.testing.assert_array_equal(arrays[key]["labels"], arrays["A"]["labels"])
    # These five combinations are explicit post-hoc diagnostics, not a model search.
    combinations = [("A", "A"), ("B", "B"), ("C", "C"), ("A", "C"), ("C", "A")]
    result = {
        "status": "exploratory_post_hoc_diagnostic",
        "scope": "Aircraft seed1 Base validation; previously used for model development",
        "new_training_runs": 0,
        "gpu_inference_runs": 0,
        "fitted_parameters": 0,
        "alpha_search": False,
        "warning": "Cross-model branch swaps are diagnostics, not new trained-method results or causal evidence.",
        "inputs": inputs,
        "combinations": {
            f"{s}_S+{d}_D": summarize(arrays[s]["S"], arrays[d]["D"], arrays[s]["labels"])
            for s, d in combinations
        },
    }
    output = HERE / "diagnostic.json"
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: v["logit_fusion_accuracy_pct"] for k, v in result["combinations"].items()}))


if __name__ == "__main__":
    main()
