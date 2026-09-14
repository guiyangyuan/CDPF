"""Read existing Base-validation predictions and description text; no inference."""
import csv
import hashlib
import json
from pathlib import Path
import re

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def family(name):
    # A transparent name-prefix diagnostic, NOT official taxonomy or model input.
    for prefix in ('737-', '747-', '757-', '767-', '777-', 'A330-', 'A340-',
                   'ATR-', 'BAE 146-', 'CRJ-'):
        if name.startswith(prefix):
            return prefix.rstrip('-')
    return name


def main():
    source = ROOT / 'reports/experiments/text-adamw-20260914'
    out = ROOT / 'reports/diagnostics/text-adamw-errors-20260914'
    out.mkdir(parents=True, exist_ok=True)
    records, arrays, hashes = {}, {}, {}
    for mode in ('classname', 'description', 'contrast'):
        folder = source / mode
        records[mode] = json.loads((folder / 'export.json').read_text())
        digest = hashlib.sha256((folder / 'logits.npz').read_bytes()).hexdigest()
        assert digest == records[mode]['logits_sha256']
        hashes[mode] = digest
        with np.load(folder / 'logits.npz', allow_pickle=False) as z:
            arrays[mode] = {key: z[key].copy() for key in z.files}
    names = records['classname']['class_order']
    labels = arrays['classname']['labels']
    base = arrays['classname']['Z'].argmax(1)
    ordered = np.sort(arrays['classname']['Z'], axis=1)
    margins = ordered[:, -1] - ordered[:, -2]
    summary, rows, class_rows = {}, [], []
    for mode in ('description', 'contrast'):
        assert records[mode]['rows'] == records['classname']['rows']
        assert records[mode]['class_order'] == names
        np.testing.assert_array_equal(labels, arrays[mode]['labels'])
        pred = arrays[mode]['Z'].argmax(1)
        win, loss = (pred == labels) & (base != labels), (pred != labels) & (base == labels)
        assert (int(win.sum()), int(loss.sum())) == (10, 12)
        groups = {'corrected': win, 'harmed': loss,
                  'both_correct': (pred == labels) & (base == labels),
                  'both_wrong': (pred != labels) & (base != labels)}
        summary[mode] = {'corrected': int(win.sum()), 'harmed': int(loss.sum()),
            'prediction_disagreements': int((pred != base).sum()),
            'name_prefix_within_family_harms': sum(family(names[int(labels[i])]) == family(names[int(pred[i])])
                                                 for i in np.flatnonzero(loss)),
            'corrected_indices': np.flatnonzero(win).tolist(), 'harmed_indices': np.flatnonzero(loss).tolist(),
            'baseline_top1_top2_margin_median': {k: float(np.median(margins[m])) for k, m in groups.items()},
        }
        for i in range(len(labels)):
            rows.append({'mode': mode, 'index': i, 'sample_id': records[mode]['rows'][i]['sample_id'],
                'label': int(labels[i]), 'true_class': names[int(labels[i])], 'classname_prediction': names[int(base[i])],
                'description_prediction': names[int(pred[i])], 'transition': next(k for k, mask in groups.items() if mask[i]),
                'classname_top1_top2_margin': float(margins[i])})
        for c, name in enumerate(names):
            mask = labels == c
            class_rows.append({'mode': mode, 'class': name, 'n': int(mask.sum()),
                'classname_correct': int(((base == labels) & mask).sum()),
                'description_correct': int(((pred == labels) & mask).sum()),
                'corrected': int((win & mask).sum()), 'harmed': int((loss & mask).sum())})
    templates_path = ROOT / 'templates/airplane_prompts_full.json'
    templates = json.loads(templates_path.read_text())
    text_rows, punctuation = [], []
    for name in names:
        sentences = [s.strip() for s in templates[name] if s.strip()][:20]
        canonical = [re.sub(r'\W+', ' ', s.lower()).strip() for s in sentences]
        for i, text in enumerate(canonical):
            if not text: punctuation.append({'class': name, 'description_index': i, 'text': sentences[i]})
        text_rows.append({'class': name, 'selected_count': len(sentences),
            'exact_normalized_duplicate_excess': len(canonical) - len(set(canonical)),
            'mentions_calendar_year': sum(bool(re.search(r'\b(?:19|20)\d{2}\b', s)) for s in sentences),
            'mentions_passenger_or_seating': sum(bool(re.search(r'passenger|seat|capacity', s, re.I)) for s in sentences)})
    for name, values in [('transitions.csv', rows), ('per_class.csv', class_rows), ('text_inventory.csv', text_rows)]:
        with (out / name).open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(values[0]), lineterminator="\n"); writer.writeheader(); writer.writerows(values)
    report = {'scope': 'Existing Aircraft seed1 Base validation development predictions only',
        'n': len(labels), 'new_training': 0, 'new_model_inference': 0, 'input_logits_sha256': hashes,
        'template_sha256': hashlib.sha256(templates_path.read_bytes()).hexdigest(), 'modes': summary,
        'same_harmed_indices': sorted(set(summary['description']['harmed_indices']) & set(summary['contrast']['harmed_indices'])),
        'text_inventory': {'n': sum(x['selected_count'] for x in text_rows),
            'within_class_normalized_duplicate_excess': sum(x['exact_normalized_duplicate_excess'] for x in text_rows),
            'year_mentions': sum(x['mentions_calendar_year'] for x in text_rows),
            'passenger_or_seating_mentions': sum(x['mentions_passenger_or_seating'] for x in text_rows),
            'punctuation_only': punctuation},
        'limitations': ['Prefix families are an explicit heuristic, not official taxonomy.',
                       'Text keyword counts do not label all sentences as good/bad or prove causation.',
                       'No image pixels inspected; no claim that an individual visual attribute is visible.',
                       'No validation threshold search or new accuracy claim.']}
    (out / 'analysis.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
