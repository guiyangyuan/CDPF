"""Read-only analysis of frozen test predictions; never fit or select a method.

This research utility writes derived diagnostics to a new directory. Test labels
are used only to describe past errors, not to estimate deployable routing scores.
"""
import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

DATASETS = ('ucf101', 'dtd', 'fgvc_aircraft')
BRANCHES = ('student', 'description_mean', 'simple_fusion', 'description_instance',
            'mean_probability_fusion', 'adk_probability_fusion')
FUSIONS = ('simple_fusion', 'mean_probability_fusion', 'adk_probability_fusion')
COMPARISONS = (('student', 'description_mean'), ('student', 'description_instance'),
               ('student', 'simple_fusion'), ('student', 'mean_probability_fusion'),
               ('student', 'adk_probability_fusion'),
               ('simple_fusion', 'mean_probability_fusion'),
               ('mean_probability_fusion', 'adk_probability_fusion'))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def transition(labels, reference, target):
    assert len(labels) == len(reference) == len(target) and labels
    reference_correct = [a == y for a, y in zip(reference, labels)]
    target_correct = [a == y for a, y in zip(target, labels)]
    both = sum(a and b for a, b in zip(reference_correct, target_correct))
    rescued = sum(not a and b for a, b in zip(reference_correct, target_correct))
    harmed = sum(a and not b for a, b in zip(reference_correct, target_correct))
    neither = sum(not a and not b for a, b in zip(reference_correct, target_correct))
    result = dict(n=len(labels), reference_correct=sum(reference_correct), target_correct=sum(target_correct),
                  both_correct=both, rescued=rescued, harmed=harmed, neither_correct=neither,
                  predictions_differ=sum(a != b for a, b in zip(reference, target)),
                  net_correct=rescued-harmed, net_pp=100*(rescued-harmed)/len(labels))
    assert both + rescued + harmed + neither == len(labels)
    assert result['target_correct'] - result['reference_correct'] == result['net_correct']
    return result


def partition(labels, student, description, fused):
    groups = defaultdict(list)
    for i, y in enumerate(labels):
        key = ('both_correct' if student[i] == y and description[i] == y else
               'student_only' if student[i] == y else
               'description_only' if description[i] == y else 'both_wrong')
        groups[key].append(i)
    output = {}
    for key in ('both_correct', 'student_only', 'description_only', 'both_wrong'):
        indices = groups[key]
        output[key + '_n'] = len(indices)
        output[key + '_fusion_correct'] = sum(fused[i] == labels[i] for i in indices)
    assert sum(output[k+'_n'] for k in groups) == len(labels)
    assert sum(output[k+'_fusion_correct'] for k in groups) == sum(a == b for a,b in zip(labels,fused))
    return output


def write_csv(path, rows):
    assert rows
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader(); writer.writerows(rows)


def self_check():
    result = transition([0, 1, 2, 3], [0, 0, 2, 0], [0, 1, 0, 0])
    assert (result['both_correct'],result['rescued'],result['harmed'],result['neither_correct']) == (1,1,1,1)
    result = partition([0,1,2,3], [0,1,0,0], [0,0,2,1], [0,0,2,3])
    assert [result[k+'_n'] for k in ('both_correct','student_only','description_only','both_wrong')] == [1,1,1,1]
    assert result['both_wrong_fusion_correct'] == 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', required=True, type=Path)
    parser.add_argument('--receipt', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    self_check()
    study, output = args.study.resolve(), args.output.resolve()
    assert not output.exists() and study not in output.parents and output not in study.parents
    receipt = json.loads(args.receipt.read_text())
    assert receipt['state'] == 'local_complete_backup_verified'
    checked = {}
    def read(path):
        digest = sha(path)
        relative = str(path.relative_to(study))
        assert digest == receipt['study_file_sha256'][relative], relative
        checked[relative] = digest
        return path.read_text()
    assert json.loads(read(study/'status.json'))['state'] == 'passed'
    analysis = json.loads(read(study/'analysis.json'))
    assert analysis['passed']
    transitions, partitions, classes = [], [], []
    frozen_samples = {}
    score_index = {(r['dataset'],r['seed'],r['branch']):r for r in analysis['fusion_seed_branch_rows']}
    record_count = 0
    for dataset in DATASETS:
        for seed in (1,2,3):
            for split in ('base','new'):
                directory = study/f'fusion_{dataset}'/f'seed{seed}'/split
                metric = json.loads(read(directory/'metrics.json'))
                rows = [json.loads(line) for line in read(directory/'predictions.jsonl').splitlines()]
                assert sha(directory/'predictions.jsonl') == metric['predictions']['sha256']
                assert (metric['dataset_key'],metric['seed'],metric['subsample'],metric['split']) == (dataset,seed,split,'test')
                samples = [(row['relative_path'],row['label'],row['label_classname']) for row in rows]
                assert len(set(x[0] for x in samples)) == len(samples) == metric['predictions']['count']
                assert hashlib.sha256(json.dumps(samples,ensure_ascii=False).encode()).hexdigest() == metric['samples']['test_sha256']
                key = dataset,split
                if key in frozen_samples:
                    assert samples == frozen_samples[key]
                else:
                    frozen_samples[key] = samples
                labels = [row['label'] for row in rows]
                assert all(type(y) is int and 0 <= y < len(metric['class_order']) for y in labels)
                assert all(set(row['predictions']) == set(BRANCHES) for row in rows)
                predictions = {branch:[row['predictions'][branch]['label'] for row in rows] for branch in BRANCHES}
                for branch in BRANCHES:
                    correct = sum(a == y for a,y in zip(predictions[branch],labels))
                    expected = score_index[dataset,seed,branch]['counts'][split]
                    assert expected == {'correct':correct,'total':len(rows)}
                    assert metric['metrics'][branch]['correct'] == correct
                for reference,target in COMPARISONS:
                    transitions.append(dict(dataset=dataset,seed=seed,split=split,reference=reference,target=target,
                        **transition(labels,predictions[reference],predictions[target])))
                for fusion in FUSIONS:
                    part = partition(labels,predictions['student'],predictions['description_mean'],predictions[fusion])
                    if fusion != 'adk_probability_fusion':
                        assert part['both_correct_n'] == part['both_correct_fusion_correct']
                    partitions.append(dict(dataset=dataset,seed=seed,split=split,fusion=fusion,**part))
                    for label in sorted(set(labels)):
                        indices = [i for i,y in enumerate(labels) if y==label]
                        cls = transition([labels[i] for i in indices],[predictions['student'][i] for i in indices],
                                         [predictions[fusion][i] for i in indices])
                        classes.append(dict(dataset=dataset,seed=seed,split=split,fusion=fusion,
                            label=label,classname=metric['class_order'][label],**cls))
                record_count += len(rows)
    assert len(transitions)==126 and len(partitions)==54 and len(frozen_samples)==6
    grouped = defaultdict(list)
    for row in transitions:
        grouped[row['dataset'],row['split'],row['reference'],row['target']].append(row)
    summaries = []
    for (dataset,split,reference,target), rows in grouped.items():
        assert sorted(r['seed'] for r in rows)==[1,2,3]
        summaries.append(dict(dataset=dataset,split=split,reference=reference,target=target,n_per_seed=rows[0]['n'],
            rescued_mean=mean(r['rescued'] for r in rows),harmed_mean=mean(r['harmed'] for r in rows),
            net_pp_mean=mean(r['net_pp'] for r in rows),net_pp_sd=stdev(r['net_pp'] for r in rows),
            disagreements_mean=mean(r['predictions_differ'] for r in rows)))
    # All input hashes are rechecked after analysis; no model loading or inference.
    assert all(sha(study/name)==digest for name,digest in checked.items())
    output.mkdir(parents=True)
    for name,rows in [('transitions.csv',transitions),('fusion-partitions.csv',partitions),
                      ('class-transitions.csv',classes),('transition-summaries.csv',summaries)]:
        write_csv(output/name,rows)
    payload = dict(passed=True,analysis_type='descriptive_only_no_fitting',
        test_prediction_records=record_count,unique_test_images=sum(len(v) for v in frozen_samples.values()),
        seed_dependence='same test images recur across seeds; never interpret pooled rows as independent samples',
        input_receipt_sha256=sha(args.receipt),input_hashes=checked,
        fusion_partitions=partitions,transition_summaries=summaries,
        limitations=['Only frozen six-branch argmax predictions, not full model KD0/KD1 per-image predictions',
                    'No logits, margins, entropy or calibration measurements',
                    'Test labels are for retrospective diagnosis only; no fitting, thresholds or test-selected method',
                    'An oracle choosing a correct input branch is not an upper bound for score fusion: both wrong branches can fuse correctly'])
    (output/'diagnostic.json').write_text(json.dumps(payload,indent=2)+'\n')
    print(json.dumps({k:v for k,v in payload.items() if k not in ('input_hashes','fusion_partitions','transition_summaries')},indent=2))


if __name__ == '__main__':
    main()
