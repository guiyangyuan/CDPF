# Task 1 final config-audit fix

Status: DONE

## Implementation

- CPU preparation now resolves every generated train/evaluation command through the immutable source tree's real `train.setup_cfg` entrypoint. This applies the repository defaults, `extend_cfg`, dataset YAML, reproduction YAML, explicit CLI fields, and trailing override list in the same order as execution.
- The resulting complete effective configurations are frozen in `preparation.json`, grouped by stage and command index. Plan-only behavior is unchanged and does not resolve configs, access data, or require a GPU.
- The independent audit reads each emitted training and evaluation `effective_config.yaml`, verifies its seed against the directory/command identity, and requires exact equality with the corresponding source-derived prepared configuration. This pins optimizer values, batch sizes, input transforms, loader settings, model settings, and all other effective fields, including drift shared by every variant or seed.
- Existing cross-variant family comparison remains as a second guard for the explicitly permitted structural differences.

## Verification

- Focused command: `PYTHONPATH="$PWD:$PWD/Dassl.pytorch" /Users/bytedance/Documents/Codex/2026-09-08/https-github-com-zhengli97-awesome-prompt/work/tc-venv/bin/python -m unittest tests.repro.test_cross_dataset -v`
  - Result: 14 tests passed in 2.889 seconds.
  - The new real-config test accepts source-derived train/evaluation configurations and rejects seed-2-only LR drift, uniform batch-size drift, and a mismatched seed identity.
- Existing repro command: `PYTHONPATH="$PWD:$PWD/Dassl.pytorch" /Users/bytedance/Documents/Codex/2026-09-08/https-github-com-zhengli97-awesome-prompt/work/tc-venv/bin/python -m unittest tests.repro.test_assets tests.repro.test_components tests.repro.test_preflight tests.repro.test_regressions tests.repro.test_runner tests.repro.test_runtime_compatibility tests.repro.test_structure tests.repro.test_trainers -v`
  - Result: 45 tests passed in 2.555 seconds.

## Constraints and concerns

- No server or GPU operation was performed. The immutable model source, existing preparation-v1 evidence, old runners/configuration, datasets, and reports were not modified.
- A fresh preparation-v2 is required because the prepared evidence schema now contains complete source-derived effective configurations and the operational script hashes change with this fix.
