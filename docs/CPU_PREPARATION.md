# 无卡准备与后续运行

当前阶段以 [PROJECT_STATUS.md](../PROJECT_STATUS.md) 为准。无卡容器内存上限实际为 2 GiB；不要按宿主机 free 输出推断可用内存。准备阶段完整 CLIP 诊断在本机通过，服务器无卡尝试被 SIGKILL；现已开卡并在120GiB内存下复验通过，见 [本批记录](../reports/experiments/ucf101-kd-study-20260910/README.md)；轻量测试/数据预检不受这一结果替代。下面的正式运行命令仅在用户开卡后使用；`--plan-only` 不训练。

2026-09-10 无卡准备检查已完成：[完成报告](../reports/preparation/无卡准备完成.md)、[检查记录与已生成的命令计划](../reports/preparation/cpu-prep-20260910/)。这些计划记录的是命令预览，不是已执行的训练日志。

## 服务器布局

根目录 `/root/autodl-tmp/textconsistency`：

- `.venv/`：Python3.12.3、torch2.8.0+cu128、torchvision0.23.0+cu128。
- `src/`：旧三种子时期的源码快照，保留不变。
- `releases/cpu-prep-20260910/`：本轮准备的代码；训练模块没有改变，脚本/配置增加 KD 控制和数据集选项。
- `references/official-coop-20260910/`：未经源码修改的固定 CoOp 上游 commit，CPU 行为诊断及官方参考命令使用它。
- `data/{ucf101,dtd,fgvc_aircraft}/`：已验证 ZIP 提取的数据，带固定划分和来源标记；具体完成情况看准备报告。
- `assets/`：数据 ZIP 和 CLIP 权重；不能提交到 Git。
- `setup-logs/cpu-prep-20260910/`：CPU 检查、预检、诊断、计划及 readiness。
- `runs/`：后续 GPU 输出，每个实验必须用新目录。

激活本轮代码可直接执行 `source /root/autodl-tmp/textconsistency/activate-current.sh`，等价步骤如下：

```bash
export TC_ROOT=/root/autodl-tmp/textconsistency
source "$TC_ROOT/.venv/bin/activate"
export TC_SRC="$TC_ROOT/releases/cpu-prep-20260910"
export PYTHONPATH="$TC_SRC:$TC_SRC/Dassl.pytorch"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
cd "$TC_SRC"
```

镜像原有旧 Dassl 安装元数据与当前依赖冲突；本轮把 release 内 `Dassl.pytorch/dassl.egg-info` 移到 setup-logs 下保存，仅变更安装元数据，训练 Python 源码未改。不要额外安装 vendored Dassl 中旧版 requirements。

## 无卡可以运行的检查

```bash
python -m pip check
python -B -m unittest discover -s tests -v
python scripts/repro/preflight.py --dataset ucf101 --data-root "$TC_ROOT/data"
python scripts/repro/preflight.py --dataset dtd --data-root "$TC_ROOT/data"
python scripts/repro/preflight.py --dataset fgvc_aircraft --data-root "$TC_ROOT/data"
```

数据 ZIP 获取后，用 `prepare_assets.py --dtd-zip ... --aircraft-zip ... --data-root ...` 核验 SHA256 并安全解压。DTD 必须使用包内固定 Zhou JSON；Aircraft 使用 variants 和 train/val/test metadata，缺失时预检直接失败，不重新随机拆分。

## 第一批：UCF101 蒸馏对照

开卡前可以生成命令预览：

```bash
python scripts/repro/run_pilot.py --dataset ucf101 --data-root "$TC_ROOT/data" --output-root "$TC_ROOT/runs/ucf101-kd0-s123-v1" --seeds 1 2 3 --variants hierarchical hierarchical_adapter --kd-weight 0 --plan-only
python scripts/repro/run_pilot.py --dataset ucf101 --data-root "$TC_ROOT/data" --output-root "$TC_ROOT/runs/ucf101-kd1-s123-v1" --seeds 1 2 3 --variants hierarchical hierarchical_adapter --kd-weight 1 --plan-only
```

正式运行前先核验 CUDA，再使用新的 output-root，删除命令末尾的 `--plan-only`。两组共 12 次训练、72 次独立分支评估，均为20 epochs/16-shot/seed1,2,3。只改变 KD 权重，teacher CE=1、融合 alpha=.5、模型结构不变。KD1 本轮成对重跑，不把旧目录冒充本轮训练。

```bash
python -c 'import torch; assert torch.cuda.is_available(), "GPU 未启用"; print(torch.cuda.get_device_name(0))'
```

每个目录保存 protocol/effective_config/manifest/checkpoint/metrics；默认无 CUDA 会在训练前报错退出。`--device cpu` 只用于明确限定的诊断，不在无卡阶段运行完整 benchmark。

## 独立 CoOp 对照

```bash
python scripts/repro/plan_official_coop.py --upstream-root "$TC_ROOT/references/official-coop-20260910" --data-root "$TC_ROOT/data" --output-root "$TC_ROOT/runs/official-coop-matched20-s123-v1" --recipe matched20
python scripts/repro/plan_official_coop.py --upstream-root "$TC_ROOT/references/official-coop-20260910" --data-root "$TC_ROOT/data" --output-root "$TC_ROOT/runs/official-coop-reference200-s123-v1" --recipe reference200
```

该工具**只生成 JSON 命令，不执行训练**。matched20 与当前基线对齐训练条件；reference200 使用官方 YAML 的200轮、batch32、词语初始化，精度明确改为FP32。开卡后先按计划核对环境、CLIP缓存、采样与输出路径，再执行命令并记录实际 checkpoint SHA。官方训练入口不产生本项目 TC 格式的全部审计字段，需独立收集日志和来源，不能直接塞进 TC 汇总器。详见 [基线核查](../reports/preparation/CoOp基线核查.md)。

## 后续：两个新数据集

DTD 和 Aircraft 均先保留当前六组、三种子、20轮统一协议；不按测试分数单独调参。使用 `run_pilot.py --dataset dtd` 或 `--dataset fgvc_aircraft`、新输出目录及 `--seeds 1 2 3`。其余超参数不变，默认 KD1，默认 CUDA。

随机分组、参数量控制、AAPE/ADK 正式比较尚未实现或完成。本轮完成环境准备不代表完成整个投稿实验计划。

## 每批实验结束后

核验日志与完整 checkpoint；保存全部种子和三个分支，记录失败和重跑原因；将小体积汇总/审计同步 GitHub，并更新 PROJECT_STATUS.md。完整权重和数据另行备份。服务器是否关机由用户决定，本脚本不会调用平台开卡/关机接口。
