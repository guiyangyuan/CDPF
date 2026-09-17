#!/usr/bin/env python3
import importlib.util
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

准备目录 = Path('/root/autodl-tmp/textconsistency/preparation/剩余两数据集-20260917-03cc19f')
计划路径 = 准备目录 / '服务器核验/联合上下文剩余两数据集计划.json'
脚本路径 = Path('/root/autodl-tmp/textconsistency/releases/投稿标准基线-20260916-fc59a83/scripts/repro/运行三数据集联合上下文基线.py')
spec = importlib.util.spec_from_file_location('联合上下文执行器', 脚本路径)
模块 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(模块)

计划 = json.loads(计划路径.read_text(encoding='utf-8'))
源码目录, _, 输出目录 = 模块.检查计划身份(计划, require_runtime=True)
阶段 = '剩余两数据集'
阶段目录 = 输出目录 / 阶段
状态路径 = 阶段目录 / '执行状态.json'
状态 = json.loads(状态路径.read_text(encoding='utf-8'))
if 状态.get('状态') != '失败' or 状态.get('已完成命令') != 1:
    raise RuntimeError(f'拒绝恢复非预期状态：{状态.get("状态")} / {状态.get("已完成命令")}')
运行列表 = 模块.阶段运行(计划, 阶段)
if len(运行列表) != 6:
    raise RuntimeError(f'运行数错误：{len(运行列表)}')

env = os.environ.copy()
env.update({
    'CUDA_VISIBLE_DEVICES': '0',
    'CUBLAS_WORKSPACE_CONFIG': ':4096:8',
    'OMP_NUM_THREADS': '4',
    'PYTHONPATH': f'{源码目录}:{源码目录 / "Dassl.pytorch"}',
    'TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD': '1',
})
首次失败 = 状态.pop('失败命令')
状态.update({
    '状态': '恢复中',
    '首次失败命令': 首次失败,
    '恢复开始时间Unix': time.time(),
    '恢复说明': '复用StanfordCars seed1已成功训练的第10轮检查点；仅恢复检查点重载和后续命令。通过PyTorch官方环境变量加载本任务自产可信检查点，源码与冻结协议不变。',
})
模块.原子写json(状态路径, 状态)

首次运行目录 = Path(运行列表[0]['运行目录'])
失败评估目录 = 首次运行目录 / '基础测试'
if 失败评估目录.exists():
    失败评估目录.rename(首次运行目录 / '基础测试-首次失败-权重加载兼容')
失败日志 = 阶段目录 / '运行日志/01-StanfordCars-种子1-基础测试.txt'
if 失败日志.exists():
    失败日志.rename(阶段目录 / '运行日志/01-StanfordCars-种子1-基础测试-首次失败.txt')

结果行 = []
try:
    for index, 运行 in enumerate(运行列表):
        运行目录 = Path(运行['运行目录'])
        if index == 0:
            检查点 = Path(运行['检查点'])
            if not 检查点.is_file():
                raise FileNotFoundError(f'恢复所需检查点不存在：{检查点}')
            检查点SHA = 模块.sha256(检查点)
            步骤列表 = ('base', 'new')
        else:
            if 运行目录.exists():
                raise FileExistsError(f'后续运行目录已存在，拒绝覆盖：{运行目录}')
            运行目录.mkdir(parents=True)
            检查点SHA = None
            步骤列表 = ('训练', 'base', 'new')
        指标 = {}
        for 步骤 in 步骤列表:
            日志名 = {'训练': '训练', 'base': '基础测试', 'new': '新类别测试'}[步骤]
            日志路径 = 阶段目录 / '运行日志' / f"{运行['序号']:02d}-{运行['数据集']}-种子{运行['种子']}-{日志名}.txt"
            退出码, 耗时 = 模块.执行命令(运行['命令'][步骤], 源码目录, 日志路径, env)
            记录 = {
                '运行序号': 运行['序号'], '数据集': 运行['数据集'], '种子': 运行['种子'],
                '步骤': 步骤, '退出码': 退出码, '耗时秒': 耗时, '运行日志': str(日志路径),
                '恢复执行': True,
            }
            状态['命令'].append(记录)
            if 退出码 != 0:
                状态['状态'] = '失败'
                状态['恢复失败命令'] = 记录
                模块.原子写json(状态路径, 状态)
                raise RuntimeError(f'恢复命令失败，见：{日志路径}')
            if 步骤 == '训练':
                检查点 = Path(运行['检查点'])
                if not 检查点.is_file():
                    raise FileNotFoundError(f'缺少第10轮检查点：{检查点}')
                检查点SHA = 模块.sha256(检查点)
            else:
                评估目录 = 运行目录 / ('基础测试' if 步骤 == 'base' else '新类别测试')
                解析 = 模块.解析日志(评估目录 / 'log.txt')
                if list(评估目录.rglob('*.pth.tar*')):
                    raise ValueError(f'评估目录不应包含训练检查点：{评估目录}')
                解析['加载检查点SHA256'] = 检查点SHA
                指标[步骤] = 解析
            状态['已完成命令'] += 1
            模块.原子写json(状态路径, 状态)
        基础 = 指标['base']['复算准确率']
        新类 = 指标['new']['复算准确率']
        结果行.append({
            '数据集': 运行['数据集'], '种子': 运行['种子'], 'Base': 基础, 'Novel': 新类,
            'HM': 模块.调和平均(基础, 新类), '检查点SHA256': 检查点SHA,
            '基础测试日志SHA256': 指标['base']['日志SHA256'],
            '新类别测试日志SHA256': 指标['new']['日志SHA256'],
        })
    数据集行 = 模块.协议数据集[计划['协议']]
    当前摘要 = {path.as_posix(): 模块.sha256(源码目录 / path) for path in 模块.源码清单(源码目录, 数据集行)}
    if 当前摘要 != 计划['源码文件SHA256']:
        raise ValueError('执行期间源码或配置发生变化')
    if 状态['已完成命令'] != 状态['总命令']:
        raise ValueError(f"完成命令数错误：{状态['已完成命令']} / {状态['总命令']}")
    状态['状态'] = '完成'
    状态['结束时间Unix'] = time.time()
    状态['总耗时秒'] = 状态['结束时间Unix'] - 状态['开始时间Unix']
    模块.原子写json(状态路径, 状态)
    模块.写阶段结果(阶段目录, 阶段, 结果行, 状态, 计划)
    模块.写文件清单(阶段目录)
    print(json.dumps({'状态': '通过', '恢复': True, '运行数': len(运行列表), '命令数': 状态['已完成命令']}, ensure_ascii=False))
except Exception:
    if 状态.get('状态') == '恢复中':
        状态['状态'] = '失败'
        状态['结束时间Unix'] = time.time()
        模块.原子写json(状态路径, 状态)
    raise
