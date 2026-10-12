#!/usr/bin/env python3
"""Read-only SEATrack progress dashboard; --watch refreshes every 5 seconds."""
import argparse
import json
import re
import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

DEFAULT_ROOT = Path('/nas/liziyan103/seatrack/training_runs/20261011_remaining')
PATTERN = re.compile(r'\[train:\s*(\d+),\s*(\d+)\s*/\s*(\d+)\](.*)')
LABELS = {'TRAINING': '训练中', 'WAITING_GPU': '排队等待', 'VALIDATING': '检查数据中',
          'PREPARING': '准备中', 'COMPLETED': '训练及核验完成', 'FAILED': '失败',
          'VALIDATION_REUSED': '已复用数据检查'}

def read_tail(path):
    try:
        with path.open('rb') as f:
            f.seek(0, 2)
            length = f.tell()
            f.seek(max(0, length - 262144))
            return f.read().decode('utf-8', errors='replace')
    except FileNotFoundError:
        return ''

def progress(root, task, epochs):
    folder = root / task
    try:
        raw = (folder / 'status.txt').read_text()
        try:
            state = json.loads(raw)
        except ValueError:
            state = {'state': raw.split()[0] if raw.split() else 'UNKNOWN'}
    except FileNotFoundError:
        state = {'state': 'UNKNOWN'}
    matches = list(PATTERN.finditer(read_tail(folder / 'training.log')))
    checkpoints = list((folder / 'models/checkpoints' / task).glob('SEATrack_ep*.pth.tar'))
    saved_epochs = [int(m.group(1)) for p in checkpoints
                    for m in [re.search(r'_ep(\d+)\.pth\.tar$', p.name)] if m]
    saved = max(saved_epochs, default=0)
    epoch = batch = batches = 0
    metrics = ''
    if matches:
        m = matches[-1]
        epoch, batch, batches = map(int, m.group(1, 2, 3))
        metrics = m.group(4)
    fraction = (batch / batches) if batches else 0
    total = min(1.0, max(saved, (epoch - 1 + fraction) if epoch else 0) / epochs)
    if state.get('state') == 'COMPLETED':
        total = 1.0
    return state, epoch, batch, batches, fraction, total, metrics, len(checkpoints), saved

def bar(value):
    n = min(30, max(0, int(value * 30)))
    return '[' + '#' * n + '-' * (30 - n) + ']'

def render(root):
    lines = ['SEATrack 训练进度  |  ' + datetime.now(timezone(timedelta(hours=8))).strftime('%Y-%m-%d %H:%M:%S 北京时间'),
             '刷新：5 秒；Ctrl+C 退出面板（训练继续）', '=' * 68]
    for task, name, epochs in [('rgbe', 'RGB-E / VisEvent', 45), ('rgbt', 'RGB-T / LasHeR', 60)]:
        state, ep, step, steps, frac, total, metrics, count, saved = progress(root, task, epochs)
        code = state.get('state', 'UNKNOWN')
        lines.append(name + '  |  ' + LABELS.get(code, code))
        if state.get('gpus'):
            lines.append('  GPU：' + ', '.join(state['gpus']))
        if ep:
            lines.append('  最近训练记录：第 %d / %d 轮，第 %d / %d 批' % (ep, epochs, step, steps))
            lines.append('  本轮 %s %6.2f%%' % (bar(frac), 100 * frac))
        else:
            lines.append('  轮次：尚无训练批次记录 / 共 %d 轮' % epochs)
        lines.append('  总体 %s %6.2f%%' % (bar(total), 100 * total))
        values = []
        for key in ('Loss/total', 'IoU'):
            match = re.search(re.escape(key) + r':\s*([\d.eE+\-]+)', metrics)
            if match:
                values.append(key + ': ' + match.group(1))
        if values:
            lines.append('  最新指标：' + '  |  '.join(values))
        lines.append('  checkpoint 文件：%d 个；最高编号：第 %d 轮（未逐个校验内容）' % (count, saved))
        log = root / task / 'training.log'
        if log.exists():
            age = max(0, int(time.time() - log.stat().st_mtime))
            lines.append('  训练日志最近写入：%d 分 %d 秒前' % divmod(age, 60))
        if task == 'rgbt' and code == 'WAITING_GPU':
            lines.append('  等待 RGB-E 完成核验，再等待 GPU 5、6 余量满足条件')
        if state.get('error'):
            lines.append('  错误：' + str(state['error'])[:250])
        lines.append('-' * 68)
    lines.extend(['百分比按训练批次估算，不包含验证、保存及最终核验耗时。',
                  '日志每 50 批更新；显示的是最近记录，不会逐批跳动。'])
    return '\n'.join(lines)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--watch', action='store_true')
    parser.add_argument('--root', type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args()
    try:
        while True:
            screen = render(args.root)
            if args.watch and sys.stdout.isatty():
                sys.stdout.write('\033[2J\033[H')
            print(screen, flush=True)
            if not args.watch:
                break
            time.sleep(5)
    except KeyboardInterrupt:
        print('\n已退出进度面板，训练继续运行。')

if __name__ == '__main__':
    main()
