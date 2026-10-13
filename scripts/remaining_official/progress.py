#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Read-only download dashboard. No network, downloads, or process mutations."""
import argparse
from collections import deque
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import subprocess
import time


def read(path):
    try:
        return path.read_text().strip()
    except OSError:
        return ''


def stat(path):
    try:
        s = path.stat()
        return s.st_size, s.st_mtime
    except OSError:
        return 0, 0


def bar(done, total):
    fraction = min(1, max(0, done / total)) if total else 0
    n = int(fraction * 30)
    return '[' + '#' * n + '-' * (30-n) + '] %6.1f%%' % (fraction*100)


def render(base, history):
    run = base / 'evaluation_runs/20261013_remaining'
    cache = base / 'downloads/vot2022_rgbd'
    now = time.time()
    states = {'DATA_READY': '数据准备完成', 'DOWNLOADING': '下载 / 解压中',
              'EXTRACTING': '解压中'}
    rgb = read(run/'rgbt234/status.txt') or '暂无状态'
    state = read(run/'vot22/status.txt') or '暂无状态'
    try:
        meta = json.loads(read(run/'vot22/description.json'))
    except ValueError:
        meta = {'sequences': []}
    total = received = archives = completed = 0
    partials = []
    for seq in meta['sequences']:
        name = seq['name']
        completed += (run/'vot22'/(name+'.complete')).exists()
        files = [(name+'_annotations.zip', seq['annotations']['compressed'])]
        files += [(name+'_'+c+'.zip', info['compressed'])
                  for c, info in seq['channels'].items()]
        for filename, expected in files:
            size, _ = stat(cache/filename)
            part, modified = stat(cache/(filename+'.part'))
            total += expected
            # Each archive counted once; metadata sizes can differ slightly.
            received += min(expected, max(size, part))
            archives += bool(size)
            if part and not size:
                partials.append((modified, filename, part, expected))
    tick = time.monotonic()
    if history and received < history[-1][1]:
        history.clear()
    history.append((tick, received))
    while len(history) > 2 and tick-history[0][0] > 60:
        history.popleft()
    elapsed = tick-history[0][0]
    speed = (received-history[0][1])/elapsed if elapsed > 0 else 0
    try:
        cmd = subprocess.run(['pgrep', '-af', '[p]repare.py vot22'],
                             capture_output=True, text=True, timeout=2)
        running = '存在' if cmd.returncode == 0 else ('未发现' if cmd.returncode == 1 else '未知')
    except (OSError, subprocess.TimeoutExpired):
        running = '未知'
    lines = ['SEATrack 数据下载进度 | '+datetime.now(timezone(timedelta(hours=8))).strftime('%m-%d %H:%M:%S 北京时间'),
             '每5秒刷新；Ctrl+C只退出面板，不停止下载', '='*72,
             'RGBT234 | '+states.get(rgb, rgb[:150]), '-'*72,
             'VOT22-RGBD | '+states.get(state, state[:180]),
             '准备进程：'+running+'（状态文件可能保留旧状态）',
             '下载量估算 '+bar(received, total),
             '  %.2f / %.2f GiB；已落盘ZIP：%d' % (received/2**30, total/2**30, archives),
             '序列准备   '+bar(completed, len(meta['sequences'])),
             '  已下载并解压：%d / %d 个序列' % (completed, len(meta['sequences']))]
    if elapsed >= 4:
        lines.append('最近约%d秒平均新增：%.2f MiB/s' % (elapsed, speed/2**20))
        if speed > 0 and received < total and running == '存在':
            minutes = (total-received)/speed/60
            lines.append('按近期速度估算剩余下载：约%.0f分钟（不含解压、校验及重试）' % minutes)
    else:
        lines.append('速度：采样中……')
    lines += ['-'*72, '最近更新的未完成文件（不一定仍在传输）：']
    for modified, filename, part, expected in sorted(partials, reverse=True)[:4]:
        lines.append('  %s\n    %.1f / %.1f MiB；%d秒前更新' %
                     (filename, part/2**20, expected/2**20, max(0, now-modified)))
    if not partials:
        lines.append('  无 .part 文件')
    lines += ['='*72, '下载量按官方清单估算；100%不等于校验完成，以DATA_READY为准。']
    return '\n'.join(lines)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--base', type=Path, default=Path('/nas/liziyan103/seatrack'))
    p.add_argument('--watch', action='store_true')
    args = p.parse_args()
    history = deque()
    try:
        while True:
            text = render(args.base, history)
            print(('\033[2J\033[H' if args.watch else '')+text, flush=True)
            if not args.watch:
                break
            time.sleep(5)
    except KeyboardInterrupt:
        print('\n面板已退出，下载任务不受影响。')


if __name__ == '__main__':
    main()
