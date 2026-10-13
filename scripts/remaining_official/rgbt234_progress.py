#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Read-only RGBT234 inference progress; --watch refreshes every five seconds."""
import argparse
from datetime import datetime, timedelta, timezone
from pathlib import Path
import re
import subprocess
import time


def tail(path):
    try:
        with path.open('rb') as f:
            f.seek(0, 2)
            f.seek(max(0, f.tell()-262144))
            return f.read().decode('utf-8', errors='replace')
    except OSError:
        return ''


def render(base, chosen=None):
    header = ['RGBT234 官方权重测试进度 | '+datetime.now(timezone(timedelta(hours=8))).strftime('%m-%d %H:%M:%S 北京时间'),
              '每5秒刷新；Ctrl+C只退出面板，不停止测试', '='*72]
    root = base/'evaluation_runs/20261013_remaining/code/RGBT_workspace'
    runs = sorted(p for p in root.glob('official_rgbt234_full_*') if p.is_dir())
    run = root/chosen if chosen else (runs[-1] if runs else None)
    if run is None or not run.is_dir():
        return '\n'.join(header+['尚未找到全量测试结果目录；等待任务开始写出第一条轨迹。'])
    dataset = base/'datasets/rgbt234'
    names = {p.name for p in dataset.iterdir() if p.is_dir()} if dataset.is_dir() else set()
    result = run/'RGBT234'
    written = {p.stem for p in result.glob('*.txt') if p.is_file()}
    done = len(written & names) if names else len(written)
    total = len(names) or 234
    fraction = min(1, done/total)
    filled = int(fraction*30)
    bar = '['+'#'*filled+'-'*(30-filled)+']'
    log = base/'logs'/(run.name+'.log')
    text = tail(log)
    sequences = re.findall(r'Process sequence:\s*([A-Za-z0-9_.-]+)', text)
    fps = re.findall(r'([^\s,]+)\s*,\s*fps:([\d.eE+-]+)', text)
    try:
        exitcode = log.with_name(log.name+'.exitcode').read_text().strip()
    except OSError:
        exitcode = None
    try:
        ps = subprocess.run(['ps', '-eo', 'args'], capture_output=True,
                            text=True, timeout=2)
        running = any('test_rgbt_mgpus.py' in line and
                      re.search(r'--variants(?:=|\s+)'+re.escape(run.name)+r'(?:\s|$)', line)
                      for line in ps.stdout.splitlines()) if ps.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired):
        running = None
    if exitcode is not None:
        state = ('推理正常结束，待轨迹验收和指标计算' if exitcode == '0' and done == total
                 else '需要检查：退出码%s，已写出%d/%d条' % (exitcode, done, total))
    elif running:
        state = '正在运行'
    elif running is None:
        state = '进程状态未知，尚无退出码'
    else:
        state = '未发现测试进程，尚无退出码，请检查日志'
    lines = header+['任务：'+run.name, '状态：'+state,
                    '总体 %s %6.2f%%' % (bar, fraction*100),
                    '已写出轨迹：%d / %d 个序列' % (done, total)]
    if len(runs) > 1 and not chosen:
        lines.append('发现多个全量任务，当前显示名称时间戳最新的一次。')
    if sequences:
        current = sequences[-1]
        lines.append('最近处理序列：'+current+('（结果已写出）' if current in written else '（尚未写出结果）'))
    if fps:
        lines.append('最近完成：%s | %.2f FPS' % (fps[-1][0], float(fps[-1][1])))
    try:
        age = max(0, int(time.time()-log.stat().st_mtime))
        lines.append('日志最近更新：%d分%d秒前' % divmod(age, 60))
    except OSError:
        lines.append('日志暂未找到')
    if names and written-names:
        lines.append('发现额外结果文件：'+str(len(written-names)))
    lines += ['-'*72, '日志：'+str(log),
              '进度按序列计数；各序列长度不同，不代表已用时间比例。',
              '单序列内部不逐帧更新；轨迹写完后进度才增加。',
              '100%表示轨迹已写出，完整性验收和MPR/MSR需另行完成。']
    if exitcode not in (None, '0'):
        lines += ['最近日志：']+text.splitlines()[-6:]
    return '\n'.join(lines)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--base', type=Path, default=Path('/nas/liziyan103/seatrack'))
    p.add_argument('--run', help='Exact official_rgbt234_full_* directory name')
    p.add_argument('--watch', action='store_true')
    args = p.parse_args()
    try:
        while True:
            print(('\033[2J\033[H' if args.watch else '')+render(args.base, args.run), flush=True)
            if not args.watch:
                break
            time.sleep(5)
    except KeyboardInterrupt:
        print('\n已退出面板，测试任务不受影响。')


if __name__ == '__main__':
    main()
