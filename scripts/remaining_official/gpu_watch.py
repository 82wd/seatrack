#!/usr/bin/env python3
"""Read-only GPU memory monitor, Python 3.8+, no third-party dependencies."""
import argparse
import csv
from datetime import datetime
import io
import subprocess
import sys
import time


def parse_rows(output):
    rows = []
    for fields in csv.reader(io.StringIO(output)):
        if not fields:
            continue
        if len(fields) != 5:
            raise ValueError('Unexpected nvidia-smi output: ' + repr(fields))
        index, uuid, total, free, util = [s.strip() for s in fields]
        total, free = int(total), int(free)
        if total <= 0 or not 0 <= free <= total:
            raise ValueError('Invalid memory reading')
        rows.append(dict(index=int(index), uuid=uuid, total=total, free=free,
                         util=util))
    if not rows or len({r['uuid'] for r in rows}) != len(rows):
        raise ValueError('Missing or duplicate GPU readings')
    return sorted(rows, key=lambda r: r['index'])


def update_counts(rows, previous, threshold):
    return {r['uuid']: previous.get(r['uuid'], 0) + 1
            if r['free'] >= threshold else 0 for r in rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--min-free', type=int, default=4096, help='Free MiB threshold')
    parser.add_argument('--interval', type=float, default=10, help='Seconds between queries')
    parser.add_argument('--samples', type=int, default=3, help='Consecutive qualifying samples')
    parser.add_argument('--once', action='store_true', help='Show one reading and exit')
    args = parser.parse_args()
    if args.min_free <= 0 or args.interval <= 0 or args.samples <= 0:
        parser.error('Threshold, interval and sample count must be positive')
    counts, alerted = {}, set()
    try:
        while True:
            lines = [
                'SEATrack | GPU 显存监控 | ' + datetime.now().astimezone().isoformat(timespec='seconds'),
                '门槛: {} MiB | 每 {:g} 秒查询 | 连续 {} 次达标后提醒'.format(
                    args.min_free, args.interval, args.samples),
                '=' * 76,
            ]
            failed, bell = False, False
            try:
                result = subprocess.run(
                    ['nvidia-smi', '--query-gpu=index,uuid,memory.total,memory.free,utilization.gpu',
                     '--format=csv,noheader,nounits'],
                    capture_output=True, text=True, timeout=15, check=True)
                rows = parse_rows(result.stdout)
                counts = update_counts(rows, counts, args.min_free)
                ready = [r for r in rows if counts[r['uuid']] >= args.samples]
                ready_ids = {r['uuid'] for r in ready}
                bell = bool(ready_ids - alerted)
                alerted = ready_ids
                for row in rows:
                    n = min(args.samples, counts[row['uuid']])
                    status = '达到筛选门槛' if n == args.samples else (
                        '确认中 {}/{}'.format(n, args.samples) if n else '等待显存')
                    filled = int(20 * row['free'] / row['total'])
                    bar = '#' * filled + '-' * (20 - filled)
                    lines.append('GPU {} [{}] 空闲 {:5d}/{:5d} MiB | 利用率 {}% | {}'.format(
                        row['index'], bar, row['free'], row['total'], row['util'], status))
                lines.append('-' * 76)
                if ready:
                    best = max(ready, key=lambda r: r['free'])
                    lines.append('可供单序列测试筛选: GPU {}，当前空闲 {} MiB'.format(best['index'], best['free']))
                else:
                    lines.append('等待某张卡连续达到显存门槛……')
            except (OSError, ValueError, subprocess.SubprocessError) as exc:
                counts, alerted = {}, set()
                failed = True
                lines.append('查询失败，已清空连续达标计数: ' + str(exc))
            lines += ['只监控，不启动测试、不终止进程、不预占显存。',
                      '门槛不是实测最低需求；读数可能变化，启动测试前仍需确认。',
                      'Ctrl+C 退出监控；tmux 中 Ctrl+B 再按 D 可离开面板。']
            if sys.stdout.isatty():
                print('\033[2J\033[H', end='')
                if bell:
                    print('\a', end='')
            print('\n'.join(lines), flush=True)
            if args.once:
                return 1 if failed else 0
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print('\n显存监控已退出。')
        return 0


if __name__ == '__main__':
    sys.exit(main())
