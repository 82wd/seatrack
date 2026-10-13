#!/usr/bin/env python3
"""Read-only VOT22 multistart progress monitor (Python 3.8+)."""
import argparse
from datetime import datetime
from pathlib import Path
import time


def plan(work):
    root = work / 'sequences'
    names = (root / 'list.txt').read_text().split()
    jobs = {}
    for name in names:
        seq = root / name
        n = len((seq / 'groundtruth.txt').read_text().splitlines())
        anchors = [float(x) for x in (seq / 'anchor.value').read_text().splitlines()]
        if len(anchors) != n:
            raise ValueError('anchor/标注长度不一致: ' + name)
        jobs[name] = [(i, n-i if v > 0 else i+1) for i, v in enumerate(anchors) if v != 0]
        if not jobs[name]:
            raise ValueError('没有起点: ' + name)
    if len(names) != 127 or len(set(names)) != 127:
        raise ValueError('全量工作区应包含127个唯一序列')
    return jobs


def lines_cached(path, cache):
    try:
        stat = path.stat()
        key = (stat.st_size, stat.st_mtime_ns)
        if cache.get(path, (None,))[0] != key:
            with path.open('rb') as f:
                count = sum(1 for _ in f)
            cache[path] = (key, count)
        return cache[path][1]
    except OSError:
        return -1


def progress(work, jobs, cache):
    done_seq = done_jobs = done_frames = 0
    latest = None
    partial = []
    for name, anchors in jobs.items():
        folder = work / 'results/rgbd/baseline' / name
        done = 0
        for index, length in anchors:
            stem = '{}_{:08d}'.format(name, index)
            txt, binary = folder / (stem+'.txt'), folder / (stem+'.bin')
            timing = folder / (stem+'_time.value')
            try:
                trajectory = txt if txt.exists() else binary
                stat = trajectory.stat()
                valid = stat.st_size > 0 and (trajectory.suffix == '.bin' or lines_cached(txt, cache) == length)
                valid = valid and lines_cached(timing, cache) == length
                if valid:
                    done += 1
                    done_frames += length
                    item = (stat.st_mtime, name, index)
                    if latest is None or item > latest:
                        latest = item
            except OSError:
                pass
        done_jobs += done
        done_seq += int(done == len(anchors))
        if 0 < done < len(anchors):
            partial.append('{} {}/{}起点'.format(name, done, len(anchors)))
    return done_seq, done_jobs, done_frames, latest, partial


def running(work):
    pids = []
    for proc in Path('/proc').glob('[0-9]*'):
        try:
            args = (proc / 'cmdline').read_bytes().split(b'\0')
            if b'vot' in args and b'evaluate' in args and (proc / 'cwd').resolve() == work.resolve():
                pids.append(proc.name)
        except OSError:
            pass
    return pids


def render(base, work, jobs, cache, log):
    ds, dj, df, latest, partial = progress(work, jobs, cache)
    nj = sum(len(v) for v in jobs.values())
    nf = sum(n for v in jobs.values() for _, n in v)
    pids = running(work)
    code = None
    if log:
        try:
            code = Path(str(log)+'.exitcode').read_text().strip()
        except OSError:
            pass
    if pids:
        state = '运行中，评测PID: ' + ', '.join(pids)
    elif code == '0':
        state = '评测退出码0；待完整验收和指标计算' if ds == len(jobs) else '退出码0但结果计数未齐，请检查'
    elif code is not None:
        state = '评测已退出，退出码: ' + code
    else:
        state = '未发现该工作区评测进程，尚无退出码，请查看日志'
    def bar(n, total):
        fill = int(30*n/total)
        return '[{}{}] {:6.2f}%'.format('#'*fill, '-'*(30-fill), 100*n/total)
    out = ['VOT22-RGBD | '+datetime.now().astimezone().isoformat(timespec='seconds'),
           '='*76, '状态: '+state,
           '序列进度 '+bar(ds, len(jobs))+'  {}/{}'.format(ds, len(jobs)),
           '起点进度 '+bar(dj, nj)+'  {}/{}'.format(dj, nj),
           '已写出起点对应帧数: {}/{}（包含正反向重复跟踪帧）'.format(df, nf)]
    if partial:
        out.append('部分完成: '+'；'.join(partial[:5]))
    if latest:
        out.append('最近写出: {}，起点 {}，{} 秒前'.format(latest[1], latest[2], max(0, int(time.time()-latest[0]))))
    out.append('当前具体序列: 常规日志未提供，以上显示最近落盘结果')
    if log:
        try:
            age = max(0, int(time.time()-log.stat().st_mtime))
            out.append('日志最近更新: {}分{}秒前'.format(*divmod(age, 60)))
        except OSError:
            pass
        out.append('日志: '+str(log))
    else:
        out.append('尚未找到全量日志')
    out += ['-'*76, '每10秒刷新；按多起点轨迹及time文件行数计数，不采用日志中的百分比。',
            '单个起点完成落盘后更新；进度不等于耗时比例，100%不代表指标已计算。',
            '仅检查文件及行数，二进制轨迹未解码；最终仍需VOT工具验收。',
            '只读监控；Ctrl+C退出面板不会停止评测。']
    return '\n'.join(out)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base', type=Path, default=Path('/nas/liziyan103/seatrack'))
    p.add_argument('--watch', action='store_true')
    p.add_argument('--log', type=Path, help='Pin to a specific full evaluation log')
    args = p.parse_args()
    work = args.base / 'evaluation_runs/20261013_remaining/vot22/workspace'
    jobs, cache = plan(work), {}
    logs = sorted((args.base / 'logs').glob('official_vot22_full_*.log'))
    log = args.log or (logs[-1] if logs else None)
    try:
        while True:
            print(('\033[2J\033[H' if args.watch else '')+render(args.base, work, jobs, cache, log), flush=True)
            if not args.watch:
                break
            time.sleep(10)
    except KeyboardInterrupt:
        print('\n面板已退出，评测不受影响。')


if __name__ == '__main__':
    main()
