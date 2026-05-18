"""
Aggregate DeepEP V2 sweep logs into a single CSV and print a comparison table.

Inputs:
    --log-dir DIR   directory containing $tag.rank{NN}.log files (default: $DEEPEP_LOG_DIR)
    --tag TAG       sweep tag prefix, matches $SWEEP_TAG used by tray_deepep_sweep.sh
                    (also requires $tag.index.tsv to discover per-run params)

Outputs:
    <log-dir>/<tag>.summary.csv   one row per (run, op), columns:
        run_tag, ep, tokens, topk, experts, fp8, op,
        so_gbs_min, so_gbs_avg, so_gbs_max,
        su_gbs_min, su_gbs_avg, su_gbs_max,
        time_us_min, time_us_avg, time_us_max,
        copy_gbs_avg, copy_us_avg, num_ranks_seen

A markdown-style table is also printed to stdout so the operator gets an instant
read after the sweep.

Tested patterns (from tests/elastic/test_ep.py):
    "   * EP:   0/16 | dispatch:    90 GB/s (SO),    93 GB/s (SU), 123.456 us, 1234 bytes | copy: 200 GB/s, 5.000 us"
    "   - EP:   0/16 | expanded dispatch: ..."
    "   # EP:   0/16 | cached dispatch:   ..."
    "   @ EP:   0/16 | combine:           ..."
    "   + EP:   0/16 | reduced combine:   ..."
"""

import argparse
import csv
import os
import re
import statistics
import sys
from pathlib import Path

OP_NAMES = ('dispatch', 'expanded dispatch', 'cached dispatch', 'combine', 'reduced combine')
SUMMARY_OPS = ('dispatch', 'combine')  # the two that we print in the markdown summary

# NOTE: alternation order matters; longest variants come first so that
# "expanded dispatch" isn't shadowed by "dispatch", etc.
LINE_RE = re.compile(
    r'EP:\s*(?P<rank>\d+)\s*/\s*(?P<world>\d+)\s*\|\s*'
    r'(?P<op>expanded dispatch|cached dispatch|reduced combine|dispatch|combine)\s*:\s*'
    r'(?P<so>\d+)\s*GB/s\s*\(SO\),\s*'
    r'(?P<su>\d+)\s*GB/s\s*\(SU\),\s*'
    r'(?P<us>[\d.]+)\s*us,\s*'
    r'\d+\s*bytes\s*\|\s*'
    r'(?:reduce|copy):\s*(?P<copy_gbs>\d+)\s*GB/s,\s*(?P<copy_us>[\d.]+)\s*us'
    # Optional API (host-wall-time) trailer; emitted by test_ep.py when
    # bench_api_walltime is enabled. Modeled after Hybrid_ep / NCCL EP
    # "API GB/s" column, captures end-to-end (host + device) bandwidth.
    r'(?:\s*\|\s*api:\s*(?P<api_so>\d+)\s*GB/s\s*\(SO\s*api\),\s*'
    r'(?P<api_su>\d+)\s*GB/s\s*\(SU\s*api\),\s*(?P<api_us>[\d.]+)\s*us)?'
)


def _safe_stat(fn, xs):
    return float(fn(xs)) if xs else float('nan')


META_RE = {
    'ranks':   re.compile(r'^\s*>\s*Ranks:\s*(.+?)\s*$', re.M),
    'experts': re.compile(r'^\s*>\s*Experts:\s*(.+?)\s*$', re.M),
    'tokens':  re.compile(r'^\s*>\s*Tokens:\s*(.+?)\s*$', re.M),
    'sms':     re.compile(r'^\s*>\s*#SM:\s*(\d+),\s*#QPs:\s*(.+?)\s*$', re.M),
}

# `> dispatch kernel: 64 SMs, 4 notify + 8 dispatch = 12 warps/block (384 threads/block)`
# `> combine  kernel: 64 SMs, 16 warps              = 16 warps/block (512 threads/block)`
WARP_RE = re.compile(
    r'^\s*>\s*(?P<kind>dispatch|hybrid_dispatch|combine)\s*kernel:\s*'
    r'(?P<sms>\d+)\s*SMs,\s*'
    r'(?P<breakdown>.+?)\s*=\s*'
    r'(?P<total>\d+)\s*warps/block\s*\(\s*(?P<threads>\d+)\s*threads/block\)',
    re.M)


def parse_one_run(log_dir: Path, tag: str):
    """Scan all rank logs for the given tag; return (per_op rows, n_logs, meta)."""
    per_op = {op: [] for op in OP_NAMES}
    meta = {}
    log_paths = sorted(log_dir.glob(f'{tag}.rank*.log'))
    for p in log_paths:
        try:
            text = p.read_text(errors='replace')
        except OSError:
            continue
        if not meta:
            for k, regex in META_RE.items():
                mm = regex.search(text)
                if mm:
                    if k == 'sms':
                        meta['num_sms'] = mm.group(1)
                        meta['num_qps'] = mm.group(2)
                    else:
                        meta[k] = mm.group(1)
        # warp metadata (printed by tests/elastic/test_ep.py from kernel template args)
        for mm in WARP_RE.finditer(text):
            kind = mm.group('kind')
            meta.setdefault('warps', {})[kind] = {
                'sms': int(mm.group('sms')),
                'breakdown': mm.group('breakdown').strip(),
                'total': int(mm.group('total')),
                'threads': int(mm.group('threads')),
            }
        for m in LINE_RE.finditer(text):
            op = m.group('op')
            row = {
                'rank': int(m.group('rank')),
                'world': int(m.group('world')),
                'so_gbs': int(m.group('so')),
                'su_gbs': int(m.group('su')),
                'time_us': float(m.group('us')),
                'copy_gbs': int(m.group('copy_gbs')),
                'copy_us': float(m.group('copy_us')),
            }
            # Optional api: <SO> GB/s (SO api), <SU> GB/s (SU api), <us> us
            if m.group('api_us'):
                row['api_so_gbs'] = int(m.group('api_so'))
                row['api_su_gbs'] = int(m.group('api_su'))
                row['api_us'] = float(m.group('api_us'))
            per_op[op].append(row)
    return per_op, len(log_paths), meta


def aggregate(per_op_rows):
    """Reduce per-rank rows to summary stats per op."""
    out = {}
    for op, rows in per_op_rows.items():
        if not rows:
            out[op] = None
            continue
        so = [r['so_gbs'] for r in rows]
        su = [r['su_gbs'] for r in rows]
        us = [r['time_us'] for r in rows]
        copy_gbs = [r['copy_gbs'] for r in rows]
        copy_us = [r['copy_us'] for r in rows]
        api_so = [r['api_so_gbs'] for r in rows if 'api_so_gbs' in r]
        api_su = [r['api_su_gbs'] for r in rows if 'api_su_gbs' in r]
        api_us = [r['api_us'] for r in rows if 'api_us' in r]
        d = {
            'so_min': min(so), 'so_avg': _safe_stat(statistics.mean, so), 'so_max': max(so),
            'su_min': min(su), 'su_avg': _safe_stat(statistics.mean, su), 'su_max': max(su),
            'us_min': min(us), 'us_avg': _safe_stat(statistics.mean, us), 'us_max': max(us),
            'copy_gbs_avg': _safe_stat(statistics.mean, copy_gbs),
            'copy_us_avg': _safe_stat(statistics.mean, copy_us),
            'n': len(rows),
        }
        # API (host-wall-time, includes Python/launch overhead). Populated only
        # if test_ep.py emitted the trailing `api: ...` segment.
        if api_us:
            d['api_so_min'] = min(api_so); d['api_so_avg'] = _safe_stat(statistics.mean, api_so); d['api_so_max'] = max(api_so)
            d['api_su_min'] = min(api_su); d['api_su_avg'] = _safe_stat(statistics.mean, api_su); d['api_su_max'] = max(api_su)
            d['api_us_min'] = min(api_us); d['api_us_avg'] = _safe_stat(statistics.mean, api_us); d['api_us_max'] = max(api_us)
            d['api_n'] = len(api_us)
        out[op] = d
    return out


def load_index(log_dir: Path, tag: str):
    """Read the index.tsv emitted by tray_deepep_sweep.sh; return list of run dicts."""
    idx = log_dir / f'{tag}.index.tsv'
    if not idx.is_file():
        print(f'[parse] WARN: index file not found: {idx}; will scan all *.rank00.log instead', file=sys.stderr)
        runs = []
        for p in sorted(log_dir.glob(f'{tag}.*.rank00.log')):
            run_tag = p.name[:-len('.rank00.log')]
            runs.append({'tag': run_tag, 'ep': '', 'tokens': '', 'topk': '', 'experts': '', 'fp8': ''})
        return runs

    runs = []
    with open(idx) as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            runs.append(row)
    return runs


def write_csv(log_dir: Path, tag: str, runs, summaries):
    out = log_dir / f'{tag}.summary.csv'
    with open(out, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow([
            'run_tag', 'ep', 'tokens', 'topk', 'experts', 'fp8', 'op',
            'so_gbs_min', 'so_gbs_avg', 'so_gbs_max',
            'su_gbs_min', 'su_gbs_avg', 'su_gbs_max',
            'time_us_min', 'time_us_avg', 'time_us_max',
            'copy_gbs_avg', 'copy_us_avg', 'num_ranks_seen',
            # API (host wall time)
            'api_so_gbs_min', 'api_so_gbs_avg', 'api_so_gbs_max',
            'api_su_gbs_min', 'api_su_gbs_avg', 'api_su_gbs_max',
            'api_time_us_min', 'api_time_us_avg', 'api_time_us_max',
            'api_num_ranks_seen',
        ])
        for run, summary in zip(runs, summaries):
            for op in OP_NAMES:
                s = summary.get(op)
                if s is None:
                    continue
                row = [
                    run['tag'], run.get('ep', ''), run.get('tokens', ''),
                    run.get('topk', ''), run.get('experts', ''), run.get('fp8', ''),
                    op,
                    s['so_min'], f'{s["so_avg"]:.1f}', s['so_max'],
                    s['su_min'], f'{s["su_avg"]:.1f}', s['su_max'],
                    f'{s["us_min"]:.3f}', f'{s["us_avg"]:.3f}', f'{s["us_max"]:.3f}',
                    f'{s["copy_gbs_avg"]:.1f}', f'{s["copy_us_avg"]:.3f}',
                    s['n'],
                ]
                if 'api_us_avg' in s:
                    row += [
                        s['api_so_min'], f'{s["api_so_avg"]:.1f}', s['api_so_max'],
                        s['api_su_min'], f'{s["api_su_avg"]:.1f}', s['api_su_max'],
                        f'{s["api_us_min"]:.3f}', f'{s["api_us_avg"]:.3f}', f'{s["api_us_max"]:.3f}',
                        s['api_n'],
                    ]
                else:
                    row += ['', '', '', '', '', '', '', '', '', '']
                w.writerow(row)
    return out


def print_markdown(runs, summaries, metas):
    """Print a compact dispatch+combine summary, sorted by (ep, tokens, topk, experts)."""
    rows = []
    for run, summary, meta in zip(runs, summaries, metas):
        rows.append((run, summary, meta))

    def key(rs):
        run, _, _ = rs
        try:
            return (int(run.get('ep') or 0),
                    int(run.get('tokens') or 0),
                    int(run.get('topk') or 0),
                    int(run.get('experts') or 0))
        except ValueError:
            return (0, 0, 0, 0)

    rows.sort(key=key)
    print()
    # Legend (Chinese)
    print('图例:')
    print('  Scale-Up = NVL72 内 NVLink 带宽 (GB/s) = scaleup_recv_bytes / time。本测试所有 rank 都在同一 NVL72 NVLink 域内,流量全走 Scale-Up。')
    print('  kernel   = 仅设备侧 CUDA kernel 执行时间 (kineto)。GB/s = bytes / kernel_us。')
    print('  api      = 端到端 host wall time,包含 Python 调用 + kernel launch + sync。对应 Hybrid_ep / NCCL EP 报表里的 "API GB/s"。GB/s = bytes / api_us。')
    print('  单元格    = min / avg / max,跨所有 rank 聚合。')
    print()

    # Markdown table shows AVG only (cleaner). min/max still in csv output.
    W_SU       = 8    # "688.4"
    W_US       = 10   # "1130.06"
    W_COPY     = 9    # "4607.8"
    W_API_BW   = 8
    W_API_US   = 10

    def hdr(name, w):
        return f' {name:<{w}} '

    W_WARPS = 16  # "12=4n+4s+4f" hybrid worst case
    header = (
        '|' + hdr('EP', 2)
        + '|' + hdr('tokens', 6)
        + '|' + hdr('topk', 4)
        + '|' + hdr('exp', 3)
        + '|' + hdr('exp/rank', 8)
        + '|' + hdr('SMs', 4)
        + '|' + hdr('op', 8)
        + '|' + hdr('warps/blk', W_WARPS)
        + '|' + hdr('kernel Scale-Up GB/s', W_SU)
        + '|' + hdr('kernel us', W_US)
        + '|' + hdr('copy GB/s', W_COPY)
        + '|' + hdr('api Scale-Up GB/s', W_API_BW)
        + '|' + hdr('api us', W_API_US)
        + '|'
    )
    sep = '|' + '|'.join('-' * (len(c)) for c in header[1:-1].split('|')) + '|'
    print(header)
    print(sep)

    def fmt_triple(lo, avg, hi, fmt='>5.1f'):
        return f'{lo:>5.1f} / {avg:{fmt}} / {hi:>5.1f}' if isinstance(lo, float) else f'{lo:>5} / {avg:>5.1f} / {hi:>5}'

    for run, summary, meta in rows:
        try:
            ep_i = int(run.get('ep') or 0)
            exp_i = int(run.get('experts') or 0)
            exp_per_rank = (exp_i // ep_i) if ep_i > 0 else 0
        except ValueError:
            exp_per_rank = 0
        num_sms = meta.get('num_sms', '?')
        warps_meta = meta.get('warps', {}) or {}

        def _short_breakdown(b: str) -> str:
            # "4 notify + 8 dispatch" -> "4n+8d"; same for scaleout/forward/warps
            short_map = (('notify', 'n'), ('dispatch', 'd'), ('scaleout', 's'),
                         ('forward', 'f'), ('warps', 'w'))
            out = b
            for long, s in short_map:
                out = out.replace(' ' + long, s)
            return out.replace(' ', '')

        for op in SUMMARY_OPS:
            if op == 'dispatch':
                w = warps_meta.get('hybrid_dispatch') or warps_meta.get('dispatch')
            else:
                w = warps_meta.get('combine')
            warps_cell = f'{w["total"]}={_short_breakdown(w["breakdown"])}' if w else '?'

            s = summary.get(op)
            if s is None:
                cells = ['(no data)'] * 5
                row = (
                    '|' + f' {str(run.get("ep","?")):>2} '
                    + '|' + f' {str(run.get("tokens","?")):>6} '
                    + '|' + f' {str(run.get("topk","?")):>4} '
                    + '|' + f' {str(run.get("experts","?")):>3} '
                    + '|' + f' {exp_per_rank if exp_per_rank else "?":>8} '
                    + '|' + f' {num_sms:>4} '
                    + '|' + f' {op:<8} '
                    + '|' + f' {warps_cell:<{W_WARPS}} '
                    + ''.join('|' + f' {c:<{w_}} ' for c, w_ in zip(cells, (W_SU, W_US, W_COPY, W_API_BW, W_API_US)))
                    + '|'
                )
                print(row)
                continue

            su_cell  = f'{s["su_avg"]:>5.1f}'
            us_cell  = f'{s["us_avg"]:>7.2f}'
            cp_cell  = f'{s["copy_gbs_avg"]:>6.1f}'

            if 'api_us_avg' in s:
                api_su_cell = f'{s["api_su_avg"]:>5.1f}'
                api_us_cell = f'{s["api_us_avg"]:>7.2f}'
            else:
                api_su_cell = api_us_cell = '(no data)'

            row = (
                '|' + f' {str(run.get("ep","")):>2} '
                + '|' + f' {str(run.get("tokens","")):>6} '
                + '|' + f' {str(run.get("topk","")):>4} '
                + '|' + f' {str(run.get("experts","")):>3} '
                + '|' + f' {exp_per_rank if exp_per_rank else "":>8} '
                + '|' + f' {num_sms:>4} '
                + '|' + f' {op:<8} '
                + '|' + f' {warps_cell:<{W_WARPS}} '
                + '|' + f' {su_cell:<{W_SU}} '
                + '|' + f' {us_cell:<{W_US}} '
                + '|' + f' {cp_cell:<{W_COPY}} '
                + '|' + f' {api_su_cell:<{W_API_BW}} '
                + '|' + f' {api_us_cell:<{W_API_US}} '
                + '|'
            )
            print(row)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--log-dir', default=os.environ.get('DEEPEP_LOG_DIR', '/home/fizhang/deepep_logs'),
                    help='directory containing $tag.rank*.log files (default: $DEEPEP_LOG_DIR or /home/fizhang/deepep_logs)')
    ap.add_argument('--tag', required=True, help='sweep tag prefix (matches SWEEP_TAG)')
    args = ap.parse_args()

    log_dir = Path(args.log_dir)
    if not log_dir.is_dir():
        sys.exit(f'[parse] log dir not found: {log_dir}')

    runs = load_index(log_dir, args.tag)
    if not runs:
        sys.exit(f'[parse] no runs found for tag={args.tag} in {log_dir}')

    summaries = []
    metas = []
    for run in runs:
        per_op, n_logs, meta = parse_one_run(log_dir, run['tag'])
        if n_logs == 0:
            print(f'[parse] WARN: no logs for run_tag={run["tag"]}', file=sys.stderr)
        summaries.append(aggregate(per_op))
        metas.append(meta)

    csv_path = write_csv(log_dir, args.tag, runs, summaries)
    print(f'[parse] wrote {csv_path}')
    print_markdown(runs, summaries, metas)


if __name__ == '__main__':
    main()
