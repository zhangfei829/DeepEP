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


def parse_one_run(log_dir: Path, tag: str):
    """Scan all rank logs for the given tag; return dict[op] -> list of per-rank dicts."""
    per_op = {op: [] for op in OP_NAMES}
    log_paths = sorted(log_dir.glob(f'{tag}.rank*.log'))
    for p in log_paths:
        try:
            text = p.read_text(errors='replace')
        except OSError:
            continue
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
    return per_op, len(log_paths)


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


def print_markdown(runs, summaries):
    """Print a compact dispatch+combine summary, sorted by (ep, tokens, topk, experts)."""
    rows = []
    for run, summary in zip(runs, summaries):
        rows.append((run, summary))

    def key(rs):
        run, _ = rs
        try:
            return (int(run.get('ep') or 0),
                    int(run.get('tokens') or 0),
                    int(run.get('topk') or 0),
                    int(run.get('experts') or 0))
        except ValueError:
            return (0, 0, 0, 0)

    rows.sort(key=key)
    print()
    # Header. API columns (host wall-time) follow the kernel-time columns,
    # mirroring the Hybrid_ep / NCCL EP statistic layout (kernel | API).
    print('| EP | tokens | topk | exp | op       | SO GB/s (min/avg/max) | SU GB/s (min/avg/max) | time us (min/avg/max) | copy GB/s | api SU GB/s (min/avg/max) | api time us (min/avg/max) |')
    print('|----|--------|------|-----|----------|-----------------------|-----------------------|-----------------------|-----------|---------------------------|---------------------------|')
    for run, summary in rows:
        for op in SUMMARY_OPS:
            s = summary.get(op)
            if s is None:
                print(f'| {run.get("ep","?"):>2} | {run.get("tokens","?"):>6} | '
                      f'{run.get("topk","?"):>4} | {run.get("experts","?"):>3} | '
                      f'{op:<8} | (no data)              |                       |                       |           |                           |                           |')
                continue
            if 'api_us_avg' in s:
                api_su = f'{s["api_su_min"]:>3} / {s["api_su_avg"]:>5.1f} / {s["api_su_max"]:>3}'
                api_us = f'{s["api_us_min"]:>6.2f} / {s["api_us_avg"]:>6.2f} / {s["api_us_max"]:>6.2f}'
            else:
                api_su = '(no data)'
                api_us = '(no data)'
            print(
                f'| {run.get("ep",""):>2} | {run.get("tokens",""):>6} | {run.get("topk",""):>4} | {run.get("experts",""):>3} | '
                f'{op:<8} | '
                f'{s["so_min"]:>3} / {s["so_avg"]:>5.1f} / {s["so_max"]:>3} | '
                f'{s["su_min"]:>3} / {s["su_avg"]:>5.1f} / {s["su_max"]:>3} | '
                f'{s["us_min"]:>6.2f} / {s["us_avg"]:>6.2f} / {s["us_max"]:>6.2f} | '
                f'{s["copy_gbs_avg"]:>6.1f}    | '
                f'{api_su:<25} | {api_us:<25} |')


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
    for run in runs:
        per_op, n_logs = parse_one_run(log_dir, run['tag'])
        if n_logs == 0:
            print(f'[parse] WARN: no logs for run_tag={run["tag"]}', file=sys.stderr)
        summaries.append(aggregate(per_op))

    csv_path = write_csv(log_dir, args.tag, runs, summaries)
    print(f'[parse] wrote {csv_path}')
    print_markdown(runs, summaries)


if __name__ == '__main__':
    main()
