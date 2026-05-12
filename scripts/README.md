# DeepEP V2 perf sweep on GB300 4-BAY trays

One-button DeepEP V2 dispatch/combine benchmark across EP={4,8,16} trays.
Mirrors the structure of the NCCL EP `jumphost_ep16_nocopy_compare.sh` pipeline,
adapted for DeepEP V2 (NCCL Gin backend, FP8 dispatch by default).

## Layout

| File | Run from | Purpose |
|---|---|---|
| `jumphost_deepep_sweep.sh` | jumphost (interactive) | 6-stage one-button driver |
| `tray_build_deepep.sh`     | head_tray (via stage 3) | Build DeepEP against locally-built NCCL |
| `tray_deepep_sweep.sh`     | head_tray (via stage 4) | Multi-mpirun sweep matrix |
| `mpi_launch_ep.py`         | per-rank (via mpirun)   | OMPI->PyTorch env translator + spawn bypass |
| `parse_deepep_csv.py`      | jumphost (via stage 6)  | Aggregate logs into CSV + markdown table |

## Defaults (override via env)

```
HEAD_TRAY      = pod4-gb300-2-tray01-f3
TRAYS          = pod4-gb300-2-tray01-f3 .. pod4-gb300-2-tray04-f3
DEEPEP_DIR     = /home/fizhang/DeepEP
NCCL_ROOT_DIR  = /home/fizhang/nccl/build       (local NCCL master, >= 2.30.4)
MPI_HOME       = /usr/mpi/gcc/openmpi-4.1.9a1
CUDA_HOME      = /usr/local/cuda                (sm_103 / cuda 13.1)
DEEPEP_ARCH    = 10.0                           (Blackwell base; setup.py also adds DISABLE_AGGRESSIVE_PTX_INSTRS=1)
DEEPEP_BUILD_JOBS = 3                           (cicc stalls at -j16 on GB300; stick to 3)
DISABLE_LEGACY = 1                              (V2-only build, no NVSHMEM; set to 0 to also test V1)
JH_SSH_KEY     = ~/id_ed25519                   (jumphost ssh key for hopping to trays)
DEEPEP_LOG_DIR = /home/fizhang/deepep_logs      (per-rank log dir on tray, NFS shared)
JH_LOG_DIR     = /tmp/deepep_logs.<run_id>      (where logs land on jumphost)
```

### V2-only build (`DISABLE_LEGACY=1`)

By default `tray_build_deepep.sh` exports `DISABLE_LEGACY=1`, which:

- Skips compiling `csrc/kernels/legacy/{intranode,internode,internode_ll,layout}.cu`
  and `csrc/kernels/backend/nvshmem.cu` (the 70 MB internode.cu is the slowest
  one on GB300).
- Drops the `-l:libnvshmem_host.so` / `-l:libnvshmem_device.a` link, so no
  NVSHMEM install is needed at build or runtime.
- `#define DISABLE_LEGACY` for `python_api.cpp`, so `legacy::register_apis(m)`
  is not compiled in; `deep_ep.Buffer` and `deep_ep.Config` become `None` in
  Python (V2 doesn't reference either).

To also build / test the V1 path, set `DISABLE_LEGACY=0` and make sure NVSHMEM
is reachable (`pip install nvidia-nvshmem-cu13`, or untar a release, or set
`NVSHMEM_DIR`).

Sweep matrix knobs (passed through `jumphost_deepep_sweep.sh -> tray_deepep_sweep.sh`):

```
EP_SIZES       = "4 8 16"
TOKENS         = "1024 2048 4096 8192"
TOPK_EXPERTS   = "8:256 6:256"
EXTRA_ARGS     = ""        # appended to mpi_launch_ep.py, e.g. "--num-sms 8 --num-qps 9"
```

## Quick start (jumphost one-liner)

```bash
# from jumphost ~ , after you've cloned the DeepEP repo somewhere
ssh -i ~/id_ed25519 fizhang@pod4-gb300-2-tray01-f3 \
  'pkill -9 mpirun 2>/dev/null; pkill -9 -f mpi_launch_ep 2>/dev/null; pkill -9 -f test_ep 2>/dev/null; true'

TRAYS="pod4-gb300-2-tray01-f3 pod4-gb300-2-tray02-f3 pod4-gb300-2-tray03-f3 pod4-gb300-2-tray04-f3" \
GIT_REMOTE=git@github.com:<your-fork>/DeepEP.git \
GIT_REF=origin/master \
bash scripts/jumphost_deepep_sweep.sh 2>&1 | tee /tmp/deepep_sweep.log
```

After the first run, future passes can skip the ssh setup and the build:

```bash
SKIP_SSH_SETUP=1 SKIP_BUILD=1 \
EP_SIZES=16 TOKENS=8192 TOPK_EXPERTS="8:256" \
bash scripts/jumphost_deepep_sweep.sh
```

## What gets benchmarked

Per `(EP, tokens, topk, experts)` config the sweep runs `test_ep.py` with
`--test-first-only --skip-check`, which executes the **first** entry of
`enumerate_ep_modes()` (FP8 dispatch, `expert_alignment=128`, `do_handle_copy=1`,
no bias, no previous event, sync compute stream). For each config we capture
five timed kernels:

- `dispatch`
- `expanded dispatch`
- `cached dispatch`
- `combine`
- `reduced combine`

Each kernel emits per-rank lines like:
```
   * EP:   0/16 | dispatch: 90 GB/s (SO), 93 GB/s (SU), 123.456 us, 1234 bytes | copy: 200 GB/s, 5.000 us
```
`parse_deepep_csv.py` aggregates min/avg/max across ranks per (run, op).

## Per-tray manual run (skip the jumphost driver)

```bash
# on head_tray
cd /home/fizhang/DeepEP
DEEPEP_DIR=$PWD bash scripts/tray_build_deepep.sh
TRAYS="pod4-gb300-2-tray01-f3 pod4-gb300-2-tray02-f3 pod4-gb300-2-tray03-f3 pod4-gb300-2-tray04-f3" \
bash scripts/tray_deepep_sweep.sh
python scripts/parse_deepep_csv.py --log-dir /home/fizhang/deepep_logs --tag deepep_sweep_<timestamp>
```

## Single-config repro (for debugging one number)

```bash
# on head_tray (after build)
export DEEPEP_DIR=/home/fizhang/DeepEP
export NCCL_ROOT_DIR=/home/fizhang/nccl/build
export EP_NCCL_ROOT_DIR=$NCCL_ROOT_DIR
export PATH=/usr/mpi/gcc/openmpi-4.1.9a1/bin:/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/mpi/gcc/openmpi-4.1.9a1/lib:/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64:$NCCL_ROOT_DIR/lib:$LD_LIBRARY_PATH
export PYTHONPATH=$DEEPEP_DIR:$PYTHONPATH

cat > /tmp/hosts.4 <<EOF
pod4-gb300-2-tray01-f3 slots=4
pod4-gb300-2-tray02-f3 slots=4
pod4-gb300-2-tray03-f3 slots=4
pod4-gb300-2-tray04-f3 slots=4
EOF

mpirun --hostfile /tmp/hosts.4 -np 16 --map-by ppr:4:node \
  --mca pml ucx --mca btl ^openib \
  --mca plm_rsh_args "-i ~/.ssh/id_ed25519 -o StrictHostKeyChecking=accept-new" \
  -x PATH -x LD_LIBRARY_PATH -x CUDA_HOME -x MPI_HOME -x PYTHONPATH -x EP_NCCL_ROOT_DIR \
  -x NCCL_DEBUG=WARN -x EP_BUFFER_DEBUG=1 \
  -x MASTER_ADDR=pod4-gb300-2-tray01-f3 -x MASTER_PORT=8361 \
  -x DEEPEP_DIR=$DEEPEP_DIR \
  python -u $DEEPEP_DIR/scripts/mpi_launch_ep.py \
    --num-tokens 8192 --hidden 7168 --num-topk 8 --num-experts 256 \
    --skip-check --test-first-only --num-processes 4
```

## Tuning knobs to try after the baseline sweep

The first sweep gives you a feel for which configs are SO-bound (RDMA) vs.
SU-bound (NVLink) vs. copy-bound. From there:

| What you see | Knob to twist                                  | How |
|---|---|---|
| Low SO GB/s on EP16 hybrid (RDMA underutilised) | `EP_OVERRIDE_RDMA_SL` (try SL 0/1/3)            | `EXTRA_ARGS="--sl-idx 0"` or `-x EP_OVERRIDE_RDMA_SL=0` |
| `dispatch` time dominated by copy epilogue       | shrink `num-sms` for combine to free SMs        | `EXTRA_ARGS="--num-sms 12"` |
| Low SU GB/s on EP4 NVLink                        | bump `num-sms`                                  | `EXTRA_ARGS="--num-sms 24"` |
| Tail latency variance huge                       | rerun with `--prefer-overlap-with-compute=0`    | `EXTRA_ARGS="--prefer-overlap-with-compute 0"` |
| Hybrid vs direct comparison                      | toggle hybrid mode                              | `EXTRA_ARGS="--allow-hybrid-mode 0"` (and `=1` for hybrid) |
| Multi-reduce vs single-reduce combine            | toggle multiple_reduction                       | `EXTRA_ARGS="--allow-multiple-reduction 0"` |
| FP8 vs BF16 dispatch                             | uses `--test-first-only`'s first enum (FP8). Drop `--test-first-only` and grep specific cases if you need both. |

## Common pitfalls (carry-over from NCCL EP work)

| Trap | Workaround |
|---|---|
| `cd ~/fizhang/nccl` fails: jumphost `$HOME=/opt/forge/home/fizhang`, repo only on tray | Always use absolute `/home/fizhang/...` paths in remote ssh blocks. |
| ssh-quoting hell in nested commands | Use `bash -l <<'REMOTE' ... REMOTE` heredocs; quote the delimiter to disable expansion. |
| `pkill -9 -f mpi_launch_ep` kills my own ssh bash | Don't `-f` against your own shell; drop `-f` and target the exact process name (already done in this sweep). |
| `for t in $TRAYS; do ssh ... done` only runs the first tray | `ssh -n` (we always include `-n` to keep stdin out of for-loop). |
| `make -j16` cicc stalls on GB300 | `DEEPEP_BUILD_JOBS=3` (and we set `MAX_JOBS` for torch's JIT too). |
| Editing `.cu` files then running sweep -- no effect | `tray_deepep_sweep.sh` does NOT rebuild; re-run `tray_build_deepep.sh` after editing CUDA sources. |
| `mpirun` not finding peers ssh key | We pass `--mca plm_rsh_args "-i ~/.ssh/id_ed25519 -o StrictHostKeyChecking=accept-new"`; relies on stages 1-2 succeeding. |
| `setup.py` fails on `libnvshmem_host.so` | First try `DISABLE_LEGACY=1` (default in this repo) -- it drops the NVSHMEM link entirely. Only fall back to installing NVSHMEM if you need V1. |
| Non-9.0 arch fails on `.L1::no_allocate` PTX | Already exported `DISABLE_AGGRESSIVE_PTX_INSTRS=1` (setup.py also asserts it). |
| `EP_BUFFER_DEBUG=1` flooding logs | Off by default; flip on only for the first one-config smoke. |
| Variance between back-to-back runs | The sweep auto-`pkill`s stragglers on all trays between configs. |
