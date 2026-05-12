"""
MPI launcher for `tests/elastic/test_ep.py`.

Why:
    DeepEP V2's `test_ep.py` defaults to `torch.multiprocessing.spawn` for a single-node run.
    To cover multi-tray (EP > num_gpus_per_node) we need each rank to be a separate process
    launched by `mpirun`. This launcher:
      1. Translates OMPI env vars to the PyTorch dist envs that `init_dist` expects.
      2. Monkey-patches `torch.multiprocessing.spawn` so that the existing `test_ep.py`
         main block calls `test_loop(local_rank, num_local_ranks, args)` directly
         instead of spawning child processes.
      3. Optionally redirects stdout to a per-rank log file for downstream CSV parsing.

Typical usage (from `tray_deepep_sweep.sh`):

    mpirun --hostfile $HOSTS -np 16 --map-by ppr:4:node \
        -x MASTER_ADDR -x MASTER_PORT -x DEEPEP_LOG_DIR -x DEEPEP_RUN_TAG \
        -x LD_LIBRARY_PATH -x CUDA_HOME -x EP_NCCL_ROOT_DIR \
        python -u scripts/mpi_launch_ep.py \
            --num-tokens 4096 --hidden 7168 --num-topk 8 --num-experts 256 \
            --skip-check --test-first-only

Per-rank log files are written to `$DEEPEP_LOG_DIR/$DEEPEP_RUN_TAG.rank{:02d}.log`
when both env vars are set; rank 0 also writes to the original stdout (tee).
"""

import os
import runpy
import sys
from pathlib import Path


def _read_required_env(name: str) -> str:
    v = os.environ.get(name)
    if v is None:
        raise SystemExit(
            f'[mpi_launch_ep] required env {name!r} not set. '
            f'This launcher must be started under mpirun (Open MPI).'
        )
    return v


def _translate_ompi_to_pytorch() -> int:
    """
    Translate OMPI env vars to PyTorch dist envs that `tests/elastic/test_ep.py:init_dist`
    expects:
        MASTER_ADDR/MASTER_PORT - rendezvous endpoint
        WORLD_SIZE              - number of *nodes* (trays)
        RANK                    - this node's index among nodes (0-based)
    Returns the local rank within this node (a.k.a. local GPU index).
    """
    rank = int(_read_required_env('OMPI_COMM_WORLD_RANK'))
    world_size = int(_read_required_env('OMPI_COMM_WORLD_SIZE'))
    local_rank = int(_read_required_env('OMPI_COMM_WORLD_LOCAL_RANK'))
    local_size = int(_read_required_env('OMPI_COMM_WORLD_LOCAL_SIZE'))

    if world_size % local_size != 0:
        raise SystemExit(
            f'[mpi_launch_ep] world_size={world_size} is not divisible by '
            f'local_size={local_size}. Use a balanced --map-by ppr:N:node.'
        )
    num_nodes = world_size // local_size
    node_rank = rank // local_size
    if node_rank * local_size + local_rank != rank:
        raise SystemExit(
            f'[mpi_launch_ep] rank layout mismatch: rank={rank} '
            f'node_rank={node_rank} local_rank={local_rank} local_size={local_size}'
        )

    os.environ.setdefault('MASTER_ADDR', '127.0.0.1')
    os.environ.setdefault('MASTER_PORT', '8361')
    os.environ['WORLD_SIZE'] = str(num_nodes)
    os.environ['RANK'] = str(node_rank)
    os.environ['LOCAL_RANK'] = str(local_rank)
    os.environ['LOCAL_WORLD_SIZE'] = str(local_size)
    return local_rank, local_size, rank


def _redirect_stdout_to_per_rank_log(global_rank: int) -> None:
    """
    If $DEEPEP_LOG_DIR and $DEEPEP_RUN_TAG are set, redirect this rank's stdout
    (and stderr) to `$DEEPEP_LOG_DIR/$DEEPEP_RUN_TAG.rank{NN}.log`.
    Rank 0 also tees to the original stdout so the operator gets live progress.
    """
    log_dir = os.environ.get('DEEPEP_LOG_DIR')
    run_tag = os.environ.get('DEEPEP_RUN_TAG')
    if not log_dir or not run_tag:
        return

    Path(log_dir).mkdir(parents=True, exist_ok=True)
    log_path = Path(log_dir) / f'{run_tag}.rank{global_rank:02d}.log'

    log_fp = open(log_path, 'w', buffering=1)
    if global_rank == 0:
        orig_stdout = sys.stdout

        class _Tee:
            def __init__(self, *fps): self._fps = fps
            def write(self, s):
                for fp in self._fps:
                    fp.write(s)
            def flush(self):
                for fp in self._fps:
                    fp.flush()

        sys.stdout = _Tee(orig_stdout, log_fp)
    else:
        sys.stdout = log_fp
    sys.stderr = sys.stdout


def _patch_spawn_to_direct_call(local_rank: int, local_size: int) -> None:
    """
    Replace `torch.multiprocessing.spawn` with a function that runs the target
    callable in-process for our local rank, mirroring `mp.spawn`'s call shape:

        spawn(fn, args=(num_local_ranks, parsed_args), nprocs=num_local_ranks)
            -> fn(local_rank, *args) for one specific local_rank only.

    We also force the `num_local_ranks` arg to match what mpirun gave us,
    in case the user passed a stale `--num-processes` on the CLI.
    """
    import torch.multiprocessing as mp

    def fake_spawn(fn, args=(), nprocs=1, join=True, daemon=False, start_method='spawn'):
        if nprocs != local_size:
            print(f'[mpi_launch_ep] note: overriding --num-processes={nprocs} '
                  f'with OMPI local_size={local_size}', flush=True)
        coerced_args = (local_size, ) + tuple(args[1:])
        fn(local_rank, *coerced_args)

    mp.spawn = fake_spawn


def _find_test_ep_script() -> Path:
    """
    Locate `tests/elastic/test_ep.py`. We first try `$DEEPEP_DIR/tests/elastic/test_ep.py`,
    then resolve relative to this script (assumes scripts/ sits next to tests/).
    """
    explicit = os.environ.get('DEEPEP_DIR')
    if explicit:
        candidate = Path(explicit) / 'tests' / 'elastic' / 'test_ep.py'
        if candidate.is_file():
            return candidate
        raise SystemExit(f'[mpi_launch_ep] DEEPEP_DIR set to {explicit!r} but {candidate} not found')

    here = Path(__file__).resolve()
    candidate = here.parent.parent / 'tests' / 'elastic' / 'test_ep.py'
    if candidate.is_file():
        return candidate
    raise SystemExit(f'[mpi_launch_ep] cannot find test_ep.py at {candidate}')


def main() -> None:
    local_rank, local_size, global_rank = _translate_ompi_to_pytorch()
    _redirect_stdout_to_per_rank_log(global_rank)

    test_ep_path = _find_test_ep_script()
    sys.path.insert(0, str(test_ep_path.parent.parent.parent))

    # Make sure argparse in test_ep.py sees the right argv
    # (we keep all args the operator passed; only `prog name` is replaced)
    sys.argv = [str(test_ep_path)] + sys.argv[1:]

    _patch_spawn_to_direct_call(local_rank, local_size)

    if global_rank == 0:
        print(
            f'[mpi_launch_ep] global_rank={global_rank} local_rank={local_rank} '
            f'local_size={local_size} world_size={os.environ["WORLD_SIZE"]} '
            f'master={os.environ["MASTER_ADDR"]}:{os.environ["MASTER_PORT"]} '
            f'argv={sys.argv[1:]}',
            flush=True,
        )

    runpy.run_path(str(test_ep_path), run_name='__main__')


if __name__ == '__main__':
    main()
