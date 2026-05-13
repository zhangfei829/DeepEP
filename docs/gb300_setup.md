# GB300 / NVL72 DeepEP V2 Setup & Test Guide

Goal: on any 4-tray subset of an NVL72 pod (or up to 18 nodes), bring up a
clean DeepEP V2 sweep in **one command**, with all the pitfalls below
already handled by `scripts/jumphost_deepep_sweep.sh`.

---

## TL;DR — one-button command (jumphost shell)

After you SSH from your laptop to the jumphost (`hungry-hippo-fin-03-jumphost`)
and have a github fork URL at hand:

```bash
cd ~/DeepEP && git pull origin main && \
  HEAD_TRAY=pod4-gb300-2-tray06-f3 \
  TRAYS="pod4-gb300-2-tray06-f3 pod4-gb300-2-tray07-f3 pod4-gb300-2-tray08-f3 pod4-gb300-2-tray09-f3" \
  GIT_REMOTE=https://github.com/zhangfei829/DeepEP.git \
  GIT_REF=origin/main \
  bash scripts/jumphost_deepep_sweep.sh 2>&1 | tee /tmp/deepep_sweep.log
```

That's it. The script will:

1. Generate an SSH key on `HEAD_TRAY` and distribute its pubkey to every
   tray in `TRAYS` (mesh trust).
2. **Provision torch + NCCL wheel on every tray** (this is the step you
   used to forget). Repos are stored under `~/.local` which on this pod
   resolves to **`/mnt/local`, node-local disk**, so a single install
   on head_tray does NOT reach workers — we install on every node.
3. `git clone` / `git fetch` / `git reset --hard` the DeepEP repo on
   `HEAD_TRAY` to your fork, then `bash scripts/tray_build_deepep.sh`.
4. Run the dispatch/combine sweep matrix
   (`EP × tokens × topk × experts`) via `mpirun` across all trays.
5. SCP per-rank logs back to the jumphost
   (`/tmp/deepep_logs.<RUN_ID>/`).
6. Parse logs into a summary CSV + Markdown table.

Skip flags (set to `1` once a step is known-good):

| flag             | what it skips                          |
| ---------------- | -------------------------------------- |
| `SKIP_SSH_SETUP` | stages 1 + 2 (ssh key + pubkey mesh)   |
| `SKIP_PIP`       | stage 2.5 (per-tray torch/NCCL install)|
| `SKIP_BUILD`     | stage 3 (git sync + DeepEP build)      |

Re-running on the same 4 trays after a code change:

```bash
SKIP_SSH_SETUP=1 SKIP_PIP=1 ... bash scripts/jumphost_deepep_sweep.sh
```

---

## Hardware / software baseline (verified May 2026)

| component | value                                        |
| --------- | -------------------------------------------- |
| node      | NVIDIA GB300 NVL72, aarch64 (ARM64)          |
| OS        | Ubuntu 24.04.4 LTS, kernel 6.14.0-1013-nvidia|
| GPU SM    | sm_103 (Blackwell, compute cap (10,3))       |
| CUDA      | 13.1.115 at `/usr/local/cuda`                |
| MPI       | OpenMPI 4.1.9a1 at `/usr/mpi/gcc/openmpi-4.1.9a1` |
| Python    | system `/usr/bin/python3` = 3.12.3           |
| torch     | nightly 2.13.0.dev / cu130 (PyPI, aarch64)   |
| NCCL      | PyPI nvidia-nccl-cu13 **>= 2.30.4** (Gin)    |

DeepEP V2 (`ElasticBuffer`) uses NCCL's Gin device-side API
(`ncclGin_SegmentDevice` and friends), which is shipped in the
`nvidia-nccl-cu13` PyPI wheel from version **2.30.4** onward. Anything
older crashes the JIT compile with
`identifier "ncclGin_SegmentDevice" is undefined`.

---

## Login topology

```
laptop  --ssh-->  hungry-hippo-fin-03-jumphost  --ssh-->  pod4-gb300-2-tray0N-f3
```

The jumphost has a single key at `~/id_ed25519` that authorizes us
into every tray. The script reuses it as `JH_SSH_KEY`.

Inside the pod, `HEAD_TRAY` needs a separate key (`~/.ssh/id_ed25519`)
trusted by every other tray, so mpirun can launch workers. Stages 1 and 2
of the jumphost script set this up automatically.

---

## What lives where on the trays

Shared (CephFS, `mount type ceph`):

- `/home/fizhang/` — repo, NCCL build dir, logs, head_tray pubkey stash.

Node-local (`/mnt/local/home/fizhang/`, symlinked from `~`):

- `~/.local/` — pip --user installs land HERE. Critical: pip-installed
  torch is **not** shared, so every tray must `pip install --user` its
  own copy. Stage 2.5 handles this.

DO NOT use:

- `/SFS-*/` — there is a stale `/SFS-aGqda6ct/amd/venv` with an x86_64
  torch wheel that does NOT match the aarch64 trays. The probe in
  `scripts/tray_build_deepep.sh` will detect that ld.so refuses to load
  it (the error looks like "not a dynamic executable" or "cannot open
  shared object file" pointing at `libtorch_global_deps.so`).
- `/mnt/local/enroot/data/pyxis_*/...` — these are enroot/Pyxis
  container rootfs trees; they need `enroot start` and have their own
  CUDA/glibc env.

---

## Pitfalls captured (and what the script does about each)

| symptom                                                                                                | root cause                                                                              | handled by                                                                       |
| ------------------------------------------------------------------------------------------------------ | --------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------- |
| `bash: /home/fizhang/DeepEP/scripts/tray_build_deepep.sh: No such file or directory`                   | repo on tray was cloned from upstream DeepSeek; our scripts/ missing                    | stage 3 forces `git remote set-url origin "$GIT_REMOTE"`                         |
| `ssh -n ... <<HEREDOC` swallows stdin, build commands never reach the tray                             | `ssh -n` redirects stdin to `/dev/null`                                                 | separate `JSSH` (no -n) and `JSSH_N` (-n) wrappers                               |
| `ModuleNotFoundError: No module named 'torch'`                                                          | system python has no torch                                                              | `_resolve_python` probes /usr/bin, /SFS, conda, /home/.local, libtorch.so reverse lookup |
| `python3.12-config` falsely passes torch probe                                                          | `python3.12-config -c '...'` prints usage to stdout, exit 0                              | sanity check `<py> -c 'import sys'` + basename regex filter                      |
| `OSError: libtorch_global_deps.so: cannot open shared object file`                                      | torch wheel architecture mismatch (x86_64 wheel on aarch64 node) or RPATH=$ORIGIN broken | probe injects `LD_LIBRARY_PATH=<site>/torch/lib`; ldd missing-deps trace         |
| `no built .so found under build/`                                                                       | `setup.py` produces `_C.cpython-...so`, old probe looked for `deep_ep*.so`               | find pattern `*/deep_ep/*.so`                                                    |
| `undefined symbol: __cudaRegisterLinkedBinary_*`                                                        | nvcc `-rdc=true` without nvcc `--device-link` step (PyTorch CppExtension links with g++)| `setup.py` drops `-rdc=true` when `DISABLE_LEGACY=1`                             |
| `ImportError: cannot import name 'EventHandle' from 'deep_ep._C'`                                       | `EventHandle` pybind was registered inside legacy::register_apis, V2 also needs it       | moved to top-level `PYBIND11_MODULE` in `csrc/python_api.cpp`                    |
| `AssertionError: Invalid NCCL versions: ... (loaded) v.s. ... (expected)`                                | torch wheel ships `nvidia-nccl-cu13==2.29.7`, we link against >=2.30.4                  | build script symlink-swaps PyTorch wheel's `libnccl.so.2` -> user NCCL           |
| `TypeError: Too few arguments for ... CSE`                                                              | torch nightly inductor `cutedsl_kernel` has a `CSE[Any]` type-hint bug at import time   | `@torch.compile` gated behind `DEEPEP_DISABLE_TORCH_COMPILE=1` (default in scripts)|
| `pod4-gb300-2-tray07-f3: command not found`                                                             | ssh joins argv with spaces without re-quoting, kills `TRAYS="a b c d"`                  | `printf %q` + single argv to ssh                                                  |
| `identifier "ncclGin_SegmentDevice" is undefined` (during DeepEP JIT compile)                          | NCCL < 2.30.4 has no Gin device-side API                                                 | default `NCCL_ROOT_DIR` -> PyPI `nvidia-nccl-cu13>=2.30.4` wheel                  |
| `ModuleNotFoundError: No module named 'torch'` on rank04 (other trays)                                 | `~/.local` is /mnt/local (node-local disk), not shared                                  | stage 2.5 pip-installs torch+nccl on every tray                                  |
| `EP=8/16 hang or error`                                                                                  | tray workers missing python env                                                          | same as above                                                                    |

---

## Switching to a different NVL72 subset

For an 8-tray run on the next pod (`pod4-gb300-3-tray*`) for example:

```bash
HEAD_TRAY=pod4-gb300-3-tray01-f3 \
TRAYS="pod4-gb300-3-tray01-f3 pod4-gb300-3-tray02-f3 ... pod4-gb300-3-tray08-f3" \
EP_SIZES="8 16 32" \
TOKENS="1024 2048 4096 8192 16384" \
TOPK_EXPERTS="8:256 6:256" \
FP8=1 \
GIT_REMOTE=https://github.com/zhangfei829/DeepEP.git \
GIT_REF=origin/main \
bash scripts/jumphost_deepep_sweep.sh
```

Everything else (key gen, pubkey mesh, pip install, build, sweep) is
fully automatic.

To **scale from 4 trays to 18 trays without redoing setup**: keep the
same script invocation but extend `TRAYS`. Stage 2.5 will see that
trays 1-4 already have torch, and only pip-install on 5-18.

---

## Knobs you might want to flip

| env                          | effect                                                                                   |
| ---------------------------- | ---------------------------------------------------------------------------------------- |
| `DISABLE_LEGACY=0`           | also build the V1 (`Buffer` + NVSHMEM) path (requires NVSHMEM_DIR to be discoverable)    |
| `DEEPEP_DISABLE_TORCH_COMPILE=0` | re-enable `@torch.compile`; only safe if your torch nightly doesn't hit the CSE bug |
| `EP_SUPPRESS_NCCL_CHECK=1`   | skip DeepEP's "loaded == linked NCCL" strict check (last resort)                         |
| `DEEPEP_BUILD_JOBS=3`        | nvcc parallel jobs; >=4 is faster but cicc tends to deadlock on GB300                    |
| `NCCL_DEBUG=INFO`            | dump NCCL init logs to per-rank files in `$DEEPEP_LOG_DIR`                               |
| `EP_BUFFER_DEBUG=1`          | DeepEP-internal NCCL/library probe verbosity                                              |
| `PYTHON_BIN=/abs/path/python` | force a specific interpreter (skips the auto-probe)                                     |
| `NCCL_ROOT_DIR=...`          | use a hand-built NCCL instead of the PyPI wheel (must have Gin device API)               |

---

## Verifying a single tray is healthy

Quick check from the jumphost (`$T` = any tray hostname):

```bash
ssh -i ~/id_ed25519 fizhang@$T '/usr/bin/python3 -c "
import torch, nvidia.nccl, os
print(\"torch=\"+torch.__version__,
      \"cuda_avail=\"+str(torch.cuda.is_available()),
      \"cuda=\"+str(torch.version.cuda),
      \"device_cap=\"+str(torch.cuda.get_device_capability() if torch.cuda.is_available() else None),
      \"nccl_dir=\"+[p for p in nvidia.nccl.__path__][0])"'
```

Expected output on a provisioned tray:

```
torch=2.13.0.dev20260512+cu130 cuda_avail=True cuda=13.0 device_cap=(10, 3) nccl_dir=/mnt/local/home/fizhang/.local/lib/python3.12/site-packages/nvidia/nccl
```

---

## Files of interest

```
scripts/jumphost_deepep_sweep.sh    main 6-stage orchestrator (run this)
scripts/tray_build_deepep.sh        builds DeepEP on HEAD_TRAY
scripts/tray_deepep_sweep.sh        runs the sweep matrix via mpirun
scripts/mpi_launch_ep.py            OpenMPI -> PyTorch dist env translator
scripts/parse_deepep_csv.py         per-rank logs -> summary CSV + md table
docs/gb300_setup.md                 this file
```
