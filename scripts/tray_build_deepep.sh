#!/usr/bin/env bash
# tray_build_deepep.sh - build DeepEP V2 on a GB300 tray (sm_103).
#
# Run from: head_tray (anywhere on /home/fizhang shared FS)
# Result:   deep_ep/deep_ep_cpp...so symlink created under $DEEPEP_DIR for in-place use
#
# Tunables (all overridable via env):
#   DEEPEP_DIR        repo root on tray   (default /home/fizhang/DeepEP)
#   NCCL_ROOT_DIR     locally-built NCCL  (default /home/fizhang/nccl/build)  -- libnccl.so + include/ required
#   MPI_HOME          OpenMPI install     (default /usr/mpi/gcc/openmpi-4.1.9a1)
#   CUDA_HOME         CUDA install        (default /usr/local/cuda)
#   DEEPEP_ARCH       TORCH_CUDA_ARCH_LIST(default 10.0 for GB300/Blackwell)
#   DEEPEP_BUILD_JOBS parallel cuda jobs  (default 3, learned the hard way on GB300 cicc)
#   DISABLE_LEGACY    1 to strip V1/NVSHMEM entirely  (default 1 for GB300 sweep work)
#   NVSHMEM_DIR       optional, NVSHMEM install prefix. Only used when DISABLE_LEGACY=0.
#   SKIP_NCCL_CHECK   1 to skip libnccl.so presence assert (default 0)
#
# Pitfalls captured from prior NCCL EP work, see comments inline.
set -euo pipefail

log() { printf '\033[1;36m[build]\033[0m %s\n' "$*"; }
err() { printf '\033[1;31m[build:ERR]\033[0m %s\n' "$*" >&2; }

# ----- inputs / defaults -----------------------------------------------------
: "${DEEPEP_DIR:=/home/fizhang/DeepEP}"
: "${NCCL_ROOT_DIR:=/home/fizhang/nccl/build}"
: "${MPI_HOME:=/usr/mpi/gcc/openmpi-4.1.9a1}"
: "${CUDA_HOME:=/usr/local/cuda}"
: "${DEEPEP_ARCH:=10.0}"
: "${DEEPEP_BUILD_JOBS:=3}"
: "${SKIP_NCCL_CHECK:=0}"
# DISABLE_LEGACY=1 strips V1/NVSHMEM. V2 (`ElasticBuffer`) doesn't need NVSHMEM,
# so this is the default for GB300/sm_103 sweep work: faster compile, no NVSHMEM
# dependency. Set DISABLE_LEGACY=0 if you also want to test V1.
: "${DISABLE_LEGACY:=1}"

# ----- sanity ----------------------------------------------------------------
[ -d "$DEEPEP_DIR" ]   || { err "DEEPEP_DIR not found: $DEEPEP_DIR";   exit 1; }
[ -d "$CUDA_HOME" ]    || { err "CUDA_HOME not found:  $CUDA_HOME";    exit 1; }
[ -d "$MPI_HOME" ]     || { err "MPI_HOME not found:   $MPI_HOME";     exit 1; }

# Resolve a python interpreter. Some trays only have `python3`, not `python`.
: "${PYTHON_BIN:=}"
if [ -z "$PYTHON_BIN" ]; then
  for cand in python3 python; do
    if command -v "$cand" >/dev/null 2>&1; then PYTHON_BIN="$cand"; break; fi
  done
fi
[ -n "$PYTHON_BIN" ] || { err "no python interpreter found (tried python3, python). Set PYTHON_BIN=/path/to/python"; exit 1; }
export PYTHON_BIN
if [ "$SKIP_NCCL_CHECK" != "1" ]; then
  [ -f "$NCCL_ROOT_DIR/lib/libnccl.so" ]    || { err "missing $NCCL_ROOT_DIR/lib/libnccl.so   - did you run 'make src.build' on NCCL?"; exit 1; }
  [ -f "$NCCL_ROOT_DIR/include/nccl.h" ]    || { err "missing $NCCL_ROOT_DIR/include/nccl.h"; exit 1; }
fi

log "DEEPEP_DIR     = $DEEPEP_DIR"
log "NCCL_ROOT_DIR  = $NCCL_ROOT_DIR"
log "MPI_HOME       = $MPI_HOME"
log "CUDA_HOME      = $CUDA_HOME"
log "DEEPEP_ARCH    = $DEEPEP_ARCH"
log "BUILD_JOBS     = $DEEPEP_BUILD_JOBS  (use >=4 only if you've verified cicc isn't stuck on GB300)"
log "DISABLE_LEGACY = $DISABLE_LEGACY  (1 = V2-only build, no NVSHMEM)"
log "PYTHON_BIN     = $PYTHON_BIN"

# ----- compile env -----------------------------------------------------------
export PATH="$MPI_HOME/bin:$CUDA_HOME/bin:${PATH:-}"
export LD_LIBRARY_PATH="$MPI_HOME/lib:$CUDA_HOME/lib64:$CUDA_HOME/extras/CUPTI/lib64:$NCCL_ROOT_DIR/lib:${LD_LIBRARY_PATH:-}"
export CUDA_HOME MPI_HOME

# DeepEP setup.py looks for NCCL via EP_NCCL_ROOT_DIR / NCCL_DIR before falling
# back to Python wheels. Point it at the locally-built NCCL.
export EP_NCCL_ROOT_DIR="$NCCL_ROOT_DIR"

# GB300 = sm_103 (Blackwell). The repo's default 9.0 (H800) won't generate
# native SASS for us; setup.py also enforces DISABLE_AGGRESSIVE_PTX_INSTRS=1
# for anything other than 9.0 (some .L1::no_allocate variants aren't legal).
export TORCH_CUDA_ARCH_LIST="$DEEPEP_ARCH"
export DISABLE_AGGRESSIVE_PTX_INSTRS=1

# Smaller -j for the JIT side too; the JIT compiler will pick this up
export MAX_JOBS="$DEEPEP_BUILD_JOBS"

# Propagate to setup.py: when 1, V1 sources (legacy/intranode/internode/internode_ll/
# nvshmem.cu) are not compiled and NVSHMEM headers/libs are not linked.
export DISABLE_LEGACY

# ----- NVSHMEM probe (only when legacy V1 is in the build) ------------------
# setup.py links libnvshmem_host.so only when DISABLE_LEGACY=0.
# When DISABLE_LEGACY=1, V2 path uses NCCL Gin and we skip the probe entirely.
probe_nvshmem() {
  local probe
  if [ -n "${NVSHMEM_DIR:-}" ] && [ -f "$NVSHMEM_DIR/lib/libnvshmem_host.so" ]; then
    echo "$NVSHMEM_DIR"; return 0
  fi
  for probe in /usr/local/nvshmem /opt/nvshmem /home/fizhang/nvshmem; do
    if [ -f "$probe/lib/libnvshmem_host.so" ] || \
       ls "$probe/lib/libnvshmem_host.so."* >/dev/null 2>&1; then
      echo "$probe"; return 0
    fi
  done
  # pip wheel path (nvidia-nvshmem-cu13 / cu12)
  local py_lib
  py_lib=$("$PYTHON_BIN" - <<'PY' 2>/dev/null || true
import os
from importlib.metadata import distributions
for d in distributions():
    name = (d.metadata['Name'] or '').lower()
    if 'nvshmem' not in name:
        continue
    for f in (d.files or []):
        s = str(f)
        if 'libnvshmem_host.so' in s:
            lib_dir = os.path.dirname(str(f.locate()))
            root = os.path.dirname(lib_dir) if os.path.basename(lib_dir) == 'lib' else lib_dir
            print(root)
            raise SystemExit
PY
)
  if [ -n "$py_lib" ]; then echo "$py_lib"; return 0; fi
  return 1
}

if [ "$DISABLE_LEGACY" = "1" ]; then
  log "NVSHMEM        = SKIPPED (DISABLE_LEGACY=1, V2-only build)"
elif NVSHMEM_FOUND=$(probe_nvshmem); then
  export NVSHMEM_DIR="$NVSHMEM_FOUND"
  export LD_LIBRARY_PATH="$NVSHMEM_DIR/lib:$LD_LIBRARY_PATH"
  log "NVSHMEM_DIR    = $NVSHMEM_DIR (auto-detected)"
else
  err "NVSHMEM not found and DISABLE_LEGACY=0. setup.py needs libnvshmem_host.so for V1 kernels."
  err "Pick one:"
  err "  0) export DISABLE_LEGACY=1 and re-run (V2-only, no NVSHMEM needed)   <- easiest"
  err "  1) pip install nvidia-nvshmem-cu13 (or cu12)"
  err "  2) untar https://developer.download.nvidia.com/compute/nvshmem/redist/...  to /home/fizhang/nvshmem and re-run"
  err "  3) export NVSHMEM_DIR=/path/to/nvshmem and re-run"
  exit 1
fi

# ----- show resolved versions (a single source of truth on failure) ---------
log "--- toolchain ---"
which nvcc      && nvcc --version | head -n4 || true
which mpicc     && mpicc --version | head -n1 || true
which "$PYTHON_BIN" && "$PYTHON_BIN" -V       || true
"$PYTHON_BIN" -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda)" || true

# ----- build -----------------------------------------------------------------
cd "$DEEPEP_DIR"

log "cleaning previous build/"
rm -rf build dist deep_ep/*.so deep_ep.egg-info

log "running setup.py build (this can take a while)"
"$PYTHON_BIN" setup.py build 2>&1 | tail -n 40

# in-place .so symlink so `python tests/elastic/test_ep.py` finds the module
so_file=$(find build -maxdepth 3 -name 'deep_ep*.so' -type f | head -n 1)
if [ -z "$so_file" ]; then
  err "no built .so found under build/; full output above"
  exit 1
fi
ln -sf "../$so_file" "deep_ep/$(basename "$so_file")"
log "linked $so_file -> deep_ep/$(basename "$so_file")"

# ----- smoke: import + NCCL version -----------------------------------------
log "smoke: import deep_ep (this also triggers NCCL/library probes)"
EP_BUFFER_DEBUG=1 "$PYTHON_BIN" - <<'PY'
import os, deep_ep
print('OK: deep_ep version', deep_ep.__version__)
print('   ElasticBuffer :', deep_ep.ElasticBuffer)
print('   topk_idx_t    :', deep_ep.topk_idx_t)
# Legacy V1 is optional when DISABLE_LEGACY=1; Buffer/Config will be None.
print('   Buffer (V1)   :', deep_ep.Buffer)
print('   Config (V1)   :', deep_ep.Config)
PY

log "DeepEP V2 build complete at $DEEPEP_DIR"
