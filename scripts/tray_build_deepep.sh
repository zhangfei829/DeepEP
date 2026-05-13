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

# Resolve a python interpreter that has `torch` importable. DeepEP setup.py
# imports torch at top level, so a bare /usr/bin/python3 without torch is
# useless. We log every candidate we try so the user sees exactly which
# pythons exist and why each one was rejected.
_has_torch() { "$1" -c "import torch" >/dev/null 2>&1; }

# Collect candidate paths. Order: $PYTHON_BIN -> PATH pythons -> conda/venv
# -> pip --user site -> versioned /usr/bin/python3.X -> libtorch.so reverse
# lookup -> shotgun find under common roots.
_collect_python_candidates() {
  local out=()
  [ -n "${PYTHON_BIN:-}" ] && out+=("$PYTHON_BIN")

  local p
  for p in python3 python python3.10 python3.11 python3.12 python3.13; do
    p=$(command -v "$p" 2>/dev/null || true)
    [ -n "$p" ] && out+=("$p")
  done

  local conda_pys=(
    "$HOME"/anaconda3/envs/*/bin/python
    "$HOME"/miniconda3/envs/*/bin/python
    "$HOME"/anaconda3/bin/python
    "$HOME"/miniconda3/bin/python
    /opt/conda/envs/*/bin/python
    /opt/conda/bin/python
    /opt/miniconda*/bin/python
    "$HOME"/.venv/bin/python
    "$HOME"/venv/bin/python
    "$HOME"/.local/bin/python*
    /usr/bin/python3.*
  )
  for p in "${conda_pys[@]}"; do
    [ -x "$p" ] && out+=("$p")
  done

  # Reverse-lookup from any libtorch.so on the filesystem. Include shared FS
  # roots commonly used on HPC clusters (/SFS-*, /scratch, /shared, /work,
  # /data). Exclude /mnt/local/enroot (those are container rootfs blobs that
  # require `enroot start`, not directly runnable).
  local search_roots=("$HOME" /opt /usr/local /usr /scratch /shared /work /data)
  local sfs_root
  for sfs_root in /SFS-* /SFS /sfs /sfs-*; do
    [ -d "$sfs_root" ] && search_roots+=("$sfs_root")
  done
  local lib site_dir py_dir
  for lib in $(find "${search_roots[@]}" -maxdepth 10 -name 'libtorch.so' 2>/dev/null \
                | grep -v '/enroot/data/' | head -30); do
    site_dir=$(dirname "$lib")                                # .../site-packages/torch/lib
    site_dir=$(dirname "$(dirname "$site_dir")")              # .../site-packages
    py_dir=$(dirname "$(dirname "$site_dir")")                # .../lib/pythonX.Y/  -> .../
    local pyver
    pyver=$(basename "$(dirname "$site_dir")")                # pythonX.Y
    [ -x "$py_dir/bin/$pyver" ]     && out+=("$py_dir/bin/$pyver")
    [ -x "$py_dir/bin/python3" ]    && out+=("$py_dir/bin/python3")
    [ -x "$py_dir/bin/python" ]     && out+=("$py_dir/bin/python")
    # venv layout: site-packages may live under .../lib/pythonX.Y/site-packages
    # with python in ../bin/, or under .../python3.X/site-packages with
    # python in <venv_root>/bin/.
    local venv_root
    venv_root=$(dirname "$(dirname "$(dirname "$site_dir")")")  # ../../../
    [ -x "$venv_root/bin/python" ]  && out+=("$venv_root/bin/python")
    [ -x "$venv_root/bin/python3" ] && out+=("$venv_root/bin/python3")
    [ -x "$venv_root/bin/$pyver" ]  && out+=("$venv_root/bin/$pyver")
  done

  # Shotgun: any python3 under common roots (no enroot). Restrict basename
  # to real interpreter names: python, python3, python3.X (no -config,
  # -unidiff, -pip, -pyvenv junk).
  for p in $(find "${search_roots[@]}" -maxdepth 8 -name 'python3*' -type f -executable 2>/dev/null \
              | grep -v '/enroot/data/' \
              | grep -E '/(python|python3|python3\.[0-9]+)$' \
              | head -50); do
    out+=("$p")
  done

  # Filter all collected candidates by basename to drop python3.12-config etc.
  printf '%s\n' "${out[@]}" \
    | grep -E '/(python|python3|python3\.[0-9]+)$' \
    | awk '!seen[$0]++'
}

# Try `import torch` directly on $1. If it fails, fall back to injecting
# a sibling venv site-packages via PYTHONPATH (handles broken pyvenv.cfg
# where the bare interpreter doesn't auto-pick venv site-packages).
# On success: sets _CHECK_PY_PATH (PYTHONPATH addition, may be empty)
# and _CHECK_PY_TVER (torch version string). Returns 0/1.
_check_python() {
  local py="$1"
  _CHECK_PY_PATH=""
  _CHECK_PY_LDLIB=""
  _CHECK_PY_TVER=""
  # Sanity: must behave like a python interpreter (rejects python3.12-config etc).
  "$py" -c 'import sys' >/dev/null 2>&1 || return 1
  local tver
  tver=$("$py" -c 'import torch; print(torch.__version__)' 2>/dev/null || true)
  if [ -n "$tver" ]; then
    _CHECK_PY_TVER="$tver"; return 0
  fi
  local err
  err=$("$py" -c 'import torch' 2>&1 | tail -n 1)
  printf '      ! %s plain import failed: %s\n' "$py" "$err" >&2
  # Try injecting site-packages (PYTHONPATH) and torch/lib (LD_LIBRARY_PATH).
  # PyTorch wheels rely on RPATH=$ORIGIN to find sibling .so files in
  # torch/lib/; on some shared-FS mounts $ORIGIN does not resolve and the
  # `import torch` fails with a misleading "cannot open shared object file:
  # No such file or directory" pointing at libtorch_global_deps.so even
  # though the file is there. Forcing LD_LIBRARY_PATH=<site>/torch/lib
  # restores resolution.
  local py_dir venv_root site torchlib
  py_dir=$(dirname "$py")
  venv_root=$(dirname "$py_dir")
  for site in "$venv_root"/lib/python*/site-packages \
              "$venv_root"/lib64/python*/site-packages \
              "$venv_root"/lib/python*/dist-packages; do
    if [ ! -d "$site" ];       then printf '      . no site dir   %s\n' "$site" >&2; continue; fi
    if [ ! -d "$site/torch" ]; then printf '      . no torch in   %s\n' "$site" >&2; continue; fi
    torchlib="$site/torch/lib"
    printf '      ? PYTHONPATH=%s LD_LIBRARY_PATH=%s ...' "$site" "$torchlib" >&2
    tver=$(PYTHONPATH="$site${PYTHONPATH:+:$PYTHONPATH}" \
           LD_LIBRARY_PATH="$torchlib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
           "$py" -c 'import torch; print(torch.__version__)' 2>/dev/null || true)
    if [ -n "$tver" ]; then
      echo " ok (torch=$tver)" >&2
      _CHECK_PY_PATH="$site"; _CHECK_PY_LDLIB="$torchlib"; _CHECK_PY_TVER="$tver"
      return 0
    fi
    err=$(PYTHONPATH="$site${PYTHONPATH:+:$PYTHONPATH}" \
          LD_LIBRARY_PATH="$torchlib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
          "$py" -c 'import torch' 2>&1 | tail -n 1)
    echo " failed: $err" >&2
    if [ -f "$torchlib/libtorch_global_deps.so" ] && command -v ldd >/dev/null 2>&1; then
      local missing
      missing=$(LD_LIBRARY_PATH="$torchlib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
                ldd "$torchlib/libtorch_global_deps.so" 2>&1 | grep -E 'not found|=>\s*not' | head -3)
      [ -n "$missing" ] && printf '        ldd missing deps:\n%s\n' "$missing" | sed 's/^/          /' >&2
    fi
  done
  return 1
}

# Probe all candidates. Mutates global PYTHON_BIN and (if needed)
# PYTHONPATH / LD_LIBRARY_PATH.
_resolve_python() {
  echo "[build] probing python interpreters for torch..." >&2
  local p chosen="" chosen_pp="" chosen_ld=""
  while IFS= read -r p; do
    [ -n "$p" ] || continue
    if [ ! -x "$p" ]; then
      printf '  - %-50s  (not executable)\n' "$p" >&2; continue
    fi
    local ver
    ver=$("$p" -c 'import sys; print(sys.version.split()[0])' 2>/dev/null || echo "?")
    if _check_python "$p"; then
      if [ -n "$_CHECK_PY_PATH" ]; then
        printf '  + %-50s  python=%s torch=%s  <-- using (PYTHONPATH=%s LD_LIBRARY_PATH+=%s)\n' \
          "$p" "$ver" "$_CHECK_PY_TVER" "$_CHECK_PY_PATH" "$_CHECK_PY_LDLIB" >&2
      else
        printf '  + %-50s  python=%s torch=%s  <-- using\n' \
          "$p" "$ver" "$_CHECK_PY_TVER" >&2
      fi
      chosen="$p"; chosen_pp="$_CHECK_PY_PATH"; chosen_ld="$_CHECK_PY_LDLIB"; break
    else
      printf '  - %-50s  python=%s (no torch)\n' "$p" "$ver" >&2
    fi
  done < <(_collect_python_candidates)
  [ -n "$chosen" ] || return 1
  PYTHON_BIN="$chosen"
  if [ -n "$chosen_pp" ]; then
    export PYTHONPATH="$chosen_pp${PYTHONPATH:+:$PYTHONPATH}"
    echo "[build] injected PYTHONPATH=$PYTHONPATH" >&2
  fi
  if [ -n "$chosen_ld" ]; then
    export LD_LIBRARY_PATH="$chosen_ld${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    echo "[build] injected LD_LIBRARY_PATH=$LD_LIBRARY_PATH" >&2
  fi
  return 0
}

if ! _resolve_python; then PYTHON_BIN=""; fi
if [ -z "${PYTHON_BIN:-}" ] || [ ! -x "$PYTHON_BIN" ] || ! "$PYTHON_BIN" -c "import torch" >/dev/null 2>&1; then
  err "no python interpreter with torch found anywhere."
  err "Common fixes:"
  err "  1) module load python ; module load pytorch       (if your cluster uses modulefiles)"
  err "  2) source /path/to/conda/bin/activate <envname>   (then re-run this script)"
  err "  3) PYTHON_BIN=/abs/path/to/python-with-torch bash $0"
  err "  4) pip install --user 'torch>=2.10'               (then re-run)"
  exit 1
fi
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

log "running setup.py build (this can take a while; full log -> $DEEPEP_DIR/build.log)"
"$PYTHON_BIN" setup.py build 2>&1 | tee build.log | tail -n 40

# setup.py names the ext `deep_ep._C` -> _C.cpython-312-<arch>-linux-gnu.so
# located under build/lib.<plat>-cpython-<pyver>/deep_ep/. Old `deep_ep*.so`
# glob never matched because the basename is `_C.*.so`, not `deep_ep*.so`.
so_file=$(find build -maxdepth 4 -path '*/deep_ep/*' -name '*.so' -type f \
            ! -name 'lib*' 2>/dev/null | head -n 1)
if [ -z "$so_file" ]; then
  err "no built .so found under build/ ; full log at $DEEPEP_DIR/build.log"
  err "last 5 .so files seen:"
  find build -name '*.so' -type f 2>/dev/null | head -5 | sed 's/^/    /' >&2
  exit 1
fi
ln -sf "../$so_file" "deep_ep/$(basename "$so_file")"
log "linked $so_file -> deep_ep/$(basename "$so_file")"

# ----- patch PyTorch wheel NCCL to match user-built NCCL ---------------------
# DeepEP's `check_nccl_so()` (in deep_ep/__init__.py) asserts that the NCCL
# .so loaded by torch.cuda.nccl (typically the one bundled in the wheel under
# nvidia/nccl/lib/) is *byte-identical* to the NCCL we linked DeepEP against
# (NCCL_ROOT_DIR). A pip-installed torch usually ships libnccl.so.2 from
# nvidia-nccl-cu13 (2.29.x), which mismatches our user-built NCCL >=2.30 with
# Gin backend. Symlink-swap the wheel copy to the user-built one so torch
# and DeepEP share the exact same NCCL at runtime.
log "patching PyTorch wheel NCCL -> $NCCL_ROOT_DIR"
PT_NCCL_DIR=$("$PYTHON_BIN" - <<'PY' 2>/dev/null || true
import os, sys
# nvidia.nccl ships as a PEP 420 namespace package (no __init__.py, no
# __file__), so we must use __path__ or scan site-packages directly.
try:
    import nvidia.nccl
    paths = list(getattr(nvidia.nccl, '__path__', []) or [])
    for p in paths:
        d = os.path.join(p, 'lib')
        if os.path.isdir(d):
            print(d); sys.exit(0)
except Exception:
    pass
try:
    import site
    cands = list(site.getsitepackages()) + [site.getusersitepackages()]
    for sp in cands:
        d = os.path.join(sp, 'nvidia', 'nccl', 'lib')
        if os.path.isdir(d):
            print(d); sys.exit(0)
except Exception:
    pass
PY
)
USER_NCCL_SO=$(ls -1 "$NCCL_ROOT_DIR"/lib/libnccl.so.2.* 2>/dev/null | head -1)
if [ -z "$USER_NCCL_SO" ]; then
  USER_NCCL_SO=$(readlink -f "$NCCL_ROOT_DIR/lib/libnccl.so" 2>/dev/null || true)
fi
if [ -n "$PT_NCCL_DIR" ] && [ -d "$PT_NCCL_DIR" ] && [ -n "$USER_NCCL_SO" ] && [ -f "$USER_NCCL_SO" ]; then
  if [ -f "$PT_NCCL_DIR/libnccl.so.2" ] && [ ! -L "$PT_NCCL_DIR/libnccl.so.2" ]; then
    mv -f "$PT_NCCL_DIR/libnccl.so.2" "$PT_NCCL_DIR/libnccl.so.2.wheel-orig" 2>/dev/null || true
  fi
  ln -sf "$USER_NCCL_SO" "$PT_NCCL_DIR/libnccl.so.2"
  log "  PT_NCCL_DIR    = $PT_NCCL_DIR"
  log "  USER_NCCL_SO   = $USER_NCCL_SO"
  log "  ln -sf $(readlink "$PT_NCCL_DIR/libnccl.so.2")"
else
  log "  skipped (PT_NCCL_DIR='$PT_NCCL_DIR' USER_NCCL_SO='$USER_NCCL_SO')"
fi

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
