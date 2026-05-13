#!/usr/bin/env bash
# tray_deepep_sweep.sh - run DeepEP V2 dispatch/combine sweeps via mpirun across trays.
#
# Run from: head_tray (any tray with ssh-able peers; default = pod4-gb300-2-tray01-f3)
# Output:   $DEEPEP_LOG_DIR/<tag>.rank<NN>.log  per-rank stdout (parsed later)
#
# Sweep matrix (override via env):
#   EP_SIZES      list of EP world sizes, must be multiples of 4 (slots/node) -- default "4 8 16"
#   TOKENS        list of num_max_tokens_per_rank                              -- default "1024 2048 4096 8192"
#   TOPK_EXPERTS  list of "topk:experts" pairs                                 -- default "8:256 6:256"
#   FP8           1 = pass --use-fp8 flag through enumerate (default), 0 = bf16
#
# Other knobs:
#   TRAYS         hostnames slots=4 each (mpirun hostfile content)             -- default 4-tray pool
#   DEEPEP_DIR    repo on tray                                                 -- default /home/fizhang/DeepEP
#   NCCL_ROOT_DIR locally-built NCCL                                           -- default /home/fizhang/nccl/build
#   DEEPEP_LOG_DIR per-run log root                                            -- default /home/fizhang/deepep_logs
#   MASTER_PORT   PyTorch rendezvous port                                      -- default 8361
#   EXTRA_ARGS    appended to mpi_launch_ep.py (--num-sms, --num-qps, etc.)
#
# Per-run command:
#   pkill any stragglers -> mpirun -np <EP> -> per-rank log files.
#
# CI tip: pair with `parse_deepep_csv.py` to aggregate logs into a table.
set -euo pipefail

log()  { printf '\033[1;36m[sweep]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[sweep:WARN]\033[0m %s\n' "$*" >&2; }
err()  { printf '\033[1;31m[sweep:ERR]\033[0m %s\n' "$*" >&2; }

# ----- defaults --------------------------------------------------------------
: "${EP_SIZES:=4 8 16}"
: "${TOKENS:=1024 2048 4096 8192}"
: "${TOPK_EXPERTS:=8:256 6:256}"
: "${FP8:=1}"
# Disable @torch.compile during `import deep_ep` to dodge nightly inductor bugs;
# DeepEP hot path is in CUDA kernels.
: "${DEEPEP_DISABLE_TORCH_COMPILE:=1}"
export DEEPEP_DISABLE_TORCH_COMPILE

: "${TRAYS:=pod4-gb300-2-tray01-f3 pod4-gb300-2-tray02-f3 pod4-gb300-2-tray03-f3 pod4-gb300-2-tray04-f3}"
: "${DEEPEP_DIR:=/home/fizhang/DeepEP}"
: "${NCCL_ROOT_DIR:=/home/fizhang/nccl/build}"
: "${DEEPEP_LOG_DIR:=/home/fizhang/deepep_logs}"
: "${MASTER_PORT:=8361}"
: "${MPI_HOME:=/usr/mpi/gcc/openmpi-4.1.9a1}"
: "${CUDA_HOME:=/usr/local/cuda}"
: "${EXTRA_ARGS:=}"

# Resolve a python interpreter that has `torch` importable. Uses the same
# probe as tray_build_deepep.sh; chatter goes to stderr, chosen path to stdout.
_has_torch() { "$1" -c "import torch" >/dev/null 2>&1; }
_collect_python_candidates() {
  local out=()
  [ -n "${PYTHON_BIN:-}" ] && out+=("$PYTHON_BIN")
  local p
  for p in python3 python python3.10 python3.11 python3.12 python3.13; do
    p=$(command -v "$p" 2>/dev/null || true); [ -n "$p" ] && out+=("$p")
  done
  local conda_pys=(
    "$HOME"/anaconda3/envs/*/bin/python "$HOME"/miniconda3/envs/*/bin/python
    "$HOME"/anaconda3/bin/python        "$HOME"/miniconda3/bin/python
    /opt/conda/envs/*/bin/python        /opt/conda/bin/python
    /opt/miniconda*/bin/python
    "$HOME"/.venv/bin/python "$HOME"/venv/bin/python
    "$HOME"/.local/bin/python* /usr/bin/python3.*
  )
  for p in "${conda_pys[@]}"; do [ -x "$p" ] && out+=("$p"); done
  local search_roots=("$HOME" /opt /usr/local /usr /scratch /shared /work /data)
  local sfs_root
  for sfs_root in /SFS-* /SFS /sfs /sfs-*; do
    [ -d "$sfs_root" ] && search_roots+=("$sfs_root")
  done
  local lib site_dir py_dir pyver venv_root
  for lib in $(find "${search_roots[@]}" -maxdepth 10 -name 'libtorch.so' 2>/dev/null \
                | grep -v '/enroot/data/' | head -30); do
    site_dir=$(dirname "$lib")
    site_dir=$(dirname "$(dirname "$site_dir")")
    py_dir=$(dirname "$(dirname "$site_dir")")
    pyver=$(basename "$(dirname "$site_dir")")
    [ -x "$py_dir/bin/$pyver" ]     && out+=("$py_dir/bin/$pyver")
    [ -x "$py_dir/bin/python3" ]    && out+=("$py_dir/bin/python3")
    [ -x "$py_dir/bin/python" ]     && out+=("$py_dir/bin/python")
    venv_root=$(dirname "$(dirname "$(dirname "$site_dir")")")
    [ -x "$venv_root/bin/python" ]  && out+=("$venv_root/bin/python")
    [ -x "$venv_root/bin/python3" ] && out+=("$venv_root/bin/python3")
    [ -x "$venv_root/bin/$pyver" ]  && out+=("$venv_root/bin/$pyver")
  done
  for p in $(find "${search_roots[@]}" -maxdepth 8 -name 'python3*' -type f -executable 2>/dev/null \
              | grep -v '/enroot/data/' \
              | grep -E '/(python|python3|python3\.[0-9]+)$' \
              | head -50); do
    out+=("$p")
  done
  printf '%s\n' "${out[@]}" \
    | grep -E '/(python|python3|python3\.[0-9]+)$' \
    | awk '!seen[$0]++'
}
_check_python() {
  local py="$1"
  _CHECK_PY_PATH=""; _CHECK_PY_LDLIB=""; _CHECK_PY_TVER=""
  "$py" -c 'import sys' >/dev/null 2>&1 || return 1
  local tver err
  tver=$("$py" -c 'import torch; print(torch.__version__)' 2>/dev/null || true)
  if [ -n "$tver" ]; then _CHECK_PY_TVER="$tver"; return 0; fi
  err=$("$py" -c 'import torch' 2>&1 | tail -n 1)
  printf '      ! %s plain import failed: %s\n' "$py" "$err" >&2
  local py_dir venv_root site torchlib
  py_dir=$(dirname "$py"); venv_root=$(dirname "$py_dir")
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
      _CHECK_PY_PATH="$site"; _CHECK_PY_LDLIB="$torchlib"; _CHECK_PY_TVER="$tver"; return 0
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
_resolve_python() {
  echo "[sweep] probing python interpreters for torch..." >&2
  local p chosen="" chosen_pp="" chosen_ld=""
  while IFS= read -r p; do
    [ -n "$p" ] || continue
    [ -x "$p" ] || { printf '  - %-50s  (not executable)\n' "$p" >&2; continue; }
    local ver
    ver=$("$p" -c 'import sys; print(sys.version.split()[0])' 2>/dev/null || echo "?")
    if _check_python "$p"; then
      if [ -n "$_CHECK_PY_PATH" ]; then
        printf '  + %-50s  python=%s torch=%s  <-- using (PYTHONPATH=%s LD_LIBRARY_PATH+=%s)\n' "$p" "$ver" "$_CHECK_PY_TVER" "$_CHECK_PY_PATH" "$_CHECK_PY_LDLIB" >&2
      else
        printf '  + %-50s  python=%s torch=%s  <-- using\n' "$p" "$ver" "$_CHECK_PY_TVER" >&2
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
    echo "[sweep] injected PYTHONPATH=$PYTHONPATH" >&2
  fi
  if [ -n "$chosen_ld" ]; then
    export LD_LIBRARY_PATH="$chosen_ld${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    echo "[sweep] injected LD_LIBRARY_PATH=$LD_LIBRARY_PATH" >&2
  fi
  return 0
}
if ! _resolve_python; then PYTHON_BIN=""; fi
if [ -z "${PYTHON_BIN:-}" ] || [ ! -x "$PYTHON_BIN" ] || ! "$PYTHON_BIN" -c "import torch" >/dev/null 2>&1; then
  err "no python interpreter with torch found; set PYTHON_BIN or activate your conda/venv"; exit 1
fi
export PYTHON_BIN
log "PYTHON_BIN     = $PYTHON_BIN"
[ -n "${PYTHONPATH:-}" ] && log "PYTHONPATH     = $PYTHONPATH"

# Slots per node, derived from TRAYS line one (use slots=N if hostfile already set)
: "${SLOTS_PER_NODE:=4}"

# ----- prep ------------------------------------------------------------------
[ -f "$DEEPEP_DIR/scripts/mpi_launch_ep.py" ] || {
  err "launcher missing: $DEEPEP_DIR/scripts/mpi_launch_ep.py - did you sync the repo?"
  exit 1
}
mkdir -p "$DEEPEP_LOG_DIR"

# Hostfile: tray-by-tray, ppr:4:node
HOSTFILE=$(mktemp -t deepep_hosts.XXXX)
trap 'rm -f "$HOSTFILE"' EXIT
for t in $TRAYS; do
  printf '%s slots=%s\n' "$t" "$SLOTS_PER_NODE"
done > "$HOSTFILE"

NUM_TRAYS=$(wc -l < "$HOSTFILE")
log "hostfile -> $HOSTFILE"
cat "$HOSTFILE" | sed 's/^/    /'

# Master = first tray (HOSTNAME or hostname -f if you're paranoid about NSS)
MASTER_ADDR=$(awk 'NR==1{print $1}' "$HOSTFILE")
log "MASTER_ADDR=$MASTER_ADDR MASTER_PORT=$MASTER_PORT"

# ----- run env (mirrors tray_build_deepep.sh) --------------------------------
export PATH="$MPI_HOME/bin:$CUDA_HOME/bin:${PATH:-}"
export LD_LIBRARY_PATH="$MPI_HOME/lib:$CUDA_HOME/lib64:$CUDA_HOME/extras/CUPTI/lib64:$NCCL_ROOT_DIR/lib:${LD_LIBRARY_PATH:-}"
export EP_NCCL_ROOT_DIR="$NCCL_ROOT_DIR"
export PYTHONPATH="$DEEPEP_DIR:${PYTHONPATH:-}"

# Useful debug toggles (off by default; flip via env if you need them)
: "${EP_BUFFER_DEBUG:=0}"
: "${NCCL_DEBUG:=WARN}"
export EP_BUFFER_DEBUG NCCL_DEBUG

# ----- pkill stale processes everywhere -------------------------------------
log "killing stale mpirun / python on all trays (pre-flight)"
for t in $TRAYS; do
  ssh -n -o StrictHostKeyChecking=accept-new -o LogLevel=ERROR "$t" \
    'pkill -9 mpirun  2>/dev/null; pkill -9 -f mpi_launch_ep 2>/dev/null; pkill -9 -f test_ep 2>/dev/null; true' || true
done

# ----- main sweep ------------------------------------------------------------
RUN_ID=$(date +%Y%m%d_%H%M%S)
SWEEP_TAG="${SWEEP_TAG:-deepep_sweep_$RUN_ID}"
log "SWEEP_TAG=$SWEEP_TAG  LOG_DIR=$DEEPEP_LOG_DIR"

# index of (EP, tokens, topk, experts) tuples we actually ran (for parser)
INDEX_FILE="$DEEPEP_LOG_DIR/$SWEEP_TAG.index.tsv"
: > "$INDEX_FILE"
printf 'tag\tep\ttokens\ttopk\texperts\tfp8\n' >> "$INDEX_FILE"

run_one() {
  local ep="$1" tokens="$2" topk="$3" experts="$4"
  local tag="${SWEEP_TAG}.ep${ep}.tok${tokens}.top${topk}.exp${experts}"
  log "==> EP=$ep tokens=$tokens topk=$topk experts=$experts  ($tag)"

  if (( ep > NUM_TRAYS * SLOTS_PER_NODE )); then
    warn "    skip: EP=$ep > available GPUs ($((NUM_TRAYS * SLOTS_PER_NODE)))"
    return 0
  fi
  if (( ep % SLOTS_PER_NODE != 0 )); then
    warn "    skip: EP=$ep not divisible by slots_per_node=$SLOTS_PER_NODE"
    return 0
  fi
  local need_nodes=$(( ep / SLOTS_PER_NODE ))

  # Sub-hostfile with only the trays we need (keeps mpirun honest at EP<full)
  local sub_hosts="$DEEPEP_LOG_DIR/$tag.hosts"
  head -n "$need_nodes" "$HOSTFILE" > "$sub_hosts"

  printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$tag" "$ep" "$tokens" "$topk" "$experts" "$FP8" >> "$INDEX_FILE"

  # build argv for test_ep.py (consumed via runpy in mpi_launch_ep.py)
  local args=(
    --num-tokens "$tokens"
    --hidden 7168
    --num-topk "$topk"
    --num-experts "$experts"
    --skip-check
    --test-first-only
    --num-processes "$SLOTS_PER_NODE"
  )
  # default 1 toggle is built into test_ep.py; pass through optional knobs
  # (we deliberately don't override allow_hybrid_mode / multiple_reduction here -
  #  the test_ep.py defaults to both = 1 which matches V3-style production setup)
  if [ -n "$EXTRA_ARGS" ]; then
    # shellcheck disable=SC2206
    local extra=($EXTRA_ARGS)
    args+=("${extra[@]}")
  fi

  set +e
  mpirun --hostfile "$sub_hosts" -np "$ep" --map-by "ppr:${SLOTS_PER_NODE}:node" \
    --mca pml ucx --mca btl ^openib \
    --mca plm_rsh_args "-i ${HOME}/.ssh/id_ed25519 -o StrictHostKeyChecking=accept-new -o LogLevel=ERROR" \
    -x PATH -x LD_LIBRARY_PATH -x CUDA_HOME -x MPI_HOME \
    -x PYTHONPATH -x EP_NCCL_ROOT_DIR \
    -x DEEPEP_DISABLE_TORCH_COMPILE \
    -x NCCL_DEBUG -x EP_BUFFER_DEBUG \
    -x MASTER_ADDR="$MASTER_ADDR" -x MASTER_PORT="$MASTER_PORT" \
    -x DEEPEP_LOG_DIR="$DEEPEP_LOG_DIR" -x DEEPEP_RUN_TAG="$tag" \
    -x DEEPEP_DIR="$DEEPEP_DIR" \
    "$PYTHON_BIN" -u "$DEEPEP_DIR/scripts/mpi_launch_ep.py" "${args[@]}"
  local rc=$?
  set -e
  if (( rc != 0 )); then
    warn "    mpirun exit=$rc for $tag  (continuing sweep; see logs)"
  fi

  # Kill any leftover on all trays before next config (avoid bench skew)
  for t in $TRAYS; do
    ssh -n -o StrictHostKeyChecking=accept-new -o LogLevel=ERROR "$t" \
      'pkill -9 -f mpi_launch_ep 2>/dev/null; pkill -9 -f test_ep 2>/dev/null; true' || true
  done
  sleep 1
}

for ep in $EP_SIZES; do
  for tok in $TOKENS; do
    for te in $TOPK_EXPERTS; do
      topk="${te%:*}"
      experts="${te#*:}"
      run_one "$ep" "$tok" "$topk" "$experts"
    done
  done
done

log "sweep complete. logs:  $DEEPEP_LOG_DIR/$SWEEP_TAG.*.log"
log "index file:           $INDEX_FILE"
log "next:  $PYTHON_BIN $DEEPEP_DIR/scripts/parse_deepep_csv.py --log-dir $DEEPEP_LOG_DIR --tag $SWEEP_TAG"
