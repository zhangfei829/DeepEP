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

: "${TRAYS:=pod4-gb300-2-tray01-f3 pod4-gb300-2-tray02-f3 pod4-gb300-2-tray03-f3 pod4-gb300-2-tray04-f3}"
: "${DEEPEP_DIR:=/home/fizhang/DeepEP}"
: "${NCCL_ROOT_DIR:=/home/fizhang/nccl/build}"
: "${DEEPEP_LOG_DIR:=/home/fizhang/deepep_logs}"
: "${MASTER_PORT:=8361}"
: "${MPI_HOME:=/usr/mpi/gcc/openmpi-4.1.9a1}"
: "${CUDA_HOME:=/usr/local/cuda}"
: "${EXTRA_ARGS:=}"

# Resolve a python interpreter that has `torch` importable.
# Same probe logic as tray_build_deepep.sh -- kept inline for self-containment.
_has_torch() { "$1" -c "import torch" >/dev/null 2>&1; }
_resolve_python() {
  local p
  if [ -n "${PYTHON_BIN:-}" ] && _has_torch "$PYTHON_BIN"; then echo "$PYTHON_BIN"; return 0; fi
  for p in python3 python; do
    if command -v "$p" >/dev/null 2>&1 && _has_torch "$p"; then command -v "$p"; return 0; fi
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
  )
  for p in "${conda_pys[@]}"; do
    [ -x "$p" ] || continue
    if _has_torch "$p"; then echo "$p"; return 0; fi
  done
  for p in $(find "$HOME" /opt /usr/local -maxdepth 5 -name python3 -type f -executable 2>/dev/null); do
    if _has_torch "$p"; then echo "$p"; return 0; fi
  done
  return 1
}
if PYTHON_BIN=$(_resolve_python); then export PYTHON_BIN
else err "no python interpreter with torch found; set PYTHON_BIN or activate your conda/venv"; exit 1; fi
log "PYTHON_BIN     = $PYTHON_BIN"

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
