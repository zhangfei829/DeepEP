#!/usr/bin/env bash
# jumphost_deepep_sweep.sh - one-button DeepEP V2 sweep, run from the jumphost.
#
# Pipeline (6 stages):
#   [1/6] Generate (or verify) head_tray's ~/.ssh/id_ed25519 keypair
#   [2/6] Distribute head_tray's pubkey to every tray's authorized_keys
#   [3/6] git pull / sync $DEEPEP_DIR on head_tray, then build via tray_build_deepep.sh
#   [4/6] Run the sweep matrix via tray_deepep_sweep.sh on head_tray
#   [5/6] scp logs (and the index file) back to the jumphost
#   [6/6] Parse logs into a summary CSV + markdown table
#
# Run from: jumphost (hungry-hippo-fin-03-jumphost)
#
# Tunables (env):
#   HEAD_TRAY      head tray hostname            -- default pod4-gb300-2-tray01-f3
#   TRAYS          space-separated tray list     -- default 4-tray pool (must include HEAD_TRAY)
#   DEEPEP_DIR     repo on shared FS visible to all trays  -- default /home/fizhang/DeepEP
#   NCCL_ROOT_DIR  pre-built NCCL                -- default /home/fizhang/nccl/build
#   DEEPEP_LOG_DIR per-rank log dir on tray      -- default /home/fizhang/deepep_logs
#   JH_LOG_DIR     where to dump logs on jh      -- default /tmp/deepep_logs.$RUN_ID
#   GIT_REMOTE     optional git URL to clone if $DEEPEP_DIR missing
#   GIT_REF        optional git ref to checkout  -- default origin/master
#   SKIP_BUILD     1 to skip stage 3 (when iterating on .py only)
#   SKIP_SSH_SETUP 1 to skip stages 1-2 (already done)
#   EP_SIZES/TOKENS/TOPK_EXPERTS/EXTRA_ARGS    forwarded to tray_deepep_sweep.sh
#
# Example end-to-end:
#   TRAYS="pod4-gb300-2-tray01-f3 pod4-gb300-2-tray02-f3 pod4-gb300-2-tray03-f3 pod4-gb300-2-tray04-f3" \
#   bash scripts/jumphost_deepep_sweep.sh 2>&1 | tee /tmp/deepep_sweep.log
set -euo pipefail

log()  { printf '\033[1;36m[jh:%s]\033[0m %s\n' "$STAGE" "$*"; }
warn() { printf '\033[1;33m[jh:%s:WARN]\033[0m %s\n' "$STAGE" "$*" >&2; }
err()  { printf '\033[1;31m[jh:%s:ERR]\033[0m %s\n' "$STAGE" "$*" >&2; }

: "${HEAD_TRAY:=pod4-gb300-2-tray01-f3}"
: "${TRAYS:=pod4-gb300-2-tray01-f3 pod4-gb300-2-tray02-f3 pod4-gb300-2-tray03-f3 pod4-gb300-2-tray04-f3}"
: "${DEEPEP_DIR:=/home/fizhang/DeepEP}"
: "${NCCL_ROOT_DIR:=/home/fizhang/nccl/build}"
: "${DEEPEP_LOG_DIR:=/home/fizhang/deepep_logs}"
: "${RUN_ID:=$(date +%Y%m%d_%H%M%S)}"
: "${JH_LOG_DIR:=/tmp/deepep_logs.$RUN_ID}"
: "${GIT_REF:=origin/master}"
: "${SKIP_BUILD:=0}"
: "${SKIP_SSH_SETUP:=0}"

# Jumphost SSH key for hopping into trays
: "${JH_SSH_KEY:=$HOME/id_ed25519}"
[ -f "$JH_SSH_KEY" ] || { STAGE=pre err "jumphost ssh key not found: $JH_SSH_KEY"; exit 1; }

# Common ssh wrapper.
# JSSH:    no `-n`; safe for `bash <<HEREDOC` style invocations.
# JSSH_N:  with `-n`; use this inside `for t in $TRAYS; do ssh ... done` loops
#          so stdin isn't consumed by the for-loop iterator.
JSSH=(ssh   -i "$JH_SSH_KEY" -o StrictHostKeyChecking=accept-new -o LogLevel=ERROR)
JSSH_N=(ssh -n -i "$JH_SSH_KEY" -o StrictHostKeyChecking=accept-new -o LogLevel=ERROR)

# Sanity: HEAD_TRAY must be in TRAYS
case " $TRAYS " in *" $HEAD_TRAY "*) ;; *) STAGE=pre err "HEAD_TRAY=$HEAD_TRAY not in TRAYS=$TRAYS"; exit 1;; esac

mkdir -p "$JH_LOG_DIR"
STAGE=pre log "RUN_ID=$RUN_ID JH_LOG_DIR=$JH_LOG_DIR HEAD_TRAY=$HEAD_TRAY TRAYS=($TRAYS)"

#==============================================================================
# [1/6] Setup head-tray ssh key
#==============================================================================
STAGE=1
if [ "$SKIP_SSH_SETUP" = "1" ]; then
  log "SKIP_SSH_SETUP=1, skipping ssh key generation"
else
  log "ensure head_tray $HEAD_TRAY has a working ~/.ssh/id_ed25519"
  "${JSSH[@]}" "fizhang@$HEAD_TRAY" bash -l <<'REMOTE'
set -euo pipefail
mkdir -p ~/.ssh && chmod 700 ~/.ssh
if [ ! -f ~/.ssh/id_ed25519 ]; then
  ssh-keygen -t ed25519 -N '' -f ~/.ssh/id_ed25519
fi
# Stash pubkey on the shared FS so other trays can read it without ssh
cp -f ~/.ssh/id_ed25519.pub /home/fizhang/head_tray_pub.txt
chmod 644 /home/fizhang/head_tray_pub.txt
echo "head_tray pubkey: $(cat /home/fizhang/head_tray_pub.txt)"
REMOTE
fi

#==============================================================================
# [2/6] Install head pubkey on every tray + verify pairwise ssh
#==============================================================================
STAGE=2
if [ "$SKIP_SSH_SETUP" = "1" ]; then
  log "SKIP_SSH_SETUP=1, skipping pubkey distribution"
else
  for t in $TRAYS; do
    log "installing head pubkey on $t"
    # one-liner avoids needing stdin for a heredoc inside the for-loop
    "${JSSH_N[@]}" "fizhang@$t" '
set -euo pipefail
mkdir -p ~/.ssh && chmod 700 ~/.ssh
touch ~/.ssh/authorized_keys && chmod 600 ~/.ssh/authorized_keys
grep -qxFf /home/fizhang/head_tray_pub.txt ~/.ssh/authorized_keys \
  || cat /home/fizhang/head_tray_pub.txt >> ~/.ssh/authorized_keys
'
  done

  log "verifying head_tray -> each tray ssh works (head_tray to spawn mpirun)"
  for t in $TRAYS; do
    if ! "${JSSH_N[@]}" "fizhang@$HEAD_TRAY" "ssh -n -o StrictHostKeyChecking=accept-new -o LogLevel=ERROR -o BatchMode=yes $t 'hostname'" 2>/dev/null; then
      err "ssh from $HEAD_TRAY to $t failed"
      exit 1
    fi
  done
  log "ssh mesh OK"
fi

#==============================================================================
# [3/6] Sync repo + build on head_tray
#==============================================================================
STAGE=3
if [ "$SKIP_BUILD" = "1" ]; then
  log "SKIP_BUILD=1, skipping repo sync + build"
else
  log "syncing repo $DEEPEP_DIR on $HEAD_TRAY (ref=$GIT_REF)"
  GIT_REMOTE_VAL="${GIT_REMOTE:-}"
  "${JSSH[@]}" "fizhang@$HEAD_TRAY" \
    DEEPEP_DIR="$DEEPEP_DIR" GIT_REF="$GIT_REF" GIT_REMOTE="$GIT_REMOTE_VAL" \
    bash -l <<'REMOTE'
set -euo pipefail
mkdir -p "$(dirname "$DEEPEP_DIR")"
if [ ! -d "$DEEPEP_DIR/.git" ]; then
  if [ -z "$GIT_REMOTE" ]; then
    echo "[3] DEEPEP_DIR=$DEEPEP_DIR has no .git and GIT_REMOTE not given; aborting" >&2
    exit 1
  fi
  echo "[3] cloning $GIT_REMOTE -> $DEEPEP_DIR"
  git clone "$GIT_REMOTE" "$DEEPEP_DIR"
fi
cd "$DEEPEP_DIR"
# Make sure 'origin' points at the fork the user actually wants; otherwise
# `git reset --hard origin/main` would silently sync to whatever upstream URL
# was clone'd in the past (e.g. deepseek-ai/DeepEP), missing our scripts/.
if [ -n "$GIT_REMOTE" ]; then
  current_origin=$(git remote get-url origin 2>/dev/null || echo "")
  if [ "$current_origin" != "$GIT_REMOTE" ]; then
    echo "[3] origin URL '$current_origin' != GIT_REMOTE '$GIT_REMOTE'; updating"
    git remote set-url origin "$GIT_REMOTE"
  fi
fi
git fetch --all --tags --prune
git reset --hard "$GIT_REF"
echo "[3] HEAD = $(git rev-parse --short HEAD)  ($(git log -1 --pretty='%s'))"
ls scripts/ 2>/dev/null | head -n 10
[ -f scripts/tray_build_deepep.sh ] || { echo "[3] scripts/tray_build_deepep.sh missing after reset; ref/remote wrong?" >&2; exit 1; }
REMOTE

  log "building DeepEP via tray_build_deepep.sh on $HEAD_TRAY"
  "${JSSH[@]}" "fizhang@$HEAD_TRAY" \
    DEEPEP_DIR="$DEEPEP_DIR" NCCL_ROOT_DIR="$NCCL_ROOT_DIR" \
    PYTHON_BIN="${PYTHON_BIN:-}" DISABLE_LEGACY="${DISABLE_LEGACY:-1}" \
    bash -l "$DEEPEP_DIR/scripts/tray_build_deepep.sh" 2>&1 \
    | sed 's/^/    /'
fi

#==============================================================================
# [4/6] Run sweep on head_tray
#==============================================================================
STAGE=4
log "running sweep matrix on $HEAD_TRAY"
SWEEP_TAG="deepep_sweep_$RUN_ID"
"${JSSH[@]}" "fizhang@$HEAD_TRAY" \
  DEEPEP_DIR="$DEEPEP_DIR" \
  NCCL_ROOT_DIR="$NCCL_ROOT_DIR" \
  DEEPEP_LOG_DIR="$DEEPEP_LOG_DIR" \
  TRAYS="$TRAYS" \
  SWEEP_TAG="$SWEEP_TAG" \
  PYTHON_BIN="${PYTHON_BIN:-}" \
  EP_SIZES="${EP_SIZES:-4 8 16}" \
  TOKENS="${TOKENS:-1024 2048 4096 8192}" \
  TOPK_EXPERTS="${TOPK_EXPERTS:-8:256 6:256}" \
  FP8="${FP8:-1}" \
  EXTRA_ARGS="${EXTRA_ARGS:-}" \
  bash -l "$DEEPEP_DIR/scripts/tray_deepep_sweep.sh" 2>&1 \
  | sed 's/^/    /'

#==============================================================================
# [5/6] Pull logs back to jumphost
#==============================================================================
STAGE=5
log "scp logs back to $JH_LOG_DIR"
scp -i "$JH_SSH_KEY" -o StrictHostKeyChecking=accept-new -o LogLevel=ERROR -q \
  "fizhang@${HEAD_TRAY}:${DEEPEP_LOG_DIR}/${SWEEP_TAG}.*" \
  "$JH_LOG_DIR/" || { warn "scp failed; check $DEEPEP_LOG_DIR on $HEAD_TRAY"; }

#==============================================================================
# [6/6] Parse + print comparison table
#==============================================================================
STAGE=6
PARSE_SCRIPT="$(dirname "$0")/parse_deepep_csv.py"
if [ ! -f "$PARSE_SCRIPT" ]; then
  # fall back to scp'd copy (jumphost repo may not exist)
  scp -i "$JH_SSH_KEY" -o StrictHostKeyChecking=accept-new -o LogLevel=ERROR -q \
    "fizhang@${HEAD_TRAY}:${DEEPEP_DIR}/scripts/parse_deepep_csv.py" \
    "$JH_LOG_DIR/parse_deepep_csv.py"
  PARSE_SCRIPT="$JH_LOG_DIR/parse_deepep_csv.py"
fi

JH_PY=""
for cand in python3 python; do
  if command -v "$cand" >/dev/null 2>&1; then JH_PY="$cand"; break; fi
done

if [ -f "$PARSE_SCRIPT" ] && [ -n "$JH_PY" ]; then
  "$JH_PY" "$PARSE_SCRIPT" --log-dir "$JH_LOG_DIR" --tag "$SWEEP_TAG" \
    | tee "$JH_LOG_DIR/$SWEEP_TAG.summary.md"
  log "summary csv:  $JH_LOG_DIR/$SWEEP_TAG.summary.csv"
  log "summary md:   $JH_LOG_DIR/$SWEEP_TAG.summary.md"
else
  warn "parse_deepep_csv.py not found, leaving raw logs in $JH_LOG_DIR"
fi

log "done. raw logs:  $JH_LOG_DIR/${SWEEP_TAG}.*"
