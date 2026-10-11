#!/bin/bash
# Shared helpers for campaign scripts. Source it as:
#   source "$(dirname "$0")/../lib/common.sh"

set -euo pipefail
export HYDRA_FULL_ERROR=1

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TRAIN="pixi run python src/optim_env/tools/train.py"
TEST="pixi run python src/optim_env/tools/test.py"
HPO="pixi run python src/optim_env/tools/hpo.py"

section() { echo; echo "=> $*"; }

# write_manifest <campaign> [key=value ...]
# Records what a campaign actually ran (git SHA, time, parameters) next to its report, so
# results stay traceable and a later rerun with different budgets is distinguishable.
write_manifest() {
    local campaign="$1"; shift
    local out="$REPO_ROOT/outputs/reports/${campaign}_manifest.json"
    mkdir -p "$(dirname "$out")"
    {
        echo "{"
        echo "  \"campaign\": \"$campaign\","
        echo "  \"timestamp\": \"$(date -Iseconds)\","
        echo "  \"git_sha\": \"$(git -C "$REPO_ROOT" rev-parse HEAD 2>/dev/null || echo unknown)\","
        echo "  \"git_dirty\": $(test -n "$(git -C "$REPO_ROOT" status --porcelain 2>/dev/null)" && echo true || echo false),"
        echo "  \"params\": {"
        local first=1
        for kv in "$@"; do
            [ $first -eq 0 ] && echo ","
            first=0
            printf '    "%s": "%s"' "${kv%%=*}" "${kv#*=}"
        done
        echo
        echo "  }"
        echo "}"
    } > "$out"
    echo "Wrote manifest -> $out"
}
