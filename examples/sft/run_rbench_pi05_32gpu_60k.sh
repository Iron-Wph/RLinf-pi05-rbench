#!/usr/bin/env bash
set -Eeuo pipefail
if (( $# < 3 )); then
    echo "Usage: $0 RUN_DIR VENV HEAD_IP [Hydra overrides...]" >&2
    exit 2
fi
run="$1"
venv="$2"
head_ip="$3"
shift 3
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
mkdir -p "$run"
run="$(cd "$run" && pwd)"
: "${RBENCH_DATASET_ROOT:?Set the LeRobot dataset root}"
: "${RBENCH_BASE_MODEL:?Set the converted pi05_base directory}"
: "${RBENCH_NORM_STATS:?Set the RBench norm_stats.json path}"
: "${RBENCH_LOG_ROOT:?Set a shared checkpoint and log root}"
cd "$repo"
source "$venv/bin/activate"
export RAY_ADDRESS="$head_ip:6379" RLINF_NODE_RANK=0 PYTHONUNBUFFERED=1
export PYTHONPATH="$PWD" EMBODIED_PATH="$PWD/examples/sft"
export JAX_PLATFORMS=cpu HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-$run/openpi-cache}"
trap 'status=$?; printf "%s\n" "$status" > "$run/train.exit"' EXIT
exec > >(tee -a "$run/train.log") 2>&1
python - <<'PY'
import ray

ray.init(address="auto", log_to_driver=False)
nodes = {
    entry["NodeManagerAddress"]: entry["Resources"].get("GPU", 0)
    for entry in ray.nodes() if entry["Alive"]
}
assert sorted(nodes.values()) == [8] * 4, nodes
print("RAY_READY", nodes, flush=True)
ray.shutdown()
PY
python examples/sft/train_vla_sft.py --config-dir "$EMBODIED_PATH/config" --config-name rbench_sft_openpi_pi05_32gpu_60k "$@"
