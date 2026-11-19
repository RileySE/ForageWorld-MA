#!/usr/bin/env bash
# Usage: run_central_ippo_rnn.sh [config_file]
#   config_file defaults to central_ippo_rnn.yaml and is resolved inside
#   baselines/config/. For the separate-vs-central comparison:
#     ./run_central_ippo_rnn.sh predators_20_moving_central.yaml
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CONFIG_FILE="${1:-central_ippo_rnn.yaml}"

cd "${REPO_ROOT}"
exec "${PYTHON:-python}" baselines/central_ippo_rnn.py --config_file "${CONFIG_FILE}"
