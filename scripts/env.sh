# Source this file: activates the shared ROCm venv and exports the runtime environment.
# Override the venv location with VENV_DIR=/path/to/venv.

_BGEOPT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${VENV_DIR:-${_BGEOPT_ROOT}/../.venv}"

if [ -f "${VENV_DIR}/bin/activate" ]; then
  # shellcheck disable=SC1091
  source "${VENV_DIR}/bin/activate"
else
  echo "venv not found at ${VENV_DIR} (set VENV_DIR)" >&2
  return 1 2>/dev/null || exit 1
fi

# ROCm under WSL2 sees the GPU only through /dev/dxg with this flag (it lives in ~/.bashrc for interactive shells only)
export HSA_ENABLE_DXG_DETECTION=1
export PYTHONPATH="${_BGEOPT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export HF_HUB_DISABLE_TELEMETRY=1
# Rust tokenizer threads: 4x faster CPU-side prepare (no forked workers in the pipeline, so it is safe)
export TOKENIZERS_PARALLELISM=true
