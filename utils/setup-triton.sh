#!/usr/bin/env bash
set -eo pipefail

# =========================
# Config
# =========================
ENV_NAME="mls"
PYTHON_VERSION="3.11"
MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
MINICONDA_INSTALL_DIR="${HOME}/miniconda3"

# Parse command line arguments
AUTO_YES=false
while [[ $# -gt 0 ]]; do
	case $1 in
		-y|--yes)
			AUTO_YES=true
			shift
			;;
		-h|--help)
			echo "Usage: $0 [OPTIONS]"
			echo ""
			echo "Options:"
			echo "  -y, --yes    Non-interactive mode, answer yes to all prompts"
			echo "  -h, --help   Show this help message"
			exit 0
			;;
		*)
			echo "Unknown option: $1"
			exit 1
			;;
	esac
done

# =========================
# Helper functions
# =========================
ask_continue() {
	local prompt="${1:-Continue?}"
	if [ "${AUTO_YES}" = true ]; then
		echo ">>> ${prompt} [Y/n] y (auto)"
		return 0
	fi
	read -rp ">>> ${prompt} [Y/n] " answer
	case "${answer}" in
	[nN] | [nN][oO])
		echo ">>> Aborted by user."
		exit 1
		;;
	*) ;;
	esac
}

# =========================
# Detect GPU Architecture
# =========================
detect_gpu_arch() {
	echo ">>> Detecting GPU architecture..."

	if ! command -v nvidia-smi >/dev/null 2>&1; then
		echo "    WARNING: nvidia-smi not found, cannot detect GPU"
		IS_BLACKWELL=false
		return
	fi

	GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n1)
	echo "    GPU detected: ${GPU_NAME}"

	if echo "${GPU_NAME}" | grep -qiE "(B100|B200|GB200|RTX 50|Blackwell)"; then
		IS_BLACKWELL=true
		echo "    Architecture: Blackwell (CC 10.x)"
	else
		IS_BLACKWELL=false
		echo "    Architecture: Non-Blackwell"
	fi
}

echo ">>> Triton setup:"
echo "    - Installs NumPy, PyTorch, Triton"
echo "    - Uses PyTorch nightly for Blackwell (sm_120)"
echo

detect_gpu_arch
echo
ask_continue "Proceed with Triton setup?"

# =========================
# Check / Install conda
# =========================
if command -v conda >/dev/null 2>&1; then
	echo ">>> conda found: $(conda --version)"
	eval "$(conda shell.bash hook)"
elif [ -x "${MINICONDA_INSTALL_DIR}/bin/conda" ]; then
	echo ">>> conda found at ${MINICONDA_INSTALL_DIR}/bin/conda"
	eval "$("${MINICONDA_INSTALL_DIR}/bin/conda" shell.bash hook)"
elif [ -x /opt/conda/bin/conda ]; then
	echo ">>> conda found at /opt/conda/bin/conda"
	eval "$(/opt/conda/bin/conda shell.bash hook)"
else
	echo ">>> conda not found."
	ask_continue "Install Miniconda to ${MINICONDA_INSTALL_DIR}?"

	MINICONDA_INSTALLER="/tmp/miniconda_installer.sh"
	curl -fsSL "${MINICONDA_URL}" -o "${MINICONDA_INSTALLER}"
	bash "${MINICONDA_INSTALLER}" -b -p "${MINICONDA_INSTALL_DIR}"
	rm -f "${MINICONDA_INSTALLER}"

	eval "$("${MINICONDA_INSTALL_DIR}/bin/conda" shell.bash hook)"

	"${MINICONDA_INSTALL_DIR}/bin/conda" init bash
	"${MINICONDA_INSTALL_DIR}/bin/conda" init zsh
	echo ">>> Miniconda installed at ${MINICONDA_INSTALL_DIR}"
	echo ">>> Please restart your shell or run: source ~/.bashrc (or ~/.zshrc)"
fi

# =========================
# Accept conda Terms of Service
# =========================
echo ">>> Accepting conda channel Terms of Service"
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main 2>/dev/null || true
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r 2>/dev/null || true

# =========================
# Create conda environment
# =========================
if conda env list | grep -q "^${ENV_NAME} "; then
	echo ">>> Found existing conda environment: ${ENV_NAME}"
	ask_continue "Reuse existing environment?"
else
	echo ">>> Will create conda environment: ${ENV_NAME} (Python ${PYTHON_VERSION})"
	ask_continue "Create new conda environment?"
	conda create -y -n "${ENV_NAME}" python="${PYTHON_VERSION}" --override-channels -c conda-forge
fi

if [ "${AUTO_YES}" = true ]; then
	CONDA_ENV_PATH=$(conda info --envs | grep "^${ENV_NAME} " | awk '{print $NF}')
	export PATH="${CONDA_ENV_PATH}/bin:${PATH}"
	export CONDA_PREFIX="${CONDA_ENV_PATH}"
	echo ">>> Activated environment: ${ENV_NAME}"
else
	conda activate "${ENV_NAME}"
fi

# =========================
# Install Triton stack
# =========================
echo ">>> Installing NumPy"
pip install numpy

echo ">>> Installing PyTorch (required by Triton)"
if [ "${IS_BLACKWELL}" = true ]; then
	pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu128
else
	pip install torch --index-url https://download.pytorch.org/whl/cu124
fi

echo ">>> Installing Triton"
pip install triton

# =========================
# Validate key packages (non-fatal)
# =========================
echo ">>> Validating key packages (Triton)"
python - <<'PY'
import importlib

def check_any(module_names):
    last_exc = None
    for module_name in module_names:
        try:
            mod = importlib.import_module(module_name)
            version = getattr(mod, "__version__", "unknown")
            return True, f"{module_name} {version}"
        except Exception as exc:
            last_exc = exc
    return False, str(last_exc) if last_exc else "not found"

checks = {
    "triton": check_any(["triton"]),
    "numpy": check_any(["numpy"]),
    "torch": check_any(["torch"]),
}

print("    Package status:")
for name in ("triton", "numpy", "torch"):
    ok, info = checks[name]
    status = "OK" if ok else "FAIL"
    print(f"    - {name}: {status} ({info})")
PY

# =========================
# Done
# =========================
echo
echo "============================================="
echo " Triton environment is ready."
echo "============================================="
echo
echo "Activate with:"
echo "  conda activate ${ENV_NAME}"
echo
echo "Installed key packages:"
echo "  - triton"
echo "  - numpy"
if [ "${IS_BLACKWELL}" = true ]; then
	echo "  - torch (nightly, cu128 for Blackwell)"
else
	echo "  - torch (stable, cu124)"
fi
echo
