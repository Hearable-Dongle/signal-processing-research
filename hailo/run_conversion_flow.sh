#!/bin/bash
set -e

# Default values
BASE_NAME=${1:-"convtas"}
# Output filename goes to basename if not defined
OUTPUT_FILENAME=${2:-"${BASE_NAME}.har"}
HW_ARCH=${HW_ARCH:-"hailo8"}
NORM_MODE=${NORM_MODE:-"channel"}
EXPORT_PROFILE=${EXPORT_PROFILE:-"hailo_safe"}
ACT_REPLACE=${ACT_REPLACE:-"relu"}
NORM_REPLACE=${NORM_REPLACE:-"identity"}
NORM_EPS=${NORM_EPS:-"1e-8"}
DISABLE_SKIP=${DISABLE_SKIP:-"false"}
MASK_MUL_MODE=${MASK_MUL_MODE:-"normal"}
FORCE_N_SRC_1=${FORCE_N_SRC_1:-"false"}
BYPASS_CONCAT=${BYPASS_CONCAT:-"false"}
SKIP_TOPOLOGY_MODE=${SKIP_TOPOLOGY_MODE:-"concat"}
DECONV_MODE=${DECONV_MODE:-"grouped"}
TRUNCATE_K_BLOCKS=${TRUNCATE_K_BLOCKS:-"0"}
CALIB_NPZ=${CALIB_NPZ:-""}
MODEL_SCRIPT_FILE=${MODEL_SCRIPT_FILE:-""}
COMPILER_OPT_LEVEL=${COMPILER_OPT_LEVEL:-""}
LOG_FAILED_LAYERS_PATH=${LOG_FAILED_LAYERS_PATH:-""}
STOP_AFTER_HAR=${STOP_AFTER_HAR:-"false"}

SCRIPT_DIR=$(dirname "$0")
ROOT_DIR="$SCRIPT_DIR/.."

# Move to project root
if [[ "$PWD" == */hailo ]]; then
    cd ..
fi

# Interpreters
ONNX_PYTHON="hailo/to-onnx-env/bin/python"
HAILO_PYTHON="hailo/to-hailo-env/bin/python"

# Intermediate file names
RAW_ONNX="hailo/${BASE_NAME}_raw.onnx"
PATCHED_ONNX="hailo/${BASE_NAME}_patched.onnx"
HAR_OUTPUT="hailo/${OUTPUT_FILENAME}"

echo "=========================================="
echo "Starting Conversion Flow"
echo "Base Name: ${BASE_NAME}"
echo "Output HAR: ${HAR_OUTPUT}"
echo "Target HW: ${HW_ARCH}"
echo "Norm Mode: ${NORM_MODE}"
echo "Export Profile: ${EXPORT_PROFILE}"
echo "Act Replace: ${ACT_REPLACE}"
echo "Norm Replace: ${NORM_REPLACE}"
echo "Disable Skip: ${DISABLE_SKIP}"
echo "Mask Mul Mode: ${MASK_MUL_MODE}"
echo "Force N Src 1: ${FORCE_N_SRC_1}"
echo "Bypass Concat: ${BYPASS_CONCAT}"
echo "Skip Topology Mode: ${SKIP_TOPOLOGY_MODE}"
echo "Deconv Mode: ${DECONV_MODE}"
echo "Truncate K Blocks: ${TRUNCATE_K_BLOCKS}"
echo "=========================================="

echo "[Step 1] Exporting Pytorch to ONNX..."
$ONNX_PYTHON -m hailo.convtasnet_to_onnx "${RAW_ONNX}" \
    --norm_mode "${NORM_MODE}" \
    --export_profile "${EXPORT_PROFILE}" \
    --act_replace "${ACT_REPLACE}" \
    --norm_replace "${NORM_REPLACE}" \
    --norm_eps "${NORM_EPS}" \
    --disable_skip "${DISABLE_SKIP}" \
    --mask_mul_mode "${MASK_MUL_MODE}" \
    --force_n_src_1 "${FORCE_N_SRC_1}" \
    --bypass_concat "${BYPASS_CONCAT}" \
    --skip_topology_mode "${SKIP_TOPOLOGY_MODE}" \
    --deconv_mode "${DECONV_MODE}" \
    --truncate_k_blocks "${TRUNCATE_K_BLOCKS}"
if [ $? -eq 0 ]; then
    echo "Step 1 Success: Created ${RAW_ONNX}"
else
    echo "Step 1 Failed"
    exit 1
fi

echo "[Step 2] Patching ONNX model..."
$ONNX_PYTHON -m hailo.patch_onnx "${RAW_ONNX}" "${PATCHED_ONNX}"
if [ $? -eq 0 ]; then
    echo "Step 2 Success: Created ${PATCHED_ONNX}"
else
    echo "Step 2 Failed"
    exit 1
fi

echo "[Step 3] Converting ONNX to HAR..."
# Pass model_name as BASE_NAME so internal layers are named consistently if possible
$HAILO_PYTHON -m hailo.onnx_to_hailo "${PATCHED_ONNX}" "${HAR_OUTPUT}" --model_name "${BASE_NAME}" --hw_arch "${HW_ARCH}"
if [ $? -eq 0 ]; then
    echo "Step 3 Success: Created ${HAR_OUTPUT}"
else
    echo "Step 3 Failed"
    exit 1
fi

if [[ "${STOP_AFTER_HAR}" == "true" ]]; then
    echo "STOP_AFTER_HAR=true, skipping Step 4 (HEF compile)"
    echo "=========================================="
    echo "Flow Complete (HAR only)"
    echo "Final Output: ${HAR_OUTPUT}"
    echo "=========================================="
    exit 0
fi

echo "[Step 4] Optimizing and compiling HAR to HEF..."
HEF_OUTPUT="hailo/${BASE_NAME}.hef"
HEF_CMD=(
    "$HAILO_PYTHON" -m hailo.har_to_hef "${HAR_OUTPUT}" "${HEF_OUTPUT}"
    --model_name "${BASE_NAME}"
    --hw_arch "${HW_ARCH}"
)

if [[ -n "${CALIB_NPZ}" ]]; then
    HEF_CMD+=(--calib_npz "${CALIB_NPZ}")
fi

if [[ -n "${MODEL_SCRIPT_FILE}" ]]; then
    HEF_CMD+=(--model_script_file "${MODEL_SCRIPT_FILE}")
fi

if [[ -n "${COMPILER_OPT_LEVEL}" ]]; then
    HEF_CMD+=(--compiler_optimization_level "${COMPILER_OPT_LEVEL}")
fi

if [[ -n "${LOG_FAILED_LAYERS_PATH}" ]]; then
    HEF_CMD+=(--log_failed_layers_path "${LOG_FAILED_LAYERS_PATH}")
fi

"${HEF_CMD[@]}"
if [ $? -eq 0 ]; then
    echo "Step 4 Success: Created ${HEF_OUTPUT}"
else
    echo "Step 4 Failed"
    exit 1
fi

echo "=========================================="
echo "Flow Complete!"
echo "Final Output: ${HEF_OUTPUT}"
echo "=========================================="
