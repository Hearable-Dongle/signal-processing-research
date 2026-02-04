#!/bin/bash
set -e

# Default values
BASE_NAME=${1:-"convtas"}
# Output filename goes to basename if not defined
OUTPUT_FILENAME=${2:-"${BASE_NAME}.har"}

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
echo "=========================================="

echo "[Step 1] Exporting Pytorch to ONNX..."
$ONNX_PYTHON -m hailo.convtasnet_to_onnx "${RAW_ONNX}"
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
$HAILO_PYTHON -m hailo.onnx_to_hailo "${PATCHED_ONNX}" "${HAR_OUTPUT}" --model_name "${BASE_NAME}"
if [ $? -eq 0 ]; then
    echo "Step 3 Success: Created ${HAR_OUTPUT}"
else
    echo "Step 3 Failed"
    exit 1
fi

echo "=========================================="
echo "Flow Complete!"
echo "Final Output: ${HAR_OUTPUT}"
echo "=========================================="
