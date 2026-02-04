#!/bin/bash
set -e

# Default values
BASE_NAME=${1:-"convtas"}
# If second argument is provided, use it. Otherwise default to BASE_NAME.hef
OUTPUT_FILENAME=${2:-"${BASE_NAME}.hef"}

# Directory paths
SCRIPT_DIR=$(dirname "$0")
ROOT_DIR="$SCRIPT_DIR/.."
# Resolve absolute paths or run from root. 
# We assume this script is run from the project root or we adjust paths.
# The python scripts assume they are run from project root (e.g. "hailo/file.py")
# So we should cd to project root or keep paths relative to it.

# Let's ensure we are relative to project root if the user runs from hailo/
if [[ "$PWD" == */hailo ]]; then
    cd ..
fi

# Define Python Interpreters
ONNX_PYTHON="hailo/to-onnx-env/bin/python"
HAILO_PYTHON="hailo/to-hailo-env/bin/python"

# Intermediate File Names
RAW_ONNX="hailo/${BASE_NAME}_raw.onnx"
PATCHED_ONNX="hailo/${BASE_NAME}_patched.onnx"
HEF_OUTPUT="hailo/${OUTPUT_FILENAME}"

echo "=========================================="
echo "Starting Conversion Flow"
echo "Base Name: ${BASE_NAME}"
echo "Output HEF: ${HEF_OUTPUT}"
echo "=========================================="

# Step 1: Pytorch -> ONNX
echo "[Step 1] Exporting Pytorch to ONNX..."
$ONNX_PYTHON hailo/convtasnet_to_onnx.py "${RAW_ONNX}"
if [ $? -eq 0 ]; then
    echo "Step 1 Success: Created ${RAW_ONNX}"
else
    echo "Step 1 Failed"
    exit 1
fi

# Step 2: Patch ONNX
echo "[Step 2] Patching ONNX model..."
$ONNX_PYTHON hailo/patch_onnx.py "${RAW_ONNX}" "${PATCHED_ONNX}"
if [ $? -eq 0 ]; then
    echo "Step 2 Success: Created ${PATCHED_ONNX}"
else
    echo "Step 2 Failed"
    exit 1
fi

# Step 3: ONNX -> HEF
echo "[Step 3] Converting ONNX to HEF..."
# Pass model_name as BASE_NAME so internal layers are named consistently if possible
$HAILO_PYTHON hailo/onnx_to_hef.py "${PATCHED_ONNX}" "${HEF_OUTPUT}" --model_name "${BASE_NAME}"
if [ $? -eq 0 ]; then
    echo "Step 3 Success: Created ${HEF_OUTPUT}"
else
    echo "Step 3 Failed"
    exit 1
fi

echo "=========================================="
echo "Flow Complete!"
echo "Final Output: ${HEF_OUTPUT}"
echo "=========================================="
