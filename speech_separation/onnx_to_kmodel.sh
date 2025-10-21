#!/bin/bash

set -ex

readonly IMAGE_NAME="nncase-compiler"
readonly DOCKERFILE_PATH="./to_kmodel/Dockerfile"
readonly SIMPLIFIED_MODEL_NAME="simplified_model.onnx"
readonly CALIBRATION_DATA_DIR="dummy_data"


if [ "$#" -ne 2 ]; then
    echo "Error: Invalid number of arguments."
    echo "Usage: $0 <path_to_input_onnx> <path_to_output_kmodel>"
    exit 1
fi

readonly INPUT_MODEL_PATH="$1"
readonly OUTPUT_MODEL_PATH="$2"

if [ ! -d "${CALIBRATION_DATA_DIR}" ] || [ -z "$(ls -A ${CALIBRATION_DATA_DIR})" ]; then
    echo "Error: Calibration directory '${CALIBRATION_DATA_DIR}' is missing or empty."
    echo "Please create it and add a few sample input files for quantization."
    exit 1
fi


if [[ "$(docker images -q ${IMAGE_NAME} 2> /dev/null)" == "" ]]; then
  echo "Docker image '${IMAGE_NAME}' not found. Building it now..."
  docker build --platform linux/amd64 -t "${IMAGE_NAME}" -f "${DOCKER_FILE_PATH}" .
  echo "Image built successfully."
fi


echo "🚀 Starting compilation with quantization..."
echo "This step combines model simplification and compilation. If it fails, the error will be shown below."

echo "STEP 1: Simplifying the ONNX model..."
docker run --rm --platform linux/amd64 \
  -v "$(pwd)":/workspace \
  -w /workspace \
  "${IMAGE_NAME}" \
  onnxsim "${INPUT_MODEL_PATH}" "${SIMPLIFIED_MODEL_NAME}"

if [ $? -ne 0 ]; then
    echo "ERROR: ONNX simplification failed. Aborting."
    exit 1
fi
echo "ONNX simplification successful."
echo ""


echo "STEP 2: Compiling the simplified model to K-Model..."
docker run --rm --platform linux/amd64 \
  -v "$(pwd)":/workspace \
  -w /workspace \
  "${IMAGE_NAME}" \
  ncc compile "${SIMPLIFIED_MODEL_NAME}" "${OUTPUT_MODEL_PATH}" \
    -i onnx \
    -o kmodel \
    -t k210 \
    --input-type float32 \
    --quant-type uint8
    # --dataset "${CALIBRATION_DATA_DIR}" -v

DOCKER_EXIT_CODE=$?

if [ ${DOCKER_EXIT_CODE} -eq 0 ]; then
    echo ""
    echo "----------------------------------------"
    echo "Compilation successful!"
    echo "Quantized model saved to: ${OUTPUT_MODEL_PATH}"
    echo "----------------------------------------"
else
    echo ""
    echo "----------------------------------------"
    echo "Compilation failed inside the Docker container with exit code: ${DOCKER_EXIT_CODE}"
    echo "This is likely due to the model being too large for the K210's memory."
    echo "Check the verbose output above for specific memory allocation errors."
    echo "----------------------------------------"
    exit 1
fi

