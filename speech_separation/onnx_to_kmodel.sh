#!/bin/bash

# set -e: Exit immediately if a command exits with a non-zero status.
# set -x: Print commands and their arguments as they are executed for easier debugging.
set -ex

# --- Configuration ---
readonly IMAGE_NAME="nncase-compiler"
readonly DOCKERFILE_PATH="./to_kmodel/Dockerfile"
readonly SIMPLIFIED_MODEL_NAME="simplified_model.onnx"
# IMPORTANT: Create this directory and place a few sample audio clips (.npy or .bin) in it.
# These samples help the compiler quantize the model with better accuracy.
readonly CALIBRATION_DATA_DIR="dummy_data"


# --- Argument Validation ---
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


# --- Docker Image Build ---
if [[ "$(docker images -q ${IMAGE_NAME} 2> /dev/null)" == "" ]]; then
  echo "Docker image '${IMAGE_NAME}' not found. Building it now..."
  docker build --platform linux/amd64 -t "${IMAGE_NAME}" -f "${DOCKER_FILE_PATH}" .
  echo "Image built successfully."
fi


# --- Model Compilation ---
echo "🚀 Starting compilation with quantization..."
echo "This step combines model simplification and compilation. If it fails, the error will be shown below."


# --- Model Compilation (in separate steps for debugging) ---

echo "STEP 1: Simplifying the ONNX model..."
docker run --rm --platform linux/amd64 \
  -v "$(pwd)":/workspace \
  -w /workspace \
  "${IMAGE_NAME}" \
  onnxsim "${INPUT_MODEL_PATH}" "${SIMPLIFIED_MODEL_NAME}"

# Check if the simplification step succeeded
if [ $? -ne 0 ]; then
    echo "❌ ERROR: ONNX simplification failed. Aborting."
    exit 1
fi
echo "✅ ONNX simplification successful."
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

# docker run --rm --platform linux/amd64 \
#   -v "$(pwd)":/workspace \
#   -w /workspace \
#   "${IMAGE_NAME}" \
#   /bin/bash -c "onnxsim ${INPUT_MODEL_PATH} ${SIMPLIFIED_MODEL_NAME} && \
#                 ncc compile ${SIMPLIFIED_MODEL_NAME} ${OUTPUT_MODEL_PATH} \
#                   -i onnx \
#                   -o kmodel \
#                   -t k210 \
#                   --input-type float32 \
#                   --quant-type uint8 \
#                   --dataset ${CALIBRATION_DATA_DIR} -v"

# Store the exit code of the docker command immediately after it runs
DOCKER_EXIT_CODE=$?

# Explicitly check the exit code to provide a clearer success or failure message
if [ ${DOCKER_EXIT_CODE} -eq 0 ]; then
    echo ""
    echo "----------------------------------------"
    echo "✅ Compilation successful!"
    echo "Quantized model saved to: ${OUTPUT_MODEL_PATH}"
    echo "----------------------------------------"
else
    echo ""
    echo "----------------------------------------"
    echo "❌ Compilation failed inside the Docker container with exit code: ${DOCKER_EXIT_CODE}"
    echo "This is likely due to the model being too large for the K210's memory."
    echo "Check the verbose output above for specific memory allocation errors."
    echo "----------------------------------------"
    exit 1
fi

# 
# #!/bin/bash
# 
# # set -ex
# 
# 
# readonly IMAGE_NAME="nncase-compiler"
# readonly DOCKERFILE_PATH="./to_kmodel/Dockerfile"
# readonly SIMPLIFIED_MODEL_NAME="simplified_model.onnx"
# 
# if [ "$#" -ne 2 ]; then
#     echo "Error: Invalid number of arguments."
#     echo "Usage: $0 <path_to_input_onnx> <path_to_output_kmodel>"
#     exit 1
# fi
# 
# readonly INPUT_MODEL_PATH="$1"
# readonly OUTPUT_MODEL_PATH="$2"
# 
# if [[ "$(docker images -q ${IMAGE_NAME} 2> /dev/null)" == "" ]]; then
#   echo "Docker image '${IMAGE_NAME}' not found. Building it now..."
#   docker build --platform linux/amd64 -t "${IMAGE_NAME}" -f "${DOCKERFILE_PATH}" .
#   echo "Image built successfully."
# fi
# 
# echo "🚀 Starting compilation..."
# docker run --rm --platform linux/amd64 \
#   -v "$(pwd)":/workspace \
#   -w /workspace \
#   "${IMAGE_NAME}" \
#   /bin/bash -c "onnxsim ${INPUT_MODEL_PATH} ${SIMPLIFIED_MODEL_NAME} && \
#                 ncc compile ${SIMPLIFIED_MODEL_NAME} ${OUTPUT_MODEL_PATH} \
#                   -i onnx \
#                   -o kmodel \
#                   -t k210 \
#                   --input-type float32 \
#                   --quant-type uint8 \
#                   --dataset ${CALIBRATION_DATA_DIR}"
# 
# # docker run --rm --platform linux/amd64 \
# #   -v "$(pwd)":/workspace \
# #   -w /workspace \
# #   "${IMAGE_NAME}" \
# #   /bin/bash -c "onnxsim ${INPUT_MODEL_PATH} ${SIMPLIFIED_MODEL_NAME} && \
# #                 ncc compile ${SIMPLIFIED_MODEL_NAME} ${OUTPUT_MODEL_PATH} -i onnx -o kmodel -t k210 -v"
# 
# if [ $? -eq 0 ]; then
#     echo ""
#     echo "Compilation successful!"
#     echo "Model saved to: ${OUTPUT_MODEL_PATH}"
# else
#     echo ""
#     echo "Compilation failed inside the Docker container."
#     exit 1
# fi
