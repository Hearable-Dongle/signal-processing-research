#!/bin/bash

# This script runs the validation process for a predefined list of non-AI denoising models.
# It captures the output of the validation script, extracts key performance metrics 
# (Overall SII and Average RTF), and presents them in a summary table.
# NOTE: (this was vibe-coded) The end output doesn't really work but just check the logs

MODELS=("spectral-gating" "spectral-subtraction" "wiener" "wavelet" "high-pass" "notch")

LOG_DIR="validation_logs"
mkdir -p "$LOG_DIR"

# Associative arrays to hold the results for each model
declare -A RESULTS_SII
declare -A RESULTS_RTF

echo "Starting validation for non-AI models..."
echo "Logs will be saved in the '$LOG_DIR' directory."
echo ""

for model in "${MODELS[@]}"; do
    echo "----------------------------------------"
    echo "=> Running validation for: $model"
    echo "----------------------------------------"
    
    log_file="$LOG_DIR/${model}_validation.log"
    
    python -m denoise.validate_denoising \
        --samples 500 \
        --model-type "$model" \
        --save-outputs \
        --noise-type wham > "$log_file" 2>&1 # Check this log file for details

    if [ $? -eq 0 ]; then
        echo "Validation for '$model' completed successfully."
        
        sii=$(grep "Overall SII:" "$log_file" | awk '{print $NF}')
        rtf=$(grep "Average RTF:" "$log_file" | awk '{print $NF}')
        
        RESULTS_SII["$model"]=${sii:-"Not Found"}
        RESULTS_RTF["$model"]=${rtf:-"Not Found"}
    else
        echo "Validation for '$model' failed. Check logs for details: $log_file"
        RESULTS_SII["$model"]="Failed"
        RESULTS_RTF["$model"]="Failed"
    fi
    echo ""
done

# TODO: get this part working
echo "========================================================"
echo "               Validation Summary"
echo "========================================================"
printf "% -25s | % -15s | % -15s\n" "Model" "Overall SII" "Average RTF"
printf "% -25s | % -15s | % -15s\n" "-------------------------" "---------------" "---------------"

for model in "${MODELS[@]}"; do
    printf "% -25s | % -15s | % -15s\n" "$model" "${RESULTS_SII[$model]}" "${RESULTS_RTF[$model]}"
done
echo "========================================================"
echo "Detailed logs can be found in the '$LOG_DIR' directory."
