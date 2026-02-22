# Hailo 8 Conversion

See [hailo_demo](../hailo_demo/README.md) to setup a dummy project play with the Hailo SDK and Onnx models.

## Overview

Pytorch does not run on the Hailo 8, and we need to convert the model to Hailo's .hef format that runs on the hardware.

Full flow is:
Pytorch Model --> Onnx Model --> Hailo .har Model --> Hailo .hef Binary

## Environment

Two environments are needed:
### 1. Pytorch --> Onnx

```shell
python3.10 -m venv hailo/to-onnx-env
source hailo/to-onnx-env/bin/activate
pip install -r hailo/to-onnx-conv-env-reqrequirements.txt
```

### 2. Onnx --> Hailo .har

Create an account in [Hailo Developer Zone](https://hailo.ai/) and download the [Hailo Dataflow Compiler SDK](https://hailo.ai/developer-zone/software-downloads/?product=ai_accelerators&device=hailo_8_8l). I am pretty sure that this only works on a Linux machine - ask Matthew for access to remote machine if you are having issues with another OS.

```shell
python3.10 -m venv hailo/to-hailo-env # IMPORTANT: Hailo Data Conversion Flow only supports python3.10
source hailo/to-hailo-env/bin/activate
pip install <downloaded-hailo-sdk-client-whl-file>.whl
pip install -r hailo/to-hailo-env-requirements.txt
```


## Conversion flow
After setting up the environments, run the full conversion flow with:
```shell
chmod +x hailo/run_conversion_flow.sh
hailo/run_conversion_flow.sh convtas
```

To run just the HAR to HEF step (optimize + compile) independently:
```shell
hailo/to-hailo-env/bin/python -m hailo.har_to_hef hailo/convtas.har hailo/convtas.hef
```

Preferred (real calibration + explicit target):
```shell
hailo/to-hailo-env/bin/python -m hailo.har_to_hef \
  hailo/convtas.har hailo/convtas.hef \
  --hw_arch hailo8 \
  --calib_npz <path/to/calibration.npz> \
  --log_failed_layers_path hailo/compile_failure.txt
```

Full flow with environment overrides:
```shell
HW_ARCH=hailo8 \
NORM_MODE=channel \
CALIB_NPZ=<path/to/calibration.npz> \
COMPILER_OPT_LEVEL=max \
hailo/run_conversion_flow.sh convtas
```

Topology isolation overrides (for compile debugging):
```shell
DISABLE_SKIP=true|false \
MASK_MUL_MODE=normal|bypass \
FORCE_N_SRC_1=true|false \
BYPASS_CONCAT=true|false \
SKIP_TOPOLOGY_MODE=concat|project \
DECONV_MODE=grouped|ungrouped_blockdiag|reduced_deconv_128|reduced_deconv_64|conv1x1_head \
TRUNCATE_K_BLOCKS=0|1|2|... \
hailo/run_conversion_flow.sh convtas_dbg convtas_dbg.har
```

Focused targeted run sequence (baseline + skip/mask/source/K-sweep):
```shell
CALIB_NPZ=hailo/calibration_1000ms_16k_64.npz \
./hailo/realtime_sep_compile_isolation_run.sh
```

Artifacts:
- Logs: `hailo/night_runs2/<timestamp>/<run_tag>.log`
- Failure reports: `hailo/night_runs2/<timestamp>/<run_tag>.failure.txt(.json)`
- Summary table: `hailo/night_runs2/summary.tsv`

Generate calibration NPZ from LibriMix (example: 200 ms clips at 16 kHz):
```shell
hailo/to-hailo-env/bin/python -m hailo.generate_calibration_npz \
  --output hailo/calibration_200ms_16k_64.npz \
  --num_samples 64 \
  --clip_ms 200 \
  --sample_rate 16000 \
  --layout nt
```

## TODO:
- [ ] Need to get latency and memory usage number on the actual Hailo
