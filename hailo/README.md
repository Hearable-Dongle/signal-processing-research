# Hailo 8 Conversion


See [hailo_demo](../hailo_demo/README.md) to setup a dummy project play with the Hailo SDK and Onnx models.

## Overview

Pytorch does not run on the Hailo 8, and we need to convert the model to Hailo's proprietary .har format.

Full flow is:
Pytorch Model --> Onnx Model --> Hailo .har Model

## Environment

Two environments are needed:
### 1. Pytorch --> Onnx

```shell
python3.10 -m venv hailo/to-onnx-env
source hailo/to-onnx-env/bin/activate
pip install -r hailo/onnx-conv-env-req.txt
```

### 2. Onnx --> Hailo .har

Create an account in [Hailo Developer Zone](https://hailo.ai/) and download the [Hailo Dataflow Compiler SDK](https://hailo.ai/developer-zone/software-downloads/?product=ai_accelerators&device=hailo_8_8l)

```shell
python3.10 -m venv hailo/to-hailo-env # IMPORTANT: Hailo Data Conversion Flow only supports python3.10
source hailo/to-hailo-env/bin/activate
pip install <downloaded-hailo-sdk-client-whl-file>.whl
pip install -r hailo/hailo-env-req.txt
```




