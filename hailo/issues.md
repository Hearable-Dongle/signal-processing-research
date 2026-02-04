# Issues and Suggestions

## 1. Unsupported Dimensions (262144)
**Symptom:** The Hailo compiler fails with `Unsupported dimensions: ... [-1, 1, 1999, 262144]`.
**Diagnosis:** This was caused by the `PReLU` layer. The `PReLU` operation, when exported to ONNX (likely with opset issues or 1D weights), caused the Hailo compiler to misinterpret the channel dimensions or layout, exploding the 512 channels to 262144 ($512^2$).
**Resolution:** Replaced standard `nn.PReLU` with a custom `PReLU4D` implementation using supported elementwise operations (`Max`, `Min`, `Mul`, `Add`) with explicit 4D broadcasting (`[1, C, 1, 1]`). This ensures the graph remains fully compatible with Hailo's 4D expectation.

## 2. GlobLN Implementation
**Symptom:** Potential for ambiguity using `Conv2d` for elementwise scaling.
**Resolution:** Updated `GlobLN` monkey patch to use explicit `Mul` and `Add` for `gamma` and `beta` application, avoiding reliance on `Conv2d` groups which can be fragile during translation.

## 3. Calibration Data & Input Shapes
**Symptom:** `ValueError: Couldn't detect CalibrationDataType` and `BadInputsShape`.
**Diagnosis:** 
1. `runner.optimize` requires `calib_data` to be a `numpy` array (or dict of arrays), not a list of arrays, when passed in a dictionary.
2. The input shape expectation for `NHWC` layout requires removing the batch dimension from the data sample if the runner treats the first dimension of the provided data as the sample count.
**Resolution:** 
- Converted `calib_data` list to a single stacked `numpy` array.
- Transposed data from `NCHW` (`[1, 1, 1, 16000]`) to `NHWC` (`[1, 1, 16000, 1]`).
- Squeezed the batch dimension to match the network's expected input shape `(1, 16000, 1)`.
- Used explicit input name `convtas/input_layer1`.

## 4. Status
The pipeline now successfully reaches the Optimization and Calibration stage. The `Unsupported dimensions` error is resolved.
