## Hardware Regression Testing
### Step 1: Generate Reference Data
First, use `graph_net.paddle.test_reference_device` on a trusted setting (e.g., a specific hardware/compiler version) to generate baseline logs and output files.
```bash
python -m graph_net.paddle.test_reference_device \
    --model-path /path/to/all_models/ \
    --reference-dir ./gold_reference \
    --compiler cinn \
    --device cuda
# --reference-dir: (Required) Directory where the output .log (performance/config) and .pdout (output tensors) files will be saved.
# --compiler: Specifies the compiler backend.
```
### Step 2: Run Regression Test
After changing hardware, run the correctness test script. This script reads the reference data, re-runs the models using the same configuration, and compares the new results against the "golden" reference.
```bash
python -m graph_net.paddle.test_device_correctness \
    --reference-dir ./golden_reference \
    --device cuda
```
This script will report any failures (e.g., compilation errors, output mismatches) and print a performance comparison (speedup/slowdown) against the reference log, allowing you to identify regressions quickly.
