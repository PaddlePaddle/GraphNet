# Init Input Tensor Constraints

## Goal

Initialize `input_tensor_constraints.py` for graph_net samples by running dimension generalization passes. This identifies which input tensor dimensions are symbolic (dynamic) vs. concrete (static), enabling downstream symbolic dimension reification.

## Architecture Overview

- **Constraint initializer**: `graph_net/constraint_util.py` — `UpdateInputTensorConstraints` handler
- **Data input predicator**: `graph_net/torch/constraint_util.py` — `NaiveDataInputPredicator`
- **Model runnable predicators**:
  - `ModelRunnablePredicator` — actually runs the model (slower, more accurate)
  - `ShapePropagatablePredicator` — shape propagation only (faster, less accurate)
- **Dimension generalizer**: `graph_net/torch/static_to_dynamic.py` — `StaticToDynamic` with configurable passes
- **Model path handler**: `graph_net.model_path_handler` — iterates model paths and dispatches to handler

## Script Usage

All scripts take `<model_path_prefix>` as the first argument (the directory containing sample model paths).

### Step 1: Initialize constraints (single GPU)

```bash
# Full accuracy (ModelRunnablePredicator, default):
bash init_input_tensor_constraints.sh <model_path_prefix> <cuda_device> [suffix]

# Faster (ShapePropagatablePredicator):
bash init_input_tensor_constraints.sh --fast <model_path_prefix> <cuda_device> [suffix]
```

- `<cuda_device>`: GPU id (e.g. `0`, `1`)
- `[suffix]`: optional suffix appended to model path list filename for sharding (e.g. `01`)

### Step 2: Multi-GPU batch initialization

```bash
bash batch_init.sh <model_path_prefix> [num_gpus] [log_dir]
```

- `[num_gpus]`: number of GPUs to use (default: 8)
- `[log_dir]`: log directory (default: `/tmp/init-constraints-logs`)

### Step 3: Monitor and restart tasks

```bash
bash monitor_tasks.sh <model_path_prefix> [monitor|status|restart] [num_gpus] [log_dir]
```

- `monitor` (default): continuously monitor and auto-restart stopped tasks
- `status`: show current task status
- `restart`: one-shot restart all stopped tasks

Note: GPU 0 OOM is expected behavior and will be ignored during restart.

### Step 4: Check progress

```bash
# Count samples with initialized constraints (any, including empty []):
bash count_handled.sh <model_path_prefix>

# Count samples with generalized (symbolic) dimensions ([S0, ...]):
bash count_generalized.sh <model_path_prefix>
```

## Workflow

1. **Shard the model list** into per-GPU splits (append suffix `01`, `02`, ... to the list file)
2. **Run `batch_init.sh`** to launch all GPUs in parallel
3. **Monitor with `monitor_tasks.sh`** to auto-restart failed tasks
4. **Check progress** with `count_handled.sh` and `count_generalized.sh`
5. **Proceed to reification** — use `../reify-dims/` scripts to map symbolic shapes to concrete dimensions
