# Sym Dim Reifier Rule Generation

## Goal

Generate reifier rules that provide practical and typical concrete dimensions for symbolic shapes in graph_net samples. Each sample's `input_tensor_constraints.py` may contain symbolic dimensions (e.g. `S0`, `S1`). A reifier maps a symbolic shape pattern (e.g. `[(S0,S1)]`) to a list of concrete dimension combinations for benchmarking.

## Architecture Overview

- **Reifier base class**: `graph_net/torch/sym_dim_reifiers/reifier_base.py` — defines `match()` and `reify()` interface.
- **CV reifier**: `graph_net/torch/sym_dim_reifiers/naive_cv_sym_dim_reifier.py` — handles CV-related patterns (image batch/HW dims).
- **NLP reifier**: `graph_net/torch/sym_dim_reifiers/naive_nlp_sym_dim_reifier.py` — handles NLP-related patterns (batch/seq dims).
- **Reifier manager**: `graph_net/torch/sym_dim_reifiers/reifier_mgr.py` — loads reifier by name, expects `ConcreteReifier` class.
- **Reifier factory**: `graph_net/torch/reifier_factory.py` — iterates reifier names (CV first, then NLP), returns first match.
- **Update handler**: `graph_net/tools/_update_sym_dim_reifier.py` — writes `"symbolic_dimension_reifier"` into `graph_net.json`.

## Key Files You May Modify

Only modify reifier rule python files under:
```
graph_net/torch/sym_dim_reifiers/
```

Specifically:
- `naive_cv_sym_dim_reifier.py` — add new CV shape patterns and their reify methods
- `naive_nlp_sym_dim_reifier.py` — add new NLP shape patterns and their reify methods

## Script Usage

All scripts take `<model_path_prefix>` as the first argument (the directory containing sample model paths). An optional second argument overrides the model path list file.

### Step 1: Get the TODO list (what symbolic shapes exist)

```bash
bash statisticize_in_tensor_symbolic_shapes.sh <model_path_prefix>
```

This outputs frequency-sorted unique symbolic shape patterns, e.g.:
```
      3 [(S0,3,640,640)]
     12 [(S0,S1)]
     45 [(S0,S1),(S0,S1),(S0,S1)]
```

- Ignore `<reified>` lines — these are already handled.
- Ignore `[]` lines — these have no symbolic dims.

### Step 2: Inspect unhandled samples for a pattern

```bash
bash remainder_in_tensor_symbolic_shapes.sh <model_path_prefix> | grep '<pattern>'
```

Replace `<pattern>` with the symbolic shape string you're working on (e.g. `"(S0,3,640,640)"`). This shows all unhandled samples matching that pattern, with their model paths.

### Step 3: Add a reifier rule

Edit the appropriate reifier file. Follow the existing pattern:

1. **Add the shape string key** to the `_get_map_cv_sym_shapes_str2reifier()` or `_get_map_nlp_sym_shapes_str2reifier()` dict.
2. **Add a reify method** that returns `{(sympy_symbol_tuple,): [[val1, val2, ...], ...]}`.

#### How to write a reify method

```python
def reify_example(self):
    S0S1 = (sympy.Symbol("S0"), sympy.Symbol("S1"))
    return {
        S0S1: [
            [dim0_val, dim1_val],  # combination 1
            [dim0_val, dim1_val],  # combination 2
            ...
        ],
    }
```

Key rules:
- The dict key is a **tuple of sympy Symbols** in order (S0, S1, S2, ...).
- Each list entry is one concrete dimension combination.
- **Dimension combinations must be sorted by total product** (ascending).
- Choose values crossing multiple granularities of 2^4 (16): e.g. 1, 16, 256, 4096.
- Provide ~9 combinations covering small to large scale.
- Reuse existing reify methods when the same dimension semantics apply (point existing map entries to shared methods).

#### Shape string format

The shape string is produced by `DynamicDimConstraints.serialize_symbolic_input_shapes_to_str()`:
- Only input tensors with at least one symbolic dim are included.
- Format: `[(dim0,dim1,...),(dim0,dim1,...)]` — no spaces, sorted by string repr.
- Concrete dims appear as integers, symbolic dims as `S0`, `S1`, etc.

### Step 4: Apply the rules

```bash
bash update_sym_dim_reifier.sh <model_path_prefix>
```

This iterates all samples and writes `"symbolic_dimension_reifier": "<reifier_name>"` into each matching sample's `graph_net.json`.

### Step 5: Verify

Re-run step 1 to confirm the count of `<reified>` cases increased and the target pattern count decreased.

## Pattern Matching Priority

`ReifierFactory.get_reifier_names()` returns `["naive_cv_sym_dim_reifier", "naive_nlp_sym_dim_reifier"]`. CV is checked first. If a shape pattern could be either CV or NLP, place it in the more specific reifier. Generic patterns like `[(S0,S1)]` may need domain-aware handling (see `reify_nlp_or_gnn_batch_s0_seq_s1` in NLP reifier).

## Example: Adding a new pattern

Say `statisticize_in_tensor_symbolic_shapes.sh` shows `5 [(S0,3,320,320)]` not yet reified.

1. `bash remainder_in_tensor_symbolic_shapes.sh <model_path_prefix> | grep "(S0,3,320,320)"` to see which models.
2. This is a CV pattern (batch, 3-channel, 320x320). Edit `naive_cv_sym_dim_reifier.py`:
   ```python
   # In _get_map_cv_sym_shapes_str2reifier:
   "[(S0,3,320,320)]": cls.reify_yolo_s0,  # reuse existing batch-only reifier
   ```
3. `bash update_sym_dim_reifier.sh <model_path_prefix>`
4. Verify with step 1.
