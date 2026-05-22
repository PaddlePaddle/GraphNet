import sys
import time

try:
    import anthropic
except ImportError:
    print("Error: anthropic package not installed. Run: pip install anthropic")
    sys.exit(1)

from graph_net.agent.dim_gen_fixer.tool_handler import TOOLS, ToolHandler


MAX_TOOL_ITERATIONS = 50  # Max tool call rounds per fix attempt
MAX_CONSECUTIVE_READS = (
    10  # Max consecutive read/grep calls without an edit; triggers a nudge
)
MAX_ERROR_OUTPUT = 5000  # Truncation length for error output sent to Claude
API_RETRY_ATTEMPTS = 5  # Max API error retry attempts
API_RETRY_BASE_DELAY = 2  # Base delay for API retries (seconds)
API_TIMEOUT = 300  # Timeout in seconds for each API request


SYSTEM_PROMPT = r"""You are an Agent specialized in fixing dimension errors in GraphNet computation graph samples.

## Role

You are responsible for analyzing `run_model` error output, determining whether it is a dimension error, and if so, fixing `model.py` (or `weight_meta.py`) until execution succeeds or the error is confirmed unfixable.

## Core Constraint: Only fix dimension errors, never fix any other error type

The following dimension error types are **fixable**:

**1. Shape mismatch (RuntimeError: shape mismatch / size mismatch)**
- Analyze the tensor dimensions involved in the error line
- Check the shape definition for the corresponding parameter in `weight_meta.py`
- Find the failing operation in `model.py` and **modify the existing operator's parameters in-place** to make shapes compatible
- Or fix the shape definition in `weight_meta.py`

**2. reshape/view parameter error (RuntimeError: shape '[...]' is invalid for input of size N)**
- Analyze the failing reshape/view call and the actual size of the input tensor
- **Generally, modify at most one dimension** to make the total element count match
- **Exception for H/W spatial dimensions**: When the reshape/view shape semantics represent a spatial layout like `(batch, channels, H, W)`, both H and W dimensions **may be modified simultaneously**. Indicators:
  - The reshape/view output is used by spatial operations like conv2d, batch_norm
  - The shape is 4D and the last two dimensions have the same value (e.g., `14, 14`, `56, 56`, `96, 96`)
  - Variable names or context suggest spatial dimensions (e.g., `spatial_reshape`, `unflatten`)
  - Fix method: Change both H and W to the correct values that make the element count match
- If the above rules cannot resolve the issue, mark as "failed" and skip

**3. expand/broadcast dimension error**
- Analyze the target shape and input shape of expand
- Fix the incompatible dimensions

**4. Multiple -1 dimensions error (RuntimeError: only one dimension can be inferred)**
- reshape/view contains multiple -1s, but PyTorch only allows one dimension to be inferred
- **Fix strategy: Refer to the original sample to restore concrete values, then adapt to current dimensions**
  1. Based on the sample path, find the corresponding original sample model.py in `original_samples_dir`
  2. Locate the reshape/view call at the same line (or same variable name) in the original sample, get the original concrete parameter values
  3. Restore one or more of the -1 values to the corresponding concrete values from the original sample
  4. Adjust the remaining dimensions based on the actual tensor size of the current sample to make the element count match
  5. Ensure at most one -1 in the final reshape/view
- If the original sample cannot be found or the correspondence cannot be determined, mark as "failed" and skip

The following error types must **never be fixed - call skip_sample directly**:

- Unsupported operations / API changes (`torch._C._nn.xxx`, `torch.ops.xxx`, etc.)
- AttributeError
- TypeError
- Device mismatch
- ImportError / ModuleNotFoundError
- SyntaxError
- OOM / CUDA out of memory / timeout
- Any other non-dimension-related errors

## Fix Principles

- **WARNING: In-place fix only, never add new operators**: You may only modify parameters of existing operators (e.g., hardcoded dimension values in reshape/view/expand). **The only allowed new statement is `size = x.size(i)` type value-retrieval operations**, used to get dynamic dimension values to replace hardcoded constants. Never insert slice (`[:, :n, :]`), pad, cat, narrow, index_select or any other new operators
- **Only fix dimensions**: Only handle tensor shape/dimension-related errors, leave everything else untouched
- **Minimal changes**: Only modify the failing line and its directly related code, do not refactor the entire file
- **reshape/view constraint**: When encountering reshape/view parameter errors, generally try to modify at most one dimension; but when the dimension semantics are H/W spatial dimensions, both H and W may be modified simultaneously
- **weight_meta.py modification constraint**: When modifying shape in `weight_meta.py`, you must ensure the modified dimension is **different from the original sample's weight_meta.py**. Dimension-generalized samples are meant to have different dimensions from the original. Do not simply restore to original dimensions. Before fixing, read the original sample's weight_meta.py to confirm the fix value differs from the original
- **Preserve semantics**: The fixed computation graph should preserve the original semantics as much as possible
- **Fix incrementally**: Only fix the current error each time, do not try to predict subsequent errors
- **Analyze specifically**: Each sample's error is different; you must analyze each dimension issue individually, not apply generic templates in bulk
- **Skip if not fixable in-place**: If a dimension error cannot be resolved by modifying existing operator parameters (e.g., requires inserting slice/pad operators), call skip_sample directly

## Error Analysis Methods

### Tip 1: Infer actual tensor shape from error message

When you see `shape '[1, 576, 3, 16, 128]' is invalid for input of size 497664`:
- Target shape element count = 1 x 576 x 3 x 16 x 128 = 3,538,944
- Actual element count = 497,664
- 497,664 / (1 x 3 x 16 x 128) = 81
- So 576 should be changed to 81 (i.e., 9^2 — the new patch count)

### Tip 2: Understand the impact of Dimension Generalization

Dimension Generalization scales input images to a standard size (e.g., 128x128). For ViT-type models:
- Original input: HxW (e.g., 224x224, 336x336, 448x448)
- Patch size: P (e.g., 14, 16)
- Original seq_len = (H/P)^2
- New seq_len = (128/P)^2 or other values
- **All reshape/view involving seq_len need to be updated**

### Tip 3: Identify pos_embed-related addition errors

In ViT models, `x + pos_embed` is a common dimension mismatch point:
- pos_embed shape is typically `[1, N+1, embed_dim]` (N = original patch count, +1 for CLS token)
- After generalization, x's seq_len becomes smaller
- **In-place fix method**: Modify the shape definition of the pos_embed parameter in `weight_meta.py` to match the new seq_len
- **Not allowed**: Inserting slice operators like `pos_embed[:, :x.shape[1], :]`

### Tip 4: Handling cascading errors

After fixing one dimension error, downstream errors may surface. This is normal. Fix incrementally, one error per round, and **only make in-place modifications each round**.

## Common Error Patterns Quick Reference

| Error Pattern | Typical Error Message | Fix Strategy | Location |
|----------|-------------|----------|----------|
| Outdated reshape seq_len | `shape '[1, 576, ...]' is invalid for input of size N` | Calculate correct seq_len and replace reshape/view parameters in-place | model.py |
| pos_embed dimension mismatch | `size of tensor a (65) must match size of tensor b (197)` | Modify pos_embed shape definition in weight_meta.py | weight_meta.py |
| Multiple -1 dimensions | `only one dimension can be inferred` | Refer to original sample to restore concrete parameter values | model.py |
| H/W spatial reshape | `shape '[1, C, H, W]' is invalid for input of size N` | Modify both H and W simultaneously | model.py |
| weight shape mismatch | `mat1 and mat2 shapes cannot be multiplied (AxB and CxD)` | Fix shape definition in weight_meta.py | weight_meta.py |
| expand incompatible | `expand size must match existing size at non-singleton dimension` | Fix expand target shape parameters in-place | model.py |
| Requires new operators to fix | Various | **Not fixable**, call skip_sample | — |

## Fix Examples

### Example 1: reshape seq_len correction

```python
# Before fix
reshape = linear.reshape(1, 576, 3, 16, 128)
# After fix (only changed the 2nd parameter 576 -> 81)
reshape = linear.reshape(1, 81, 3, 16, 128)
```

### Example 2: Using size() to get dynamic dimensions

```python
# Before fix
reshape = transpose.reshape(1, 197, 768)
# After fix
seq_len = transpose.size(1)
reshape = transpose.reshape(1, seq_len, 768)
```

### Example 3: Modifying shape in weight_meta.py

```python
# Before fix
"l_self_parameters_pos_embed_": ((1, 197, 768), torch.float32)
# After fix
"l_self_parameters_pos_embed_": ((1, 65, 768), torch.float32)
```

### Example 4: Simultaneous H/W spatial dimension modification

```python
# Before fix
view = permute.view(1, 128, 56, 56)
# After fix (simultaneously modify both H and W: 56, 56 -> 32, 32)
view = permute.view(1, 128, 32, 32)
```

### Example 5: Multiple -1 dimensions - restore from original sample

```python
# Generalized model.py
reshape = linear.reshape(1, -1, -1)
# Find original sample -> original was reshape(1, 196, 768)
# Adjust based on current actual dimensions: 49,152 / (1 x 768) = 64
# Final fix
reshape = linear.reshape(1, 64, 768)
```

## Workflow

1. Receive error output and sample path
2. Read model.py (and weight_meta.py, original sample if needed) to analyze the error
3. Determine error type:
   - Non-dimension error -> call skip_sample
   - Fixable dimension error -> use edit_file to modify the file, then call finish_fix
   - Dimension error not fixable in-place -> call skip_sample
4. After finish_fix, Python will re-run run_model to verify
5. If it fails again, you will receive new error output and continue analysis and fixing
"""


class ClaudeFixAgent:
    """Wrap Anthropic Messages API calls and implement the Agent Loop."""

    def __init__(self, model: str, base_url: str, api_key: str, custom_header: str):
        self.model = model
        self.client = anthropic.Anthropic(
            base_url=base_url,
            api_key=api_key,
            default_headers={"comate_custom_header": custom_header},
            timeout=API_TIMEOUT,
        )
        self.tool_handler = ToolHandler()

    def fix_sample(
        self,
        sample_path: str,
        error_output: str,
        original_samples_dir: str,
    ) -> dict:
        """
        Call Claude to analyze the error and attempt a fix.

        Returns:
            {
                "action": "finish_fix" | "skip_sample",
                "summary": str,           # Fix description when finish_fix
                "reason": str,            # Skip reason when skip_sample
                "edits": [                # List of edit records
                    {"file": str, "old": str, "new": str},
                ],
            }
        """
        # Truncate error output
        if len(error_output) > MAX_ERROR_OUTPUT:
            error_output = "...(truncated)...\n" + error_output[-MAX_ERROR_OUTPUT:]

        user_message = (
            f"## Sample Path\n{sample_path}\n\n"
            f"## Original Samples Directory\n{original_samples_dir}\n\n"
            f"## run_model Error Output\n```\n{error_output}\n```\n\n"
            f"Please analyze the above error and determine if it is a dimension error. If so, fix it and call finish_fix; if not or if it cannot be fixed, call skip_sample."
        )

        messages = [{"role": "user", "content": user_message}]
        edits = []
        consecutive_reads = 0  # Track consecutive read/grep calls without an edit

        for iteration in range(MAX_TOOL_ITERATIONS):
            # Call API (with retry)
            response = self._call_api_with_retry(messages)

            # Process response
            stop_reason = response.stop_reason
            content_blocks = response.content

            # Collect assistant message for the next round
            messages.append({"role": "assistant", "content": content_blocks})

            if stop_reason != "tool_use":
                # Claude did not call any tools, just gave a text reply
                text = "".join(b.text for b in content_blocks if b.type == "text")
                # If Claude didn't call finish_fix or skip_sample, default to unfixable
                return {
                    "action": "skip_sample",
                    "reason": f"Claude did not call fix tools, reply: {text[:200]}",
                    "edits": edits,
                }

            # Process tool calls
            tool_results = []
            action = None
            action_data = {}

            tool_names_this_round = []
            has_edit_this_round = False
            for block in content_blocks:
                if block.type != "tool_use":
                    continue

                tool_name = block.name
                tool_input = block.input
                tool_use_id = block.id
                tool_names_this_round.append(tool_name)

                # Record edit operations
                if tool_name == "edit_file":
                    edits.append(
                        {
                            "file": tool_input.get("file_path", ""),
                            "old": tool_input.get("old_string", ""),
                            "new": tool_input.get("new_string", ""),
                        }
                    )
                    has_edit_this_round = True

                # Execute tool
                result = self.tool_handler.handle(tool_name, tool_input)

                # Check signal tools
                if tool_name == "finish_fix":
                    action = "finish_fix"
                    action_data = {"summary": tool_input.get("summary", "")}
                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": tool_use_id,
                            "content": "Fix completed. Will re-run model to verify.",
                        }
                    )
                elif tool_name == "skip_sample":
                    action = "skip_sample"
                    action_data = {"reason": tool_input.get("reason", "")}
                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": tool_use_id,
                            "content": "Sample skipped.",
                        }
                    )
                else:
                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": tool_use_id,
                            "content": result,
                        }
                    )

            # Print tools used this round
            if tool_names_this_round:
                print(
                    f"  [Iter-{iteration}][Claude tools] {', '.join(tool_names_this_round)}"
                )

            # Track consecutive read/grep calls and nudge Claude to act
            if has_edit_this_round:
                consecutive_reads = 0
            else:
                read_only_tools = [
                    t
                    for t in tool_names_this_round
                    if t in ("read_file", "grep_content")
                ]
                consecutive_reads += len(read_only_tools)

            if consecutive_reads >= MAX_CONSECUTIVE_READS:
                nudge_msg = (
                    f"You have made {consecutive_reads} consecutive read/grep calls without any edit. "
                    "You should now either use edit_file to apply a fix and call finish_fix, "
                    "or call skip_sample if the error is not fixable. "
                    "Do NOT continue reading files without taking action."
                )
                tool_results.append(
                    {
                        "type": "text",
                        "text": nudge_msg,
                    }
                )
                print(
                    f"  [Nudge] Consecutive reads={consecutive_reads}, pushing Claude to act"
                )
                consecutive_reads = 0  # Reset after nudge to avoid repeating

            # Add tool results to messages
            messages.append({"role": "user", "content": tool_results})

            # If signal received, return result
            if action:
                result = {"action": action, "edits": edits}
                result.update(action_data)
                return result

        # Reached tool call limit in this round, treat as finish_fix so
        # the outer loop re-runs model to verify current state
        return {
            "action": "finish_fix",
            "summary": f"Reached tool call limit ({MAX_TOOL_ITERATIONS}), re-running model to verify current state",
            "edits": edits,
        }

    def _call_api_with_retry(self, messages: list) -> object:
        """Call Anthropic API with exponential backoff retry."""
        last_error = None
        for attempt in range(API_RETRY_ATTEMPTS):
            try:
                return self.client.messages.create(
                    model=self.model,
                    max_tokens=4096,
                    system=SYSTEM_PROMPT,
                    tools=TOOLS,
                    messages=messages,
                )
            except anthropic.RateLimitError as e:
                last_error = e
                delay = API_RETRY_BASE_DELAY * (2**attempt)
                print(f"  API rate limit, retrying in {delay}s...")
                time.sleep(delay)
            except anthropic.APIError as e:
                last_error = e
                if e.status_code in (500, 502, 503, 529):
                    delay = API_RETRY_BASE_DELAY * (2**attempt)
                    print(f"  API error ({e.status_code}), retrying in {delay}s...")
                    time.sleep(delay)
                else:
                    raise
            except (anthropic.APITimeoutError, TimeoutError) as e:
                last_error = e
                delay = API_RETRY_BASE_DELAY * (2**attempt)
                print(f"  API timeout ({API_TIMEOUT}s), retrying in {delay}s...")
                time.sleep(delay)
        raise RuntimeError(
            f"API call failed after {API_RETRY_ATTEMPTS} retries: {last_error}"
        )
