"""
fix_dim_gen_error_api.py - Automatically fix dimension errors in dimension-generalized
samples using the Anthropic API (Claude).

Implements intelligent error analysis and repair through Anthropic Messages API + Tool Use.

Usage:
    export ANTHROPIC_API_KEY=sk-ant-...
    python graph_net/agent/dim_gen_fixer/fix_dim_gen_error_api.py --samples-dir /path/to/samples --max-samples 1 --max-retries 2
"""

import argparse
import os
import re
import subprocess
import sys
import time
from datetime import datetime

try:
    import anthropic
except ImportError:
    print("Error: anthropic package not installed. Run: pip install anthropic")
    sys.exit(1)

# Force unbuffered stdout for background process logging
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

# ============================================================
# Constants
# ============================================================

MAX_TOOL_ITERATIONS = 50  # Max tool call rounds per fix attempt
MAX_CONSECUTIVE_READS = (
    10  # Max consecutive read/grep calls without an edit; triggers a nudge
)
MAX_FILE_SIZE = 500 * 1024  # read_file rejects files larger than 500KB
MAX_ERROR_OUTPUT = 5000  # Truncation length for error output sent to Claude
API_RETRY_ATTEMPTS = 5  # Max API error retry attempts
API_RETRY_BASE_DELAY = 2  # Base delay for API retries (seconds)
API_TIMEOUT = 300  # Timeout in seconds for each API request

# ============================================================
# System Prompt (extracted from fix_dim_gen_error.md - Claude role definition and fix rules)
# ============================================================

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


# ============================================================
# Tool Definitions for Claude API
# ============================================================

TOOLS = [
    {
        "name": "read_file",
        "description": "Read file contents. Supports offset and limit parameters for pagination. Rejects files larger than 500KB.",
        "input_schema": {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Absolute path of the file to read",
                },
                "offset": {
                    "type": "integer",
                    "description": "Line number to start reading from (1-based), default is 1",
                    "default": 1,
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of lines to read, default reads all",
                    "default": 0,
                },
            },
            "required": ["file_path"],
        },
    },
    {
        "name": "edit_file",
        "description": (
            "Edit a file: exactly match old_string and replace with new_string. "
            "old_string must match uniquely in the file. "
            "Used to modify dimension parameters in model.py or weight_meta.py."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Absolute path of the file to edit",
                },
                "old_string": {
                    "type": "string",
                    "description": "The original string to be replaced (must match exactly)",
                },
                "new_string": {
                    "type": "string",
                    "description": "The replacement string",
                },
            },
            "required": ["file_path", "old_string", "new_string"],
        },
    },
    {
        "name": "grep_content",
        "description": (
            "Search for regex pattern matches in the specified file. "
            "Used to find specific patterns (e.g., reshape, view, expand calls)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Absolute path of the file to search",
                },
                "pattern": {
                    "type": "string",
                    "description": "Regular expression pattern",
                },
            },
            "required": ["file_path", "pattern"],
        },
    },
    {
        "name": "finish_fix",
        "description": "Signal: the current fix is complete, run_model can be re-executed to verify the fix result.",
        "input_schema": {
            "type": "object",
            "properties": {
                "summary": {
                    "type": "string",
                    "description": "Brief description of this fix (what was changed and why)",
                },
            },
            "required": ["summary"],
        },
    },
    {
        "name": "skip_sample",
        "description": "Signal: skip the current sample (non-dimension error or cannot be fixed by in-place modification).",
        "input_schema": {
            "type": "object",
            "properties": {
                "reason": {
                    "type": "string",
                    "description": "Skip reason (e.g., 'Non-dimension error: AttributeError' or 'Requires inserting new operators to fix')",
                },
            },
            "required": ["reason"],
        },
    },
]


# ============================================================
# ToolHandler - Handle Claude's tool_call, operate on local file system
# ============================================================


class ToolHandler:
    """Handle Claude's tool_call results, execute operations on the local file system."""

    def handle(self, tool_name: str, tool_input: dict) -> str:
        handler = {
            "read_file": self._read_file,
            "edit_file": self._edit_file,
            "grep_content": self._grep_content,
            "finish_fix": self._finish_fix,
            "skip_sample": self._skip_sample,
        }.get(tool_name)
        if handler is None:
            return f"Error: Unknown tool '{tool_name}'"
        return handler(tool_input)

    @staticmethod
    def _read_file(inp: dict) -> str:
        file_path = inp["file_path"]
        offset = inp.get("offset", 1)
        limit = inp.get("limit", 0)

        if not os.path.isfile(file_path):
            return f"Error: File not found: {file_path}"

        file_size = os.path.getsize(file_path)
        if file_size > MAX_FILE_SIZE:
            return f"Error: File too large ({file_size} bytes > {MAX_FILE_SIZE} bytes). Use offset/limit to read portions."

        try:
            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                lines = f.readlines()

            start = max(0, offset - 1)
            end = len(lines) if limit <= 0 else min(start + limit, len(lines))
            selected = lines[start:end]

            result_lines = []
            for i, line in enumerate(selected, start=start + 1):
                result_lines.append(f"{i:>6}\t{line.rstrip()}")
            return "\n".join(result_lines) if result_lines else "(empty file)"
        except Exception as e:
            return f"Error reading file: {e}"

    @staticmethod
    def _edit_file(inp: dict) -> str:
        file_path = inp["file_path"]
        old_string = inp["old_string"]
        new_string = inp["new_string"]

        if not os.path.isfile(file_path):
            return f"Error: File not found: {file_path}"

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
        except Exception as e:
            return f"Error reading file: {e}"

        count = content.count(old_string)
        if count == 0:
            return "Error: old_string not found in file. Make sure the string matches exactly."
        if count > 1:
            return f"Error: old_string matches {count} locations in file. Provide a longer/more specific string for unique match."

        new_content = content.replace(old_string, new_string, 1)
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(new_content)
            return "OK: File edited successfully."
        except Exception as e:
            return f"Error writing file: {e}"

    @staticmethod
    def _grep_content(inp: dict) -> str:
        file_path = inp["file_path"]
        pattern = inp["pattern"]

        if not os.path.isfile(file_path):
            return f"Error: File not found: {file_path}"

        file_size = os.path.getsize(file_path)
        if file_size > MAX_FILE_SIZE:
            return f"Error: File too large ({file_size} bytes)."

        try:
            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                lines = f.readlines()
        except Exception as e:
            return f"Error reading file: {e}"

        try:
            regex = re.compile(pattern)
        except re.error as e:
            return f"Error: Invalid regex pattern: {e}"

        matches = []
        for i, line in enumerate(lines, start=1):
            if regex.search(line):
                matches.append(f"{i:>6}\t{line.rstrip()}")

        if not matches:
            return "(no matches)"
        return "\n".join(matches)

    @staticmethod
    def _finish_fix(inp: dict) -> str:
        # Handled by ClaudeFixAgent's agent loop
        return "__FINISH_FIX__"

    @staticmethod
    def _skip_sample(inp: dict) -> str:
        # Handled by ClaudeFixAgent's agent loop
        return "__SKIP_SAMPLE__"


# ============================================================
# ClaudeFixAgent - Wrap Anthropic API calls and Agent Loop
# ============================================================


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


# ============================================================
# ErrorPreprocessor - Rule-based error preprocessing
# ============================================================


class ErrorPreprocessor:
    """Rule-based error preprocessing: auto-skip non-dim errors, auto-fix simple dim errors.

    Tries to handle common error patterns without calling Claude API.
    Only falls through to the API for errors it cannot handle.
    """

    NON_DIM_ERROR_PATTERNS = [
        r"AttributeError",
        r"TypeError",
        r"ImportError",
        r"ModuleNotFoundError",
        r"SyntaxError",
        r"Expected all tensors to be on the same device",
        r"torch\._C\._nn\.\w+",
        r"torch\.ops\.\w+",
        r"is not implemented",
        r"not registered",
    ]

    def __init__(self, original_samples_dir: str):
        self.original_samples_dir = original_samples_dir

    def try_preprocess(self, sample_path: str, error_output: str) -> dict:
        """Try to handle the error without calling Claude API.

        Returns:
            {"action": "skip"|"fix"|"unhandled", "reason": str, "edits": list, "summary": str}
        """
        # Rule 0: Auto-skip non-dim errors
        skip_result = self._try_skip_non_dim(error_output)
        if skip_result:
            return skip_result

        # Rule 1: Auto-fix reshape/view parameter errors
        fix_result = self._try_fix_reshape(sample_path, error_output)
        if fix_result:
            return fix_result

        # Rule 2: Auto-fix pos_embed shape mismatch
        fix_result = self._try_fix_pos_embed(sample_path, error_output)
        if fix_result:
            return fix_result

        # Rule 3: Auto-fix expand dimension mismatch
        fix_result = self._try_fix_expand(sample_path, error_output)
        if fix_result:
            return fix_result

        # Rule 4: Auto-fix multiple -1 dimensions
        fix_result = self._try_fix_multi_minus_one(sample_path, error_output)
        if fix_result:
            return fix_result

        return {"action": "unhandled", "reason": "", "edits": [], "summary": ""}

    # ----------------------------------------------------------------
    # Rule 0: Auto-skip non-dim errors
    # ----------------------------------------------------------------

    def _try_skip_non_dim(self, error_output: str) -> dict | None:
        """Check if the error is a non-dimension error and should be skipped.

        Only checks the LAST error line to avoid false positives from traceback noise.
        """
        last_error = self._extract_last_error(error_output)
        if not last_error:
            return None

        for pattern in self.NON_DIM_ERROR_PATTERNS:
            if re.search(pattern, last_error):
                return {
                    "action": "skip",
                    "reason": f"Non-dim error: {last_error[:200]}",
                    "edits": [],
                    "summary": "Auto-skipped non-dim error",
                }
        return None

    # ----------------------------------------------------------------
    # Rule 1: Auto-fix reshape/view parameter errors
    # ----------------------------------------------------------------

    def _try_fix_reshape(self, sample_path: str, error_output: str) -> dict | None:
        """Try to auto-fix reshape/view parameter errors.

        Matches: RuntimeError: shape '[1, 576, 3, 16, 128]' is invalid for input of size 497664
        """
        match = re.search(
            r"shape '\[([\d,\s\-]+)\]' is invalid for input of size (\d+)",
            error_output,
        )
        if not match:
            return None

        target_shape = self._parse_reshape_shape(match.group(1))
        actual_size = int(match.group(2))
        if not target_shape:
            return None

        # Get traceback info
        tb_info = self._extract_traceback_info(error_output, sample_path)
        if not tb_info:
            return None

        file_path = tb_info["file"]
        line_num = tb_info["line"]

        # Read the error line
        code_line = self._read_code_line(file_path, line_num)
        if not code_line:
            return None

        # Find reshape/view call
        reshape_calls = list(re.finditer(r"\.(reshape|view)\s*\(([^)]+)\)", code_line))
        if len(reshape_calls) != 1:
            return None  # Safety: zero or multiple reshape/view on same line

        call_match = reshape_calls[0]
        method = call_match.group(1)
        args_str = call_match.group(2)

        # Parse args - only handle integer literals and -1
        arg_tokens = [a.strip() for a in args_str.split(",")]
        for token in arg_tokens:
            if token == "-1":
                continue
            if not token.lstrip("-").isdigit():
                return None  # Contains variable names, unsafe

        # Compute correct shape
        correct_shape = self._compute_correct_shape(target_shape, actual_size)
        if correct_shape is None:
            return None

        # Build replacement
        old_call = f".{method}({args_str})"
        new_args_str = ", ".join(str(d) for d in correct_shape)
        new_call = f".{method}({new_args_str})"

        if old_call == new_call:
            return None  # No change needed

        # Apply the edit
        if not self._apply_line_edit(file_path, line_num, old_call, new_call):
            return None

        return {
            "action": "fix",
            "reason": "",
            "edits": [{"file": file_path, "old": old_call, "new": new_call}],
            "summary": f"Auto-fixed {method}: {target_shape} -> {correct_shape}",
        }

    # ----------------------------------------------------------------
    # Rule 2: Auto-fix pos_embed shape mismatch
    # ----------------------------------------------------------------

    def _try_fix_pos_embed(self, sample_path: str, error_output: str) -> dict | None:
        """Try to auto-fix pos_embed shape mismatch.

        Matches: RuntimeError: The size of tensor a (65) must match the size of
                 tensor b (197) at non-singleton dimension 1
        And the error line contains a pos_embed variable.
        """
        match = re.search(
            r"The size of tensor a \((\d+)\) must match the size of tensor b \((\d+)\) at non-singleton dimension (\d+)",
            error_output,
        )
        if not match:
            return None

        new_size = int(match.group(1))
        old_size = int(match.group(2))
        dim_index = int(match.group(3))

        if new_size == old_size:
            return None

        # Get traceback info
        tb_info = self._extract_traceback_info(error_output, sample_path)
        if not tb_info:
            return None

        # Check if the error line contains pos_embed variable
        code_line = self._read_code_line(tb_info["file"], tb_info["line"])
        if not code_line or "pos_embed" not in code_line:
            return None

        # Find the pos_embed variable name (pattern: l_xxx_pos_embed_xxx)
        var_match = re.search(r"(l_\w*pos_embed\w*)", code_line)
        if not var_match:
            return None

        var_name = var_match.group(1)

        # Find in weight_meta.py
        weight_meta_path = os.path.join(sample_path, "weight_meta.py")
        if not os.path.isfile(weight_meta_path):
            return None

        shape_info = self._find_weight_meta_entry(weight_meta_path, var_name)
        if not shape_info:
            return None

        old_shape = shape_info["shape"]
        old_shape_str = shape_info["shape_str"]

        if dim_index >= len(old_shape):
            return None

        if old_shape[dim_index] != old_size:
            return None  # Shape doesn't match expected old size

        # Build new shape
        new_shape = list(old_shape)
        new_shape[dim_index] = new_size
        new_shape_tuple = tuple(new_shape)
        new_shape_str = ", ".join(str(d) for d in new_shape_tuple)

        # Check against original sample (must differ from original)
        original_path = self._find_original_sample(sample_path)
        if original_path:
            orig_weight_meta = os.path.join(original_path, "weight_meta.py")
            if os.path.isfile(orig_weight_meta):
                orig_shape_info = self._find_weight_meta_entry(
                    orig_weight_meta, var_name
                )
                if orig_shape_info and orig_shape_info["shape"] == new_shape_tuple:
                    return None  # Would revert to original, not allowed

        # Apply the edit
        if not self._apply_weight_meta_edit(
            weight_meta_path, var_name, old_shape_str, new_shape_str
        ):
            return None

        return {
            "action": "fix",
            "reason": "",
            "edits": [
                {
                    "file": weight_meta_path,
                    "old": f'"{var_name}": ({old_shape_str},',
                    "new": f'"{var_name}": ({new_shape_str},',
                }
            ],
            "summary": f"Auto-fixed pos_embed shape: {old_shape} -> {new_shape_tuple} in weight_meta.py",
        }

    # ----------------------------------------------------------------
    # Rule 3: Auto-fix expand dimension mismatch
    # ----------------------------------------------------------------

    def _try_fix_expand(self, sample_path: str, error_output: str) -> dict | None:
        """Try to auto-fix expand dimension mismatch for rel_pos_bias etc.

        Matches: RuntimeError: The expanded size of the tensor (145) must match the
                 existing size (197) at non-singleton dimension 3.
                 Target sizes: [1, 12, 145, 145]. Tensor sizes: [1, 12, 197, 197]
        """
        match = re.search(
            r"The expanded size of the tensor \((\d+)\) must match the existing size \((\d+)\) at non-singleton dimension (\d+)\.\s*Target sizes: \[([^\]]+)\]\.\s*Tensor sizes: \[([^\]]+)\]",
            error_output,
        )
        if not match:
            return None

        target_sizes = self._parse_reshape_shape(match.group(4))
        tensor_sizes = self._parse_reshape_shape(match.group(5))

        if not target_sizes or not tensor_sizes:
            return None

        if len(target_sizes) != len(tensor_sizes):
            return None

        # Find differing dimensions
        diffs = []
        for i in range(len(target_sizes)):
            if target_sizes[i] != tensor_sizes[i]:
                diffs.append(i)

        if len(diffs) > 2:
            return None  # Too many differences

        # Get traceback info
        tb_info = self._extract_traceback_info(error_output, sample_path)
        if not tb_info:
            return None

        code_line = self._read_code_line(tb_info["file"], tb_info["line"])
        if not code_line:
            return None

        # Check variable name contains relative_position or pos
        if not re.search(r"relative_position|pos", code_line, re.IGNORECASE):
            return None

        var_match = re.search(
            r"(l_\w*(?:relative_position|pos)\w*)", code_line, re.IGNORECASE
        )
        if not var_match:
            return None

        var_name = var_match.group(1)

        # Find in weight_meta.py
        weight_meta_path = os.path.join(sample_path, "weight_meta.py")
        if not os.path.isfile(weight_meta_path):
            return None

        shape_info = self._find_weight_meta_entry(weight_meta_path, var_name)
        if not shape_info:
            return None

        old_shape = shape_info["shape"]
        old_shape_str = shape_info["shape_str"]

        # Modify the shape to match target_sizes
        new_shape = list(old_shape)
        for d in diffs:
            if d < len(new_shape):
                new_shape[d] = target_sizes[d]

        new_shape_tuple = tuple(new_shape)
        new_shape_str = ", ".join(str(d) for d in new_shape_tuple)

        # Check against original sample (must differ from original)
        original_path = self._find_original_sample(sample_path)
        if original_path:
            orig_weight_meta = os.path.join(original_path, "weight_meta.py")
            if os.path.isfile(orig_weight_meta):
                orig_shape_info = self._find_weight_meta_entry(
                    orig_weight_meta, var_name
                )
                if orig_shape_info and orig_shape_info["shape"] == new_shape_tuple:
                    return None  # Would revert to original

        # Apply the edit
        if not self._apply_weight_meta_edit(
            weight_meta_path, var_name, old_shape_str, new_shape_str
        ):
            return None

        return {
            "action": "fix",
            "reason": "",
            "edits": [
                {
                    "file": weight_meta_path,
                    "old": f'"{var_name}": ({old_shape_str},',
                    "new": f'"{var_name}": ({new_shape_str},',
                }
            ],
            "summary": f"Auto-fixed expand shape: {old_shape} -> {new_shape_tuple} in weight_meta.py",
        }

    # ----------------------------------------------------------------
    # Rule 4: Auto-fix multiple -1 dimensions
    # ----------------------------------------------------------------

    def _try_fix_multi_minus_one(
        self, sample_path: str, error_output: str
    ) -> dict | None:
        """Try to auto-fix multiple -1 dimensions in reshape/view.

        Matches: RuntimeError: only one dimension can be inferred
        """
        if "only one dimension can be inferred" not in error_output:
            return None

        # Get traceback info
        tb_info = self._extract_traceback_info(error_output, sample_path)
        if not tb_info:
            return None

        file_path = tb_info["file"]
        line_num = tb_info["line"]

        # Read the error line
        code_line = self._read_code_line(file_path, line_num)
        if not code_line:
            return None

        # Find reshape/view call
        reshape_calls = list(re.finditer(r"\.(reshape|view)\s*\(([^)]+)\)", code_line))
        if len(reshape_calls) != 1:
            return None

        call_match = reshape_calls[0]
        method = call_match.group(1)
        args_str = call_match.group(2)

        # Parse args
        arg_tokens = [a.strip() for a in args_str.split(",")]
        minus_one_count = sum(1 for t in arg_tokens if t == "-1")
        if minus_one_count < 2:
            return None

        # Check all args are integers or -1
        for token in arg_tokens:
            if token != "-1" and not token.lstrip("-").isdigit():
                return None

        # Find original sample
        original_path = self._find_original_sample(sample_path)
        if not original_path:
            return None

        # Read original model.py at the same line
        original_model_path = os.path.join(original_path, "model.py")
        if not os.path.isfile(original_model_path):
            return None

        orig_code_line = self._read_code_line(original_model_path, line_num)
        if not orig_code_line:
            return None

        # Find the corresponding reshape/view in the original
        orig_calls = list(
            re.finditer(r"\.(reshape|view)\s*\(([^)]+)\)", orig_code_line)
        )
        if len(orig_calls) != 1:
            return None

        orig_args_str = orig_calls[0].group(2)
        orig_arg_tokens = [a.strip() for a in orig_args_str.split(",")]

        # Restore -1 values from original
        new_arg_tokens = list(arg_tokens)
        for i, token in enumerate(arg_tokens):
            if token == "-1" and i < len(orig_arg_tokens):
                new_arg_tokens[i] = orig_arg_tokens[i]

        # Check if we still have at most one -1
        remaining_minus_one = sum(1 for t in new_arg_tokens if t == "-1")
        if remaining_minus_one > 1:
            return None

        # Build replacement
        old_call = f".{method}({args_str})"
        new_args_str = ", ".join(new_arg_tokens)
        new_call = f".{method}({new_args_str})"

        if old_call == new_call:
            return None

        # Apply the edit
        if not self._apply_line_edit(file_path, line_num, old_call, new_call):
            return None

        return {
            "action": "fix",
            "reason": "",
            "edits": [{"file": file_path, "old": old_call, "new": new_call}],
            "summary": f"Auto-fixed multiple -1 in {method}: restored from original sample",
        }

    # ----------------------------------------------------------------
    # Helper methods: error parsing
    # ----------------------------------------------------------------

    @staticmethod
    def _extract_last_error(error_output: str) -> str:
        """Extract the last error message from the output.

        Only checks the last Error/Exception line, not the full traceback,
        to avoid false positives from traceback noise.
        """
        error_lines = re.findall(
            r"^.*?(?:Error|Exception):.*$", error_output, re.MULTILINE
        )
        if error_lines:
            return error_lines[-1].strip()
        return ""

    @staticmethod
    def _extract_traceback_info(
        error_output: str, sample_path: str = ""
    ) -> dict | None:
        """Extract traceback info (file, line) from error output.

        Prefers entries in model.py or weight_meta.py files.
        Returns {"file": str, "line": int} or None.
        """
        pattern = r'File "([^"]+)", line (\d+)'
        matches = re.findall(pattern, error_output)

        if not matches:
            return None

        # Prefer entries in model.py or weight_meta.py
        for file_path, line_num in reversed(matches):
            basename = os.path.basename(file_path)
            if basename in ("model.py", "weight_meta.py"):
                return {"file": file_path, "line": int(line_num)}

        # Fall back to entries in the sample directory
        if sample_path:
            for file_path, line_num in reversed(matches):
                if file_path.startswith(sample_path):
                    return {"file": file_path, "line": int(line_num)}

        # Last resort: last entry
        file_path, line_num = matches[-1]
        return {"file": file_path, "line": int(line_num)}

    @staticmethod
    def _parse_reshape_shape(shape_str: str) -> list:
        """Parse shape string like '1, 576, 3, 16, 128' into a list of ints."""
        try:
            return [int(x.strip()) for x in shape_str.split(",") if x.strip()]
        except ValueError:
            return []

    # ----------------------------------------------------------------
    # Helper methods: shape computation
    # ----------------------------------------------------------------

    @staticmethod
    def _compute_correct_shape(target_shape: list, actual_size: int) -> list | None:
        """Compute the correct shape given target shape and actual element count.

        Handles:
        - Single -1 dimension: compute the inferred value
        - Single dimension that needs changing
        - H/W spatial dimension pair (adjacent same-value dims)
        - Two dimensions that need changing together
        """
        from math import isqrt

        if not target_shape:
            return None

        # Compute product of known dimensions
        product = 1
        has_minus_one = False
        for d in target_shape:
            if d == -1:
                has_minus_one = True
            else:
                product *= d

        if product == actual_size and not has_minus_one:
            return None  # Already correct

        # Case 1: Single -1 - compute it
        if target_shape.count(-1) == 1:
            idx = target_shape.index(-1)
            other_product = 1
            for i, d in enumerate(target_shape):
                if i != idx and d != -1:
                    other_product *= d
            if other_product == 0 or actual_size % other_product != 0:
                return None
            new_shape = list(target_shape)
            new_shape[idx] = actual_size // other_product
            return new_shape

        # Case 2: No -1
        if not has_minus_one:
            # Try single dimension fix
            fixable_dims = []
            for i, dim_i in enumerate(target_shape):
                if dim_i == 0:
                    continue
                other_product = 1
                for j, d in enumerate(target_shape):
                    if j != i:
                        other_product *= d
                if other_product > 0 and actual_size % other_product == 0:
                    new_val = actual_size // other_product
                    if new_val != dim_i and new_val > 0:
                        fixable_dims.append((i, new_val))

            if len(fixable_dims) == 1:
                new_shape = list(target_shape)
                new_shape[fixable_dims[0][0]] = fixable_dims[0][1]
                return new_shape

            # When multiple dims are fixable, prefer the one whose new value
            # is a perfect square (typical seq_len like 64, 81, 196)
            if len(fixable_dims) > 1:
                perfect_square_fixes = [
                    (i, v) for i, v in fixable_dims if isqrt(v) * isqrt(v) == v
                ]
                if len(perfect_square_fixes) == 1:
                    new_shape = list(target_shape)
                    new_shape[perfect_square_fixes[0][0]] = perfect_square_fixes[0][1]
                    return new_shape

            # Try H/W spatial dimension pair fix
            for i in range(len(target_shape) - 1):
                if target_shape[i] == target_shape[i + 1] and target_shape[i] > 0:
                    other_product = 1
                    for j, d in enumerate(target_shape):
                        if j != i and j != i + 1:
                            other_product *= d
                    if other_product > 0 and actual_size % other_product == 0:
                        hw_product = actual_size // other_product
                        sqrt_val = isqrt(hw_product)
                        if (
                            sqrt_val * sqrt_val == hw_product
                            and sqrt_val != target_shape[i]
                        ):
                            new_shape = list(target_shape)
                            new_shape[i] = sqrt_val
                            new_shape[i + 1] = sqrt_val
                            return new_shape

            # If exactly 2 dims can be fixed individually, try them together
            if len(fixable_dims) == 2:
                i1, v1 = fixable_dims[0]
                i2, v2 = fixable_dims[1]
                new_shape = list(target_shape)
                new_shape[i1] = v1
                new_shape[i2] = v2
                new_product = 1
                for d in new_shape:
                    new_product *= d
                if new_product == actual_size:
                    return new_shape

        return None

    # ----------------------------------------------------------------
    # Helper methods: code operations
    # ----------------------------------------------------------------

    @staticmethod
    def _read_code_line(file_path: str, line_num: int) -> str | None:
        """Read a specific line from a file (1-based line number)."""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
            if 1 <= line_num <= len(lines):
                return lines[line_num - 1].rstrip()
            return None
        except Exception:
            return None

    @staticmethod
    def _apply_line_edit(
        file_path: str, line_num: int, old_str: str, new_str: str
    ) -> bool:
        """Apply an edit to a specific line in a file.

        Reads the file, verifies old_str exists on the specified line,
        replaces old_str with new_str, and writes back.
        """
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            if old_str not in content:
                return False

            lines = content.split("\n")
            if line_num < 1 or line_num > len(lines):
                return False

            if old_str not in lines[line_num - 1]:
                return False

            new_content = content.replace(old_str, new_str, 1)

            with open(file_path, "w", encoding="utf-8") as f:
                f.write(new_content)

            return True
        except Exception:
            return False

    # ----------------------------------------------------------------
    # Helper methods: weight_meta operations
    # ----------------------------------------------------------------

    @staticmethod
    def _find_weight_meta_entry(weight_meta_path: str, var_name: str) -> dict | None:
        """Find a variable's shape definition in weight_meta.py.

        Returns {"shape": tuple, "shape_str": str} or None.
        The shape_str is the raw text between the tuple parentheses,
        used for exact string matching during edits.
        """
        try:
            with open(weight_meta_path, "r", encoding="utf-8") as f:
                content = f.read()

            # Match: "var_name": ((1, 197, 768), torch.float32)
            pattern = rf'"{re.escape(var_name)}"\s*:\s*\(\(([^)]+)\)'
            match = re.search(pattern, content)
            if not match:
                return None

            shape_str = match.group(1).strip()
            try:
                shape = tuple(int(x.strip()) for x in shape_str.split(",") if x.strip())
            except ValueError:
                return None

            return {"shape": shape, "shape_str": shape_str}
        except Exception:
            return None

    @staticmethod
    def _apply_weight_meta_edit(
        weight_meta_path: str, var_name: str, old_shape_str: str, new_shape_str: str
    ) -> bool:
        """Apply a shape edit in weight_meta.py.

        Tries multiple format patterns:
        1. "var_name": ((shape),  -- shape as inner tuple with closing paren
        2. "var_name": ((shape,   -- shape as inner tuple without closing paren
        3. "var_name": (shape,    -- shape as flat tuple
        """
        try:
            with open(weight_meta_path, "r", encoding="utf-8") as f:
                content = f.read()

            # Try with double parentheses and inner tuple close: ((shape),
            old_pattern = f'"{var_name}": (({old_shape_str}),'
            new_pattern = f'"{var_name}": (({new_shape_str}),'

            if old_pattern in content:
                new_content = content.replace(old_pattern, new_pattern, 1)
                with open(weight_meta_path, "w", encoding="utf-8") as f:
                    f.write(new_content)
                return True

            # Try with double parentheses without close: ((shape,
            old_pattern = f'"{var_name}": (({old_shape_str},'
            new_pattern = f'"{var_name}": (({new_shape_str},'

            if old_pattern in content:
                new_content = content.replace(old_pattern, new_pattern, 1)
                with open(weight_meta_path, "w", encoding="utf-8") as f:
                    f.write(new_content)
                return True

            # Try with single parentheses: (shape,
            old_pattern = f'"{var_name}": ({old_shape_str},'
            new_pattern = f'"{var_name}": ({new_shape_str},'

            if old_pattern in content:
                new_content = content.replace(old_pattern, new_pattern, 1)
                with open(weight_meta_path, "w", encoding="utf-8") as f:
                    f.write(new_content)
                return True

            return False
        except Exception:
            return False

    # ----------------------------------------------------------------
    # Helper methods: original sample lookup
    # ----------------------------------------------------------------

    def _find_original_sample(self, sample_path: str) -> str | None:
        """Find the original (non-dimension-generalized) sample path.

        Searches the original_samples_dir for a directory with the same
        basename as the given sample.
        """
        if not self.original_samples_dir or not os.path.isdir(
            self.original_samples_dir
        ):
            return None

        sample_name = os.path.basename(sample_path)

        # Walk through original_samples_dir to find a matching directory
        for root, dirs, files in os.walk(self.original_samples_dir):
            if os.path.basename(root) == sample_name and "model.py" in files:
                return root

        return None


# ============================================================
# ModelRunner - Execute run_model and capture output
# ============================================================


class ModelRunner:
    """Execute graph_net.torch.run_model and capture output."""

    def __init__(self, timeout: int, gpu_id: int | None = None):
        self.timeout = timeout
        self.gpu_id = gpu_id

    def run(self, sample_path: str) -> dict:
        """
        Execute run_model.

        Returns:
            {
                "success": bool,
                "returncode": int,
                "output": str,      # Full output (stdout + stderr)
            }
        """
        cmd = [
            sys.executable,
            "-m",
            "graph_net.torch.run_model",
            "--model-path",
            sample_path,
        ]
        env = os.environ.copy()
        if self.gpu_id is not None:
            env["CUDA_VISIBLE_DEVICES"] = str(self.gpu_id)
        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout,
                cwd=self._get_project_root(),
                env=env,
            )
            output = proc.stdout + proc.stderr
            return {
                "success": proc.returncode == 0,
                "returncode": proc.returncode,
                "output": output,
            }
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "returncode": -1,
                "output": f"TIMEOUT: run_model exceeded {self.timeout}s",
            }
        except Exception as e:
            return {
                "success": False,
                "returncode": -1,
                "output": f"ERROR: {e}",
            }

    @staticmethod
    def _get_project_root() -> str:
        """Get the project root directory."""
        # Script is in graph_net/agent/dim_gen_fixer/, root is its grand-grand-parent directory
        script_dir = os.path.dirname(os.path.abspath(__file__))
        return os.path.dirname(os.path.dirname(os.path.dirname(script_dir)))


# ============================================================
# FixLogger - Write per-sample fix_log.md and summary report
# ============================================================


class FixLogger:
    """Write fix log files."""

    def __init__(self, log_dir: str):
        self.log_dir = log_dir

    def write_sample_log(
        self,
        sample_name: str,
        sample_path: str,
        status: str,
        retry_count: int,
        attempts: list,
        final_output_summary: str,
    ):
        """Write per-sample fix_log.md."""
        sample_log_dir = os.path.join(self.log_dir, sample_name)
        os.makedirs(sample_log_dir, exist_ok=True)
        log_path = os.path.join(sample_log_dir, "fix_log.md")

        lines = [
            f"# {sample_name} Fix Log",
            "",
            f"- **Sample Path**: {sample_path}",
            f"- **Status**: {status}",
            f"- **Fix Count**: {retry_count}",
            "",
        ]

        if not attempts:
            lines.append("## Change Log")
            lines.append("")
            lines.append("No fix needed")
            lines.append("")
        else:
            lines.append("## Change Log")
            lines.append("")
            for i, attempt in enumerate(attempts, 1):
                lines.append(f"### Attempt {i}")
                lines.append("")
                lines.append(
                    f"- **Error Message**: {attempt.get('error_summary', 'N/A')}"
                )
                if attempt.get("action") == "skip_sample":
                    lines.append(f"- **Skip Reason**: {attempt.get('reason', 'N/A')}")
                else:
                    for edit in attempt.get("edits", []):
                        lines.append(f"- **Fixed File**: {edit.get('file', 'N/A')}")
                        lines.append("- **Original Code**:")
                        lines.append("```python")
                        lines.append(edit.get("old", ""))
                        lines.append("```")
                        lines.append("- **Changed To**:")
                        lines.append("```python")
                        lines.append(edit.get("new", ""))
                        lines.append("```")
                    if attempt.get("summary"):
                        lines.append(f"- **Fix Reason**: {attempt['summary']}")
                lines.append("")

        lines.append("## Final Status")
        lines.append("")
        if status.startswith("OK"):
            lines.append("- **Exit Code**: 0")
        else:
            lines.append("- **Exit Code**: Non-zero")
        lines.append(f"- **Final Output**: {final_output_summary}")
        lines.append("")

        with open(log_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

    def write_report(
        self,
        samples_dir: str,
        results: list,
    ):
        """Write summary report fix_samples_report.md."""
        report_path = os.path.join(self.log_dir, "fix_samples_report.md")
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        total = len(results)
        original_ok = sum(1 for r in results if r["status"] == "OK(original)")
        fixed_ok = sum(1 for r in results if r["status"] == "OK(fixed)")
        failed = sum(1 for r in results if r["status"] == "Failed")
        skipped_dim = sum(1 for r in results if r["status"] == "Skipped(non-dim error)")
        skipped_res = sum(
            1 for r in results if r["status"] == "Skipped(insufficient resources)"
        )
        success_rate = (
            f"{(original_ok + fixed_ok) / total * 100:.1f}%" if total else "N/A"
        )

        # Preprocessing statistics
        preprocessed_skip = sum(
            1 for r in results if r.get("preprocess_action") == "skip"
        )
        preprocessed_fix_ok = sum(
            1
            for r in results
            if r.get("preprocess_action") == "fix" and r["status"].startswith("OK")
        )
        preprocessed_fix_failed = sum(
            1
            for r in results
            if r.get("preprocess_action") == "fix" and r["status"] == "Failed"
        )
        api_fix_ok = sum(
            1
            for r in results
            if r.get("preprocess_action") in ("unhandled", None)
            and r["status"] == "OK(fixed)"
        )

        lines = [
            "# Sample Auto-Fix Report",
            "",
            f"- **Sample Directory**: {samples_dir}",
            f"- **Execution Time**: {now}",
            f"- **Max Retries**: {max(r.get('max_retries', 0) for r in results) if results else 0}",
            "",
            "## Summary Statistics",
            "",
            "| Metric | Count |",
            "|--------|-------|",
            f"| Total Samples | {total} |",
            f"| OK (original) | {original_ok} |",
            f"| OK (after fix) | {fixed_ok} |",
            f"| Fix Failed | {failed} |",
            f"| Skipped (non-dim error) | {skipped_dim} |",
            f"| Skipped (insufficient resources) | {skipped_res} |",
            "",
            f"Success Rate: {success_rate}",
            "",
            "## Preprocessing Statistics",
            "",
            "| Metric | Count |",
            "|--------|-------|",
            f"| Preprocessed auto-skip | {preprocessed_skip} |",
            f"| Preprocessed auto-fix success | {preprocessed_fix_ok} |",
            f"| Preprocessed auto-fix then failed | {preprocessed_fix_failed} |",
            f"| API fix success | {api_fix_ok} |",
            "",
            "## Sample Results",
            "",
            "| Sample Name | Status | Fix Count | Log Path |",
            "|-------------|--------|-----------|----------|",
        ]

        for r in results:
            log_rel = os.path.relpath(
                os.path.join(self.log_dir, r["sample_name"], "fix_log.md")
            )
            lines.append(
                f"| {r['sample_name']} | {r['status']} | {r['retry_count']} | {log_rel} |"
            )

        # Failed samples list
        failed_samples = [r for r in results if r["status"] == "Failed"]
        if failed_samples:
            lines.append("")
            lines.append("## Failed Samples")
            lines.append("")
            lines.append("| Sample Name | Last Error Type | Last Error Summary |")
            lines.append("|-------------|-----------------|--------------------|")
            for r in failed_samples:
                err_summary = r.get("last_error_summary", "N/A")
                err_type = r.get("last_error_type", "Unknown")
                # Truncate long error messages
                if len(err_summary) > 80:
                    err_summary = err_summary[:77] + "..."
                lines.append(f"| {r['sample_name']} | {err_type} | {err_summary} |")

        # Skipped samples list (non-dimension error)
        skipped_dim_samples = [
            r for r in results if r["status"] == "Skipped(non-dim error)"
        ]
        if skipped_dim_samples:
            lines.append("")
            lines.append("## Skipped Samples (Non-Dimension Error)")
            lines.append("")
            lines.append("| Sample Name | Error Type | Error Summary |")
            lines.append("|-------------|------------|---------------|")
            for r in skipped_dim_samples:
                err_summary = r.get("skip_reason", "N/A")
                if len(err_summary) > 80:
                    err_summary = err_summary[:77] + "..."
                lines.append(f"| {r['sample_name']} | Non-dim error | {err_summary} |")

        # Skipped samples list (insufficient resources)
        skipped_res_samples = [
            r for r in results if r["status"] == "Skipped(insufficient resources)"
        ]
        if skipped_res_samples:
            lines.append("")
            lines.append("## Skipped Samples (Insufficient Resources)")
            lines.append("")
            lines.append("| Sample Name | Skip Reason |")
            lines.append("|-------------|-------------|")
            for r in skipped_res_samples:
                lines.append(f"| {r['sample_name']} | {r.get('skip_reason', 'N/A')} |")

        lines.append("")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))


# ============================================================
# SampleProcessor - Complete fix flow for a single sample
# ============================================================


class SampleProcessor:
    """Complete fix flow for a single sample (retry loop)."""

    def __init__(
        self,
        agent: ClaudeFixAgent,
        runner: ModelRunner,
        logger: FixLogger,
        max_retries: int,
        original_samples_dir: str,
        preprocessor: ErrorPreprocessor | None = None,
    ):
        self.agent = agent
        self.runner = runner
        self.logger = logger
        self.max_retries = max_retries
        self.original_samples_dir = original_samples_dir
        self.preprocessor = preprocessor

    def process(self, sample_path: str) -> dict:
        """
        Process a single sample, return result dict.

        Returns:
            {
                "sample_name": str,
                "sample_path": str,
                "status": str,          # OK(original)/OK(fixed)/Failed/Skipped(non-dim error)/Skipped(insufficient resources)
                "retry_count": int,
                "attempts": list,       # Records of each attempt
                "last_error_summary": str,
                "last_error_type": str,
                "skip_reason": str,
                "max_retries": int,
            }
        """
        sample_name = os.path.basename(sample_path)
        attempts = []
        retry_count = 0

        print(f"\n{'='*60}")
        print(f"Processing sample: {sample_name}")
        print(f"Path: {sample_path}")
        print(f"{'='*60}")

        preprocess_action = None  # Track first preprocess action for reporting

        for retry in range(self.max_retries + 1):
            # Step B: Execute run_model
            print(f"\n  [{retry}] Running run_model...")
            result = self.runner.run(sample_path)

            if result["success"]:
                # Execution succeeded
                if retry == 0:
                    status = "OK(original)"
                    print("  OK: Sample passes without fix")
                else:
                    status = "OK(fixed)"
                    print(f"  OK: Sample passes after fix (attempt {retry})")

                self.logger.write_sample_log(
                    sample_name,
                    sample_path,
                    status,
                    retry_count,
                    attempts,
                    "Execution succeeded",
                )
                return {
                    "sample_name": sample_name,
                    "sample_path": sample_path,
                    "status": status,
                    "retry_count": retry_count,
                    "attempts": attempts,
                    "last_error_summary": "",
                    "last_error_type": "",
                    "skip_reason": "",
                    "max_retries": self.max_retries,
                    "preprocess_action": preprocess_action,
                }

            # Execution failed
            error_output = result["output"]
            error_summary = self._extract_error_summary(error_output)
            error_type = self._extract_error_type(error_output)

            # Check for insufficient resources
            if self._is_resource_error(error_output):
                status = "Skipped(insufficient resources)"
                skip_reason = self._extract_resource_reason(error_output)
                print(f"  SKIP (insufficient resources): {skip_reason}")
                attempts.append(
                    {
                        "error_summary": error_summary,
                        "action": "skip_sample",
                        "reason": skip_reason,
                        "edits": [],
                    }
                )
                self.logger.write_sample_log(
                    sample_name,
                    sample_path,
                    status,
                    retry_count,
                    attempts,
                    error_summary,
                )
                return {
                    "sample_name": sample_name,
                    "sample_path": sample_path,
                    "status": status,
                    "retry_count": retry_count,
                    "attempts": attempts,
                    "last_error_summary": error_summary,
                    "last_error_type": error_type,
                    "skip_reason": skip_reason,
                    "max_retries": self.max_retries,
                    "preprocess_action": preprocess_action,
                }

            if retry >= self.max_retries:
                break

            # Step A: Backup before first fix
            if retry == 0:
                self._backup_files(sample_path)

            # Step B: Try preprocessing before calling Claude API
            if self.preprocessor:
                preprocess_result = self.preprocessor.try_preprocess(
                    sample_path, error_output
                )

                if preprocess_result["action"] == "skip":
                    preprocess_action = "skip"
                    reason = preprocess_result.get(
                        "reason", "Auto-skipped by preprocessor"
                    )
                    status = "Skipped(non-dim error)"
                    print(f"  PREPROCESS SKIP: {reason}")
                    attempts.append(
                        {
                            "error_summary": error_summary,
                            "action": "skip_sample",
                            "reason": reason,
                            "edits": preprocess_result.get("edits", []),
                            "summary": preprocess_result.get("summary", ""),
                        }
                    )
                    self.logger.write_sample_log(
                        sample_name,
                        sample_path,
                        status,
                        retry_count,
                        attempts,
                        error_summary,
                    )
                    return {
                        "sample_name": sample_name,
                        "sample_path": sample_path,
                        "status": status,
                        "retry_count": retry_count,
                        "attempts": attempts,
                        "last_error_summary": error_summary,
                        "last_error_type": error_type,
                        "skip_reason": reason,
                        "max_retries": self.max_retries,
                        "preprocess_action": "skip",
                    }

                if preprocess_result["action"] == "fix":
                    if preprocess_action is None:
                        preprocess_action = "fix"
                    retry_count += 1
                    summary = preprocess_result.get("summary", "Preprocessed auto-fix")
                    print(f"  PREPROCESS FIX: {summary}")
                    attempts.append(
                        {
                            "error_summary": error_summary,
                            "action": "finish_fix",
                            "summary": summary,
                            "edits": preprocess_result.get("edits", []),
                        }
                    )
                    # Continue loop, re-execute run_model to verify
                    continue

            # Step C: Call Claude to analyze the error and fix
            print(f"  [{retry}] Calling Claude to analyze error...")
            fix_result = self.agent.fix_sample(
                sample_path=sample_path,
                error_output=error_output,
                original_samples_dir=self.original_samples_dir,
            )

            retry_count += 1

            if fix_result["action"] == "skip_sample":
                reason = fix_result.get("reason", "Unknown reason")
                # Determine if it's a dimension error that can't be fixed, or a non-dimension error
                if self._is_dim_error_but_unfixable(reason, error_output):
                    status = "Failed"
                else:
                    status = "Skipped(non-dim error)"
                print(f"  SKIP: {reason}")
                attempts.append(
                    {
                        "error_summary": error_summary,
                        "action": "skip_sample",
                        "reason": reason,
                        "edits": fix_result.get("edits", []),
                        "summary": fix_result.get("reason", ""),
                    }
                )
                self.logger.write_sample_log(
                    sample_name,
                    sample_path,
                    status,
                    retry_count,
                    attempts,
                    error_summary,
                )
                return {
                    "sample_name": sample_name,
                    "sample_path": sample_path,
                    "status": status,
                    "retry_count": retry_count,
                    "attempts": attempts,
                    "last_error_summary": error_summary,
                    "last_error_type": error_type,
                    "skip_reason": reason,
                    "max_retries": self.max_retries,
                    "preprocess_action": preprocess_action,
                }

            # finish_fix - record the modification
            summary = fix_result.get("summary", "")
            print(f"  OK: Claude completed fix: {summary}")
            attempts.append(
                {
                    "error_summary": error_summary,
                    "action": "finish_fix",
                    "summary": summary,
                    "edits": fix_result.get("edits", []),
                }
            )
            # Continue loop, re-execute run_model

        # Reached max retries
        status = "Failed"
        print(f"  FAIL: Reached max retries ({self.max_retries})")
        self.logger.write_sample_log(
            sample_name,
            sample_path,
            status,
            retry_count,
            attempts,
            error_summary if attempts else "Reached max retries",
        )
        return {
            "sample_name": sample_name,
            "sample_path": sample_path,
            "status": status,
            "retry_count": retry_count,
            "attempts": attempts,
            "last_error_summary": error_summary if attempts else "",
            "last_error_type": error_type if attempts else "",
            "skip_reason": "",
            "max_retries": self.max_retries,
            "preprocess_action": preprocess_action,
        }

    @staticmethod
    def _backup_files(sample_path: str):
        """Backup model.py and weight_meta.py before the first fix."""
        for fname in ("model.py", "weight_meta.py"):
            src = os.path.join(sample_path, fname)
            bak = os.path.join(sample_path, fname + ".bak")
            if os.path.isfile(src) and not os.path.isfile(bak):
                try:
                    import shutil

                    shutil.copy2(src, bak)
                except Exception as e:
                    print(f"  Warning: Failed to backup {fname}: {e}")

    @staticmethod
    def _extract_error_summary(output: str) -> str:
        """Extract error output summary (last few lines)."""
        lines = output.strip().split("\n")
        # Take the last 5 lines
        summary_lines = lines[-5:] if len(lines) > 5 else lines
        return "\n".join(summary_lines)

    @staticmethod
    def _extract_error_type(output: str) -> str:
        """Extract error type from error output."""
        # Match common error types
        patterns = [
            r"(\w+Error):",
            r"(\w+Exception):",
        ]
        for pattern in patterns:
            matches = re.findall(pattern, output)
            if matches:
                return matches[-1]
        return "Unknown"

    @staticmethod
    def _is_resource_error(output: str) -> bool:
        """Check if the error is due to insufficient resources."""
        resource_keywords = [
            "CUDA out of memory",
            "out of memory",
            "OOM",
            "TIMEOUT",
            "Killed",
        ]
        output_lower = output.lower()
        return any(kw.lower() in output_lower for kw in resource_keywords)

    @staticmethod
    def _extract_resource_reason(output: str) -> str:
        """Extract the reason for insufficient resources."""
        if "CUDA out of memory" in output or "out of memory" in output:
            return "CUDA out of memory"
        if "TIMEOUT" in output:
            return "Execution timeout"
        if "Killed" in output:
            return "Process killed (likely OOM)"
        return "Insufficient resources"

    @staticmethod
    def _is_dim_error_but_unfixable(reason: str, error_output: str) -> bool:
        """Check if it's a dimension error that cannot be fixed in-place.

        Only checks the skip reason from Claude, not the full error output,
        because the error output may contain dimension-related words in the
        traceback even for non-dimension errors (e.g., device mismatch).
        """
        dim_keywords = [
            "shape",
            "dimension",
            "size mismatch",
            "reshape",
            "view",
            "invalid for input of size",
            "only one dimension",
            "expand",
        ]
        reason_lower = reason.lower()
        # Only check the reason from Claude, trust Claude's classification
        return any(kw in reason_lower for kw in dim_keywords)


# ============================================================
# MainOrchestrator - Main workflow
# ============================================================


class MainOrchestrator:
    """Main workflow: scan -> process samples -> generate summary report.

    Parallelism strategy:
    - run_model execution is parallelized across GPUs (via ThreadPoolExecutor)
    - Claude API calls are serialized (single API key cannot handle concurrent
      requests at the gateway, concurrent calls get queued/blocked)
    - Flow: batch-run_model on all GPUs -> collect failures -> serial API fix
      -> batch-run_model to verify -> repeat
    """

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_dir = os.path.join(
            args.output_dir, f"fix_sample_logs-{self.timestamp}"
        )
        os.makedirs(self.log_dir, exist_ok=True)

        self.agent = ClaudeFixAgent(
            model=args.model,
            base_url=args.base_url,
            api_key=args.api_key,
            custom_header=args.custom_header,
        )
        self.logger = FixLogger(log_dir=self.log_dir)

        # Create error preprocessor for rule-based handling
        self.preprocessor = ErrorPreprocessor(
            original_samples_dir=args.original_samples_dir
        )

        # Parse GPU list for parallel run_model execution
        self.gpu_ids = self._parse_gpus(args.gpus)

        # ModelRunner for the processor: use first GPU if available,
        # otherwise no GPU binding (sequential mode)
        default_gpu = self.gpu_ids[0] if self.gpu_ids else None
        self.processor = SampleProcessor(
            agent=self.agent,
            runner=ModelRunner(timeout=args.timeout, gpu_id=default_gpu),
            logger=self.logger,
            max_retries=args.max_retries,
            original_samples_dir=args.original_samples_dir,
            preprocessor=self.preprocessor,
        )

    @staticmethod
    def _parse_gpus(gpus_str: str) -> list:
        """Parse GPU list from comma-separated string like '0,1,2,3'."""
        if not gpus_str:
            return []
        try:
            return [int(g.strip()) for g in gpus_str.split(",") if g.strip()]
        except ValueError:
            print(
                f"Error: Invalid --gpus format: {gpus_str}. "
                "Expected comma-separated integers like '0,1,2,3'"
            )
            sys.exit(1)

    def run(self):
        """Execute the main workflow."""
        print(f"Log directory: {self.log_dir}")
        print(f"Max retries: {self.args.max_retries}")
        print(f"Timeout: {self.args.timeout}s")
        print(f"Claude model: {self.args.model}")
        print(f"Original samples directory: {self.args.original_samples_dir}")
        print(f"GPUs for parallel run_model: {self.gpu_ids or 'sequential'}")

        # Step 3.1: Scan samples
        samples = self._scan_samples()
        print(f"\nFound {len(samples)} samples to process")

        if not samples:
            print("No samples to process, exiting.")
            return

        # If no sample_list_with_status provided, generate one in the log directory
        self.sample_list_path = self.args.sample_list_with_status
        if not self.sample_list_path:
            self.sample_list_path = os.path.join(self.log_dir, "sample_list.txt")
            self._generate_sample_list(self.sample_list_path, samples)
            print(f"Generated sample list: {self.sample_list_path}")

        # Step 3.2: Process samples
        if self.gpu_ids:
            results = self._process_samples_parallel(samples)
        else:
            results = self._process_samples_sequential(samples)

        # Step 3.3: Generate summary report
        samples_dir = self.args.samples_dir or ""
        self.logger.write_report(samples_dir, results)

        # Print summary
        self._print_summary(results)

    def _process_samples_sequential(self, samples: list) -> list:
        """Process samples sequentially (no GPU parallelism)."""
        results = []
        for i, sample_path in enumerate(samples, 1):
            print(f"\n--- [{i}/{len(samples)}] ---")
            result = self.processor.process(sample_path)
            results.append(result)
            self._update_sample_status(
                self.sample_list_path, sample_path, result["status"]
            )
        return results

    def _process_samples_parallel(self, samples: list) -> list:
        """Process samples with parallel run_model, serial API calls.

        Strategy:
        1. Batch-run all samples' run_model in parallel across GPUs
        2. Samples that pass -> record as OK(original)
        3. Samples that fail -> process through serial fix loop
           (run_model runs on the next available GPU, API calls are serial)
        """

        # Phase 1: Parallel initial run_model to identify failures
        print(f"\n=== Phase 1: Parallel run_model ({len(self.gpu_ids)} GPUs) ===")
        phase1_results = []  # Full result dicts for handled samples
        failed_samples = []  # (sample_path, error_output) for samples that need fixing

        run_results = self._batch_run_model(samples)

        for sample_path, run_result in zip(samples, run_results):
            sample_name = os.path.basename(sample_path)
            if run_result["success"]:
                print(f"  PASS: {sample_name}")
                self.logger.write_sample_log(
                    sample_name,
                    sample_path,
                    "OK(original)",
                    0,
                    [],
                    "Execution succeeded",
                )
                self._update_sample_status(
                    self.sample_list_path, sample_path, "OK(original)"
                )
                phase1_results.append(
                    {
                        "sample_name": sample_name,
                        "sample_path": sample_path,
                        "status": "OK(original)",
                        "retry_count": 0,
                        "attempts": [],
                        "last_error_summary": "",
                        "last_error_type": "",
                        "skip_reason": "",
                        "max_retries": self.args.max_retries,
                        "preprocess_action": None,
                    }
                )
            else:
                error_output = run_result["output"]
                error_summary = SampleProcessor._extract_error_summary(error_output)
                error_type = SampleProcessor._extract_error_type(error_output)

                # Check resource errors
                if SampleProcessor._is_resource_error(error_output):
                    skip_reason = SampleProcessor._extract_resource_reason(error_output)
                    print(f"  SKIP (resources): {sample_name} - {skip_reason}")
                    self.logger.write_sample_log(
                        sample_name,
                        sample_path,
                        "Skipped(insufficient resources)",
                        0,
                        [
                            {
                                "error_summary": error_summary,
                                "action": "skip_sample",
                                "reason": skip_reason,
                                "edits": [],
                            }
                        ],
                        error_summary,
                    )
                    self._update_sample_status(
                        self.sample_list_path,
                        sample_path,
                        "Skipped(insufficient resources)",
                    )
                    phase1_results.append(
                        {
                            "sample_name": sample_name,
                            "sample_path": sample_path,
                            "status": "Skipped(insufficient resources)",
                            "retry_count": 0,
                            "attempts": [
                                {
                                    "error_summary": error_summary,
                                    "action": "skip_sample",
                                    "reason": skip_reason,
                                    "edits": [],
                                }
                            ],
                            "last_error_summary": error_summary,
                            "last_error_type": error_type,
                            "skip_reason": skip_reason,
                            "max_retries": self.args.max_retries,
                            "preprocess_action": None,
                        }
                    )
                elif self.preprocessor:
                    # Try preprocessing to skip non-dim errors
                    preprocess_result = self.preprocessor.try_preprocess(
                        sample_path, error_output
                    )
                    if preprocess_result["action"] == "skip":
                        reason = preprocess_result.get("reason", "Auto-skipped")
                        print(f"  SKIP (preprocess): {sample_name} - {reason[:100]}")
                        self.logger.write_sample_log(
                            sample_name,
                            sample_path,
                            "Skipped(non-dim error)",
                            0,
                            [
                                {
                                    "error_summary": error_summary,
                                    "action": "skip_sample",
                                    "reason": reason,
                                    "edits": [],
                                    "summary": preprocess_result.get("summary", ""),
                                }
                            ],
                            error_summary,
                        )
                        self._update_sample_status(
                            self.sample_list_path, sample_path, "Skipped(non-dim error)"
                        )
                        phase1_results.append(
                            {
                                "sample_name": sample_name,
                                "sample_path": sample_path,
                                "status": "Skipped(non-dim error)",
                                "retry_count": 0,
                                "attempts": [
                                    {
                                        "error_summary": error_summary,
                                        "action": "skip_sample",
                                        "reason": reason,
                                        "edits": [],
                                        "summary": preprocess_result.get("summary", ""),
                                    }
                                ],
                                "last_error_summary": error_summary,
                                "last_error_type": error_type,
                                "skip_reason": reason,
                                "max_retries": self.args.max_retries,
                                "preprocess_action": "skip",
                            }
                        )
                    else:
                        print(f"  FAIL: {sample_name}")
                        failed_samples.append((sample_path, error_output))
                else:
                    print(f"  FAIL: {sample_name}")
                    failed_samples.append((sample_path, error_output))

        print(
            f"\nPhase 1 done: {len(phase1_results)} handled, "
            f"{len(failed_samples)} failed, need fix"
        )

        # Phase 2: Serial fix loop for failed samples
        # run_model is dispatched to GPUs in round-robin, API calls are serial
        results = list(phase1_results)

        # Process failed samples one by one (serial API, GPU-parallel run_model)
        if failed_samples:
            print(
                f"\n=== Phase 2: Fix {len(failed_samples)} failed samples (serial API) ==="
            )
        for i, (sample_path, _) in enumerate(failed_samples, 1):
            print(f"\n--- Fix [{i}/{len(failed_samples)}] ---")
            result = self.processor.process(sample_path)
            results.append(result)
            self._update_sample_status(
                self.sample_list_path, sample_path, result["status"]
            )

        # Sort by original sample order
        sample_order = {s: idx for idx, s in enumerate(samples)}
        results.sort(key=lambda r: sample_order.get(r.get("sample_path", ""), 0))
        return results

    def _batch_run_model(self, samples: list) -> list:
        """Run model on all samples in parallel across GPUs. Returns results list."""
        from concurrent.futures import ThreadPoolExecutor, as_completed

        num_gpus = len(self.gpu_ids)
        # Assign samples to GPUs round-robin
        gpu_assignments = {}
        for i, sample_path in enumerate(samples):
            gpu_id = self.gpu_ids[i % num_gpus]
            gpu_assignments.setdefault(gpu_id, []).append(sample_path)

        # Collect all (sample_path, gpu_id) pairs in order
        tasks = []
        for i, sample_path in enumerate(samples):
            tasks.append((sample_path, self.gpu_ids[i % num_gpus]))

        results_map = {}

        def run_on_gpu(sample_path: str, gpu_id: int) -> tuple:
            runner = ModelRunner(timeout=self.args.timeout, gpu_id=gpu_id)
            return (sample_path, runner.run(sample_path))

        with ThreadPoolExecutor(max_workers=num_gpus) as executor:
            futures = {executor.submit(run_on_gpu, sp, gid): sp for sp, gid in tasks}
            for future in as_completed(futures):
                try:
                    sample_path, run_result = future.result()
                    results_map[sample_path] = run_result
                except Exception as e:
                    sample_path = futures[future]
                    results_map[sample_path] = {
                        "success": False,
                        "returncode": -1,
                        "output": f"ERROR: {e}",
                    }

        # Return results in original sample order
        return [results_map[sp] for sp in samples]

    def _generate_sample_list(self, list_path: str, samples: list):
        """Generate a sample_list.txt file with all samples marked as FAILED."""
        with open(list_path, "w", encoding="utf-8") as f:
            for sample_path in samples:
                f.write(f"FAILED\t{sample_path}\n")

    def _scan_samples(self) -> list:
        """Scan samples to process, return list of paths."""
        if self.args.sample_list_with_status:
            return self._scan_from_status_file(self.args.sample_list_with_status)
        elif self.args.samples_dir:
            return self._scan_from_dir(self.args.samples_dir)
        else:
            print("Error: Must provide --samples-dir or --sample-list-with-status")
            sys.exit(1)

    def _scan_from_status_file(self, status_file: str) -> list:
        """Scan from a status file with sample list."""
        if not os.path.isfile(status_file):
            print(f"Error: Sample list file not found: {status_file}")
            sys.exit(1)

        samples = []
        with open(status_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.rstrip("\n")
                if not line.strip():
                    continue
                parts = line.split("\t", 1)
                if len(parts) < 2:
                    continue
                status = parts[0].strip()
                path = parts[1].strip()
                # FAILED or blank status -> needs processing
                if status in ("", "FAILED"):
                    samples.append(path)

        if self.args.max_samples > 0:
            samples = samples[: self.args.max_samples]

        return samples

    def _scan_from_dir(self, samples_dir: str) -> list:
        """Scan all subdirectories containing model.py from the given directory."""
        if not os.path.isdir(samples_dir):
            print(f"Error: Sample directory not found: {samples_dir}")
            sys.exit(1)

        samples = []
        for root, dirs, files in os.walk(samples_dir):
            if "model.py" in files:
                samples.append(root)

        samples.sort()

        if self.args.max_samples > 0:
            samples = samples[: self.args.max_samples]

        return samples

    @staticmethod
    def _update_sample_status(status_file: str, sample_path: str, status: str):
        """Update sample status in-place in the sample_list_with_status file."""
        if status.startswith("OK"):
            new_status = "PASS"
        else:
            new_status = "FAILED"

        try:
            with open(status_file, "r", encoding="utf-8") as f:
                lines = f.readlines()

            updated = []
            for line in lines:
                parts = line.rstrip("\n").split("\t", 1)
                if len(parts) >= 2 and parts[1].strip() == sample_path:
                    updated.append(f"{new_status}\t{sample_path}\n")
                else:
                    updated.append(line)

            with open(status_file, "w", encoding="utf-8") as f:
                f.writelines(updated)
        except Exception as e:
            print(f"  Warning: Failed to update status file: {e}")

    @staticmethod
    def _print_summary(results: list):
        """Print a brief summary."""
        total = len(results)
        original_ok = sum(1 for r in results if r["status"] == "OK(original)")
        fixed_ok = sum(1 for r in results if r["status"] == "OK(fixed)")
        failed = sum(1 for r in results if r["status"] == "Failed")
        skipped_dim = sum(1 for r in results if r["status"] == "Skipped(non-dim error)")
        skipped_res = sum(
            1 for r in results if r["status"] == "Skipped(insufficient resources)"
        )

        # Preprocessing statistics
        preprocessed_skip = sum(
            1 for r in results if r.get("preprocess_action") == "skip"
        )
        preprocessed_fix_ok = sum(
            1
            for r in results
            if r.get("preprocess_action") == "fix" and r["status"].startswith("OK")
        )
        preprocessed_fix_failed = sum(
            1
            for r in results
            if r.get("preprocess_action") == "fix" and r["status"] == "Failed"
        )
        api_fix_ok = sum(
            1
            for r in results
            if r.get("preprocess_action") in ("unhandled", None)
            and r["status"] == "OK(fixed)"
        )

        print(f"\n{'='*60}")
        print("Fix Summary")
        print(f"{'='*60}")
        print(f"Total samples:       {total}")
        print(f"OK (original):       {original_ok}")
        print(f"OK (after fix):      {fixed_ok}")
        print(f"Fix failed:          {failed}")
        print(f"Skipped (non-dim):   {skipped_dim}")
        print(f"Skipped (resources): {skipped_res}")
        if total:
            rate = (original_ok + fixed_ok) / total * 100
            print(f"Success rate:        {rate:.1f}%")
        print(f"{'='*60}")
        print("Preprocessing Stats:")
        print(f"  Auto-skipped:      {preprocessed_skip}")
        print(f"  Auto-fix success:  {preprocessed_fix_ok}")
        print(f"  Auto-fix failed:   {preprocessed_fix_failed}")
        print(f"  API fix success:   {api_fix_ok}")
        print(f"{'='*60}")


# ============================================================
# CLI
# ============================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Automatically fix dimension errors in dimension-generalized samples using the Anthropic API (Claude)"
    )
    parser.add_argument(
        "--samples-dir",
        type=str,
        default=None,
        help="Root directory of samples to fix, each subdirectory is a sample",
    )
    parser.add_argument(
        "--sample-list-with-status",
        type=str,
        default=None,
        help="Path to sample list file with status, format: STATUS<TAB>sample_absolute_path per line",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="Maximum number of samples to process, 0 means all (default: 0)",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=10,
        help="Maximum fix retry attempts per sample (default: 10)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=600,
        help="Timeout in seconds for each run_model execution (default: 600)",
    )
    parser.add_argument(
        "--original-samples-dir",
        type=str,
        default="samples/",
        help="Root directory of original (non-dimension-generalized) samples (default: samples/)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="claude-sonnet-4-6",
        help="Claude model name (default: claude-sonnet-4-6)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=".",
        help="Output directory for logs and reports (default: current directory)",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        required=True,
        help="Anthropic API base URL (required)",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        required=True,
        help="Anthropic API Key (required)",
    )
    parser.add_argument(
        "--gpus",
        type=str,
        default=None,
        help="Comma-separated GPU IDs for parallel run_model execution, e.g. '0,1,2,3' (default: sequential)",
    )
    parser.add_argument(
        "--custom-header",
        type=str,
        required=True,
        help="Custom header value for API requests (required)",
    )

    args = parser.parse_args()

    if not args.samples_dir and not args.sample_list_with_status:
        parser.error("Must provide --samples-dir or --sample-list-with-status")

    return args


def main():
    args = parse_args()
    orchestrator = MainOrchestrator(args)
    orchestrator.run()


if __name__ == "__main__":
    main()
