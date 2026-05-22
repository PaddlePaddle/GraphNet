import os
import re


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
