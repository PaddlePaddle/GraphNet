import os
import re


MAX_FILE_SIZE = 500 * 1024  # read_file rejects files larger than 500KB


# Tool Definitions for Claude API
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
