# graph_net/status.py

import os
import sys
import argparse


def list_directory(src_dir: str):
    """
    List all files and subdirectories directly under src_dir.
    """
    try:
        entries = os.listdir(src_dir)
    except OSError as e:
        print(f"Error reading directory '{src_dir}': {e}", file=sys.stderr)
        sys.exit(1)

    if not entries:
        print(f"No files or directories in '{src_dir}'.")
        return

    for name in sorted(entries):
        print(name)


def main():
    parser = argparse.ArgumentParser(
        prog="python -m graph_net.status",
        description="List contents of the $GRAPH_NET_EXTRACT_WORKSPACE directory (like ls)",
    )
    args = parser.parse_args()

    ws = os.environ.get("GRAPH_NET_EXTRACT_WORKSPACE")
    if not ws:
        parser.error("Environment variable GRAPH_NET_EXTRACT_WORKSPACE is not set")
    if not os.path.isdir(ws):
        parser.error(
            f'The path specified by GRAPH_NET_EXTRACT_WORKSPACE ("{ws}") is not a valid directory'
        )

    list_directory(ws)


if __name__ == "__main__":
    main()
