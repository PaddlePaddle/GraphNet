# graph_net/unpack.py

import os
import sys
import zipfile
import argparse


def unpack_archive(src_path: str, dest_dir: str):
    """
    Unpack the ZIP archive at src_path into the directory dest_dir.
    If dest_dir does not exist, it is created.
    """
    # Verify source ZIP exists
    if not os.path.isfile(src_path):
        print(
            f"Error: archive '{src_path}' does not exist or is not a file",
            file=sys.stderr,
        )
        sys.exit(1)

    # Create destination directory if needed
    if not os.path.isdir(dest_dir):
        try:
            os.makedirs(dest_dir, exist_ok=True)
        except Exception as e:
            print(
                f"Error: failed to create directory '{dest_dir}': {e}", file=sys.stderr
            )
            sys.exit(1)

    # Extract all contents
    try:
        with zipfile.ZipFile(src_path, "r") as zf:
            zf.extractall(dest_dir)
    except zipfile.BadZipFile:
        print(f"Error: '{src_path}' is not a valid ZIP archive", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: failed to unpack '{src_path}': {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Unpacked '{src_path}' → '{dest_dir}'")


def main():
    parser = argparse.ArgumentParser(
        prog="python -m graph_net.unpack",
        description="Unpack a ZIP archive into a specified directory",
    )
    parser.add_argument(
        "--src", required=True, help="Path to the ZIP archive to unpack"
    )
    parser.add_argument(
        "--dst",
        required=True,
        help="Destination directory where files will be unpacked",
    )

    args = parser.parse_args()

    unpack_archive(args.src, args.dst)


if __name__ == "__main__":
    main()
