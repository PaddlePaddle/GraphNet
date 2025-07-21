import os
import sys
import zipfile
import argparse
import shutil


def pack_directory(src_dir: str, output_path: str):
    """
    Pack all files and subdirectories under src_dir into a ZIP file at output_path.
    """
    src_dir = os.path.normpath(src_dir)
    with zipfile.ZipFile(output_path, 'w', compression=zipfile.ZIP_DEFLATED, allowZip64=True) as zf:
        for root, _, files in os.walk(src_dir):
            for fname in files:
                full_path = os.path.join(root, fname)
                arcname = os.path.relpath(full_path, start=src_dir)
                zf.write(full_path, arcname=arcname)
    print(f'Packed "{src_dir}" → "{output_path}"')


def clear_directory(src_dir: str):
    """
    Remove all files and subdirectories under src_dir, preserving the directory itself.
    """
    for entry in os.listdir(src_dir):
        path = os.path.join(src_dir, entry)
        try:
            if os.path.isfile(path) or os.path.islink(path):
                os.unlink(path)
            elif os.path.isdir(path):
                shutil.rmtree(path)
        except Exception as e:
            print(f'Failed to remove {path}: {e}', file=sys.stderr)
    print(f'Cleared contents of "{src_dir}"')


def parse_boolean_flag(value: str) -> bool:
    """
    Parse a string to boolean, accepting 'True' or 'False' (case-insensitive).
    """
    lower = value.lower()
    if lower == 'true':
        return True
    if lower == 'false':
        return False
    raise argparse.ArgumentTypeError(f"clear_after_pack must be 'True' or 'False', got '{value}'")


def main():
    parser = argparse.ArgumentParser(
        prog='python -m graph_net.pack',
        description='Pack the $GRAPH_NET_EXTRACT_WORKSPACE directory into ZIP (clear_after_pack is required)'
    )
    parser.add_argument(
        '--output',
        metavar='OUTPUT_PATH',
        help='Specify the output ZIP file path (default is <workspace>.zip)',
    )
    parser.add_argument(
        '--clear-after-pack',
        dest='clear_after_pack',
        required=True,
        type=parse_boolean_flag,
        help="Specify whether to clear workspace after packing: 'True' or 'False'"
    )

    args = parser.parse_args()

    ws = os.environ.get('GRAPH_NET_EXTRACT_WORKSPACE')
    if not ws:
        parser.error('Environment variable GRAPH_NET_EXTRACT_WORKSPACE is not set')
    if not os.path.isdir(ws):
        parser.error(f'The path specified by GRAPH_NET_EXTRACT_WORKSPACE ("{ws}") is not a valid directory')

    # Determine output path
    if args.output:
        output_path = args.output
    else:
        base = os.path.basename(ws.rstrip(os.sep)) or 'workspace'
        output_path = f"{base}.zip"

    # Perform pack
    pack_directory(ws, output_path)

    # Optionally clear after pack
    if args.clear_after_pack:
        clear_directory(ws)


if __name__ == '__main__':
    main()