# graph_net/pack.py

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


def main():
    parser = argparse.ArgumentParser(
        prog='python -m graph_net.pack',
        description='Pack or clear the directory defined by $GRAPH_NET_EXTRACT_WORKSPACE'
    )
    parser.add_argument(
        '--clear',
        action='store_true',
        help='Clear all contents of the $GRAPH_NET_EXTRACT_WORKSPACE directory'
    )
    parser.add_argument(
        '--output',
        metavar='OUTPUT_PATH',
        help='Specify the output ZIP file path (default is <workspace>.zip)'
    )

    args = parser.parse_args()

    ws = os.environ.get('GRAPH_NET_EXTRACT_WORKSPACE')
    if not ws:
        parser.error('Environment variable GRAPH_NET_EXTRACT_WORKSPACE is not set')
    if not os.path.isdir(ws):
        parser.error(f'The path specified by GRAPH_NET_EXTRACT_WORKSPACE ("{ws}") is not a valid directory')

    # If --clear flag is provided, perform the clear operation and exit
    if args.clear:
        clear_directory(ws)
        return

    # Pack operation
    # Default output name: <workspace>.zip in current working directory
    if args.output:
        output_path = args.output
    else:
        base = os.path.basename(ws.rstrip(os.sep)) or 'workspace'
        output_path = f"{base}.zip"

    pack_directory(ws, output_path)


if __name__ == '__main__':
    main()
