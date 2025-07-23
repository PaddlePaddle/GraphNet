# graph_net/config.py

import os
import sys
import json
import argparse


def load_env_workspace():
    ws = os.environ.get('GRAPH_NET_EXTRACT_WORKSPACE')
    if not ws:
        print('Error: GRAPH_NET_EXTRACT_WORKSPACE is not set', file=sys.stderr)
        sys.exit(1)
    return ws


def write_config(path: str, data: dict):
    # Ensure parent dir exists
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)
    print(f'Config written to {path}')


def main():
    parser = argparse.ArgumentParser(
        prog='python -m graph_net.config',
        description='Set user config for graph_net'
    )
    parser.add_argument('--global', dest='global_', action='store_true',
                        help='Write config globally to ~/.graph_net/config.json')
    parser.add_argument('--username', required=True,
                        help='Username to set in config')
    parser.add_argument('--email', required=True,
                        help='Email to set in config')

    args = parser.parse_args()

    # Prepare config data
    config_data = {
        'username': args.username,
        'email': args.email
    }

    if args.global_:
        # global config
        home = os.path.expanduser('~')
        config_path = os.path.join(home, '.graph_net', 'config.json')
    else:
        # workspace config
        ws = load_env_workspace()
        config_path = os.path.join(ws, 'config.json')

    write_config(config_path, config_data)


if __name__ == '__main__':
    main()
