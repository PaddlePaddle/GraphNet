"""
Command-line tool for MultiDtypeGenerator.

Usage:
    python -m graph_net.torch.multi_dtype_generate \
        --source-sample-path /path/to/source/sample \
        --output-base-path /path/to/output \
        --dtype-list float16 bfloat16
"""

import argparse
import sys
from pathlib import Path
from graph_net.torch.multi_dtype_generator import MultiDtypeGenerator


def main():
    parser = argparse.ArgumentParser(
        description='Generate low-precision samples from existing float32 samples'
    )
    parser.add_argument(
        '--source-sample-path',
        type=str,
        required=True,
        help='Path to the source sample directory'
    )
    parser.add_argument(
        '--output-base-path',
        type=str,
        required=True,
        help='Base path for output samples'
    )
    parser.add_argument(
        '--dtype-list',
        type=str,
        nargs='+',
        default=['float16', 'bfloat16'],
        choices=['float16', 'bfloat16', 'float8'],
        help='List of target dtypes to generate (default: float16 bfloat16)'
    )
    parser.add_argument(
        '--filter-config',
        type=str,
        default=None,
        help='Path to filter configuration file (JSON format)'
    )
    
    args = parser.parse_args()
    
    # Load filter config if provided
    filter_config = {}
    if args.filter_config:
        import json
        with open(args.filter_config, 'r') as f:
            filter_config = json.load(f)
    
    # Create generator
    generator = MultiDtypeGenerator(
        source_sample_path=args.source_sample_path,
        output_base_path=args.output_base_path,
        dtype_list=args.dtype_list,
        filter_config=filter_config,
    )
    
    # Generate samples
    print(f"Generating samples from: {args.source_sample_path}")
    print(f"Output base path: {args.output_base_path}")
    print(f"Target dtypes: {args.dtype_list}")
    
    generated_paths = generator.generate()
    
    if generated_paths:
        print(f"\nSuccessfully generated {len(generated_paths)} sample(s):")
        for path in generated_paths:
            print(f"  - {path}")
        return 0
    else:
        print("\nNo samples were generated successfully.")
        return 1


if __name__ == '__main__':
    sys.exit(main())

