import argparse
import os
from tools.torch_to_paddle.utils import (
    get_model_path_list,
    get_convert_log,
    filter_and_save_unconverted_api,
    save_sample_api_convert_rate,
)
from tools.torch_to_paddle.file_processors import (
    convert_api_from_unstable_to_stable,
    convert_sample_from_torch_to_paddle,
)


def main(args):
    # Convert samples from torch to paddle.
    model_path_prefix = args.model_path_prefix
    model_path_list = list(get_model_path_list(args.model_path_list))
    output_dir = args.output_dir
    log_dir = args.log_dir
    summary_dir = output_dir

    for model_path in model_path_list:
        abs_model_path = os.path.join(model_path_prefix, model_path)
        abs_output_dir = os.path.join(output_dir, model_path.split("samples/", 1)[-1])
        abs_log_dir = os.path.join(log_dir, model_path)
        get_convert_log(abs_model_path, abs_log_dir)
        filter_and_save_unconverted_api(model_path, abs_log_dir, summary_dir)
        convert_api_from_unstable_to_stable(
            model_path, abs_model_path, abs_output_dir, summary_dir
        )
        convert_sample_from_torch_to_paddle(abs_model_path, abs_output_dir, abs_log_dir)
        save_sample_api_convert_rate(model_path, abs_log_dir, summary_dir)
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert samples from torch to paddle."
    )
    parser.add_argument(
        "--model-path-list",
        type=str,
        required=False,
        default=None,
        help="Path of file containing model paths.",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        required=False,
        default=None,
        help="Output directory of samples from torch to paddle.",
    )

    parser.add_argument(
        "--model-path-prefix",
        type=str,
        required=False,
        default=None,
        help="Path prefix of samples in list of model path.",
    )

    parser.add_argument(
        "--log-dir",
        type=str,
        required=False,
        default=None,
        help="Log directory of convert samples from torch to paddle.",
    )

    args = parser.parse_args()
    main(args=args)
