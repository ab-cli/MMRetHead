import argparse
import yaml
import ast
import os
import re


DEFAULT_MMLONGBENCH_ROOT = os.path.join("data", "MMLongBench")
DEFAULT_MMLONGBENCH_DATA_ROOT = os.path.join(DEFAULT_MMLONGBENCH_ROOT, "mmlb_data")
DEFAULT_MMLONGBENCH_IMAGE_ROOT = os.path.join(DEFAULT_MMLONGBENCH_ROOT, "mmlb_image")


def default_detection_output_dir(model_name_or_path):
    model_name = os.path.basename(str(model_name_or_path or "model").rstrip("/")) or "model"
    model_name = re.sub(r"[^A-Za-z0-9._-]+", "_", model_name).strip("._-") or "model"
    return os.path.join("results", "detection", model_name)

def parse_arguments():
    parser = argparse.ArgumentParser(description="evaluation on downstream tasks")
    parser.add_argument("--config", type=str, default=None, help="path to config file")

    # model setting
    parser.add_argument("--model_name_or_path", type=str, default=None)
    parser.add_argument("--attn_implementation", type=str, default=None, help="Implementation of self-attention. None means using the default (flash_attention_2 for most models).")

    # data paths
    parser.add_argument("--datasets", type=str, default=None)
    parser.add_argument("--test_file_root", type=str, default=None,
                        help=f"MM-NIAH data root. Defaults to {DEFAULT_MMLONGBENCH_DATA_ROOT}.")
    parser.add_argument("--image_file_root", type=str, default=None,
                        help=f"MM-NIAH image root. Defaults to {DEFAULT_MMLONGBENCH_IMAGE_ROOT}.")
    parser.add_argument("--test_files", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None, help="path to save the predictions")
    parser.add_argument("--overwrite", action="store_true", help="whether to the saved file")
    parser.add_argument("--max_test_samples", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=32)
    parser.add_argument("--preprocessing_num_workers", type=int, default=8)
    parser.add_argument("--haystack_variant", type=str, choices=["none", "image_ratio"], default="none",
                        help="Optional raw-sample haystack transform applied before MM-NIAH loading.")
    parser.add_argument("--haystack_image_ratio", type=float, default=None,
                        help="Target image fraction among kept MM-NIAH distractors when haystack_variant=image_ratio.")
    parser.add_argument("--haystack_distractor_budget", type=int, default=None,
                        help="Optional number of distractor ctx items to keep after haystack transformation.")
    parser.add_argument("--haystack_text_segment_token_budget", type=int, default=None,
                        help="Approximate token budget for grouped MM-NIAH text distractor segments.")
    parser.add_argument("--haystack_seed", type=int, default=42,
                        help="Sampling seed for haystack composition transforms.")

    # evaluation settings
    parser.add_argument("--input_max_length", type=str, default='8192', help="the maximum number of tokens of the input, we truncate the end of the context; can be separated by comma to match the specified datasets")
    parser.add_argument("--test_length", type=str, default="4,8,16,32,64,128", help="list the length to be tested.")

    # generation settings
    parser.add_argument("--do_sample", type=ast.literal_eval, choices=[True, False], default=False, help="whether to use sampling (false is greedy), overwrites temperature")
    parser.add_argument("--generation_max_length", type=str, default='10', help="max number of tokens to generate, can be separated by comma to match the specified datasets")
    parser.add_argument("--generation_min_length", type=int, default=0, help="min number of tokens to generate")
    parser.add_argument("--temperature", type=float, default=1.0, help="generation temperature")
    parser.add_argument("--top_p", type=float, default=1.0, help="top-p parameter for nucleus sampling")
    parser.add_argument("--stop_newline", type=ast.literal_eval, choices=[True, False], default=False, help="whether to stop generation at newline")
    parser.add_argument("--do_prefill", action="store_true", help="prefill the context to save memory")

    # model specific settings
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--no_cuda", action="store_true", help="disable cuda")
    parser.add_argument("--no_bf16", action="store_true", help="disable bf16 and use fp32")
    parser.add_argument("--no_torch_compile", action="store_true", help="disable cuda")
    parser.add_argument("--use_chat_template", type=ast.literal_eval, choices=[True, False], default=True, help="whether to use chat template")
    parser.add_argument("--rope_theta", type=int, default=None, help="override rope theta")
    parser.add_argument("--use_yarn", action="store_true", help="yarn extension")
    parser.add_argument("--image_resize", type=float, default=None, help="Image scaling factor, where 1.0 means original size and 0.5 means half the original size")
    parser.add_argument("--max_image_num", type=int, default=None, help="the max image number for models with dynamic cropping (e.g., internvl1.5/2/2.5, phi3/3.5)")
    parser.add_argument("--vision_batch_size", type=int, default=None, help="the batch size for InternVL3's and Gemma3's vision tower since its implementation has O(N^2) memory cost (N is the image number)")
    parser.add_argument("--max_image_size", type=int, default=None, help="Max image size for Gemini to prevent over resizing and splitting")
    parser.add_argument("--add_null_score", type=ast.literal_eval, choices=[True, False], default=True, help="whether to add null score to the predictions")
    parser.add_argument("--save_activation_frequency", action="store_true", help="save query-token argmax-in-gold activation frequency sidecar")

    # debug parameters
    parser.add_argument("--debug", action="store_true", help="for debugging")
    parser.add_argument("--count_tokens", action="store_true", help="instead of running generation, just count the number of tokens (only for HF models not API)")
    parser.add_argument("--dry_run", action="store_true", help="Test the data loading speed.")

    args = parser.parse_args()
    config = yaml.safe_load(open(args.config)) if args.config is not None else {}
    parser.set_defaults(**config)
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = default_detection_output_dir(args.model_name_or_path)

    if args.test_file_root is None:
        args.test_file_root = DEFAULT_MMLONGBENCH_DATA_ROOT

    if args.image_file_root is None:
        args.image_file_root = DEFAULT_MMLONGBENCH_IMAGE_ROOT

    if args.rope_theta is not None:
        args.output_dir = args.output_dir + f"-override-rope{args.rope_theta}"

    return args
