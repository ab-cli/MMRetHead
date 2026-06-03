import torch


def load_LLM(args):
    kwargs = {}
    lower_model_name = args.model_name_or_path.lower().split("/")[-1]

    if "qwen2.5-vl" in lower_model_name or "qvq-" in lower_model_name:
        from .qwen2_5_vl import Qwen2VLModel
        model_cls = Qwen2VLModel
    elif "qwen3-vl" in lower_model_name:
        from .qwen3_vl import Qwen3VLModel
        model_cls = Qwen3VLModel
    elif "gemma-3" in lower_model_name:
        from .gemma3 import Gemma3VLModel
        model_cls = Gemma3VLModel
    else:
        raise ValueError(f"Unsupported VLM model: {args.model_name_or_path}")

    if args.no_torch_compile:
        kwargs["torch_compile"] = False
    if args.no_bf16:
        kwargs["torch_dtype"] = torch.float16
    if args.rope_theta is not None:
        kwargs["rope_theta"] = args.rope_theta
    if args.use_yarn:
        kwargs["use_yarn"] = True
    if args.do_prefill:
        kwargs["do_prefill"] = args.do_prefill
    if args.attn_implementation is not None:
        kwargs["attn_implementation"] = args.attn_implementation

    if args.image_resize:
        kwargs["image_resize"] = args.image_resize
    if args.max_image_num is not None:
        kwargs["max_image_num"] = args.max_image_num
    if args.max_image_size is not None:
        kwargs["max_image_size"] = args.max_image_size
    kwargs["add_null_score"] = args.add_null_score

    model = model_cls(
        args.model_name_or_path,
        temperature=args.temperature,
        top_p=args.top_p,
        max_length=args.input_max_length,
        generation_max_length=args.generation_max_length,
        generation_min_length=args.generation_min_length,
        do_sample=args.do_sample,
        stop_newline=args.stop_newline,
        use_chat_template=args.use_chat_template,
        **kwargs,
    )

    return model
