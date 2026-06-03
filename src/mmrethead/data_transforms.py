import copy
import random
import warnings


_WARNED_UNACHIEVABLE_IMAGE_RATIO_CONFIGS = set()


def build_haystack_variant_config(args):
    variant_name = getattr(args, "haystack_variant", "none")
    if variant_name in [None, "none"]:
        return None

    variant_cfg = {
        "name": variant_name,
        "image_ratio": getattr(args, "haystack_image_ratio", None),
        "distractor_budget": getattr(args, "haystack_distractor_budget", None),
        "text_segment_token_budget": getattr(args, "haystack_text_segment_token_budget", None),
        "seed": getattr(args, "haystack_seed", None),
    }

    if variant_cfg["name"] != "image_ratio":
        raise ValueError(f"Unsupported haystack variant {variant_cfg['name']}")

    if variant_cfg["image_ratio"] is None:
        raise ValueError("--haystack_image_ratio is required when --haystack_variant=image_ratio")
    if not 0.0 <= variant_cfg["image_ratio"] <= 1.0:
        raise ValueError("--haystack_image_ratio must be between 0 and 1")

    variant_cfg["requested_ratio_name"] = variant_cfg["name"]
    variant_cfg["requested_ratio"] = float(variant_cfg["image_ratio"])
    variant_cfg["target_text_ratio"] = 1.0 - float(variant_cfg["image_ratio"])

    if variant_cfg["distractor_budget"] is None:
        raise ValueError("--haystack_distractor_budget is required when --haystack_variant is a ratio transform")

    if variant_cfg["distractor_budget"] < 0:
        raise ValueError("--haystack_distractor_budget must be non-negative")

    if variant_cfg["text_segment_token_budget"] is not None and variant_cfg["text_segment_token_budget"] <= 0:
        raise ValueError("--haystack_text_segment_token_budget must be positive when provided")

    return variant_cfg


def haystack_variant_suffix(args):
    variant_name = getattr(args, "haystack_variant", "none")
    if variant_name in [None, "none"]:
        return ""

    suffix = f"_haystack-{variant_name}"

    image_ratio = getattr(args, "haystack_image_ratio", None)
    if variant_name == "image_ratio" and image_ratio is not None:
        ratio_tag = str(int(round(float(image_ratio) * 1000))).zfill(3)
        suffix += f"-image{ratio_tag}"

    distractor_budget = getattr(args, "haystack_distractor_budget", None)
    if distractor_budget is not None:
        suffix += f"-budget{distractor_budget}"

    text_segment_token_budget = getattr(args, "haystack_text_segment_token_budget", None)
    if text_segment_token_budget is not None:
        suffix += f"-textseg{text_segment_token_budget}"

    return suffix


def apply_haystack_composition_variant(sample, variant_cfg):
    if variant_cfg is None:
        return sample

    if "ctxs" not in sample or "image_list" not in sample:
        return sample

    if variant_cfg["name"] == "image_ratio":
        return apply_mm_niah_ratio_variant(sample, variant_cfg)

    raise ValueError(f"Unsupported haystack variant {variant_cfg['name']}")


def warn_unachievable_ratio(
    sample_key,
    ratio_name,
    requested_ratio,
    distractor_budget,
    available_text_count,
    available_image_count,
    achieved_text_count,
    achieved_image_count,
):
    """Warn once per unique pool/budget/ratio signature when clipping changes the requested mix."""
    achieved_total = max(achieved_text_count + achieved_image_count, 1)
    achieved_ratio = achieved_image_count / achieved_total
    warn_key = (
        ratio_name,
        round(float(requested_ratio), 6),
        distractor_budget,
        available_text_count,
        available_image_count,
        achieved_text_count,
        achieved_image_count,
    )
    if warn_key in _WARNED_UNACHIEVABLE_IMAGE_RATIO_CONFIGS:
        return

    _WARNED_UNACHIEVABLE_IMAGE_RATIO_CONFIGS.add(warn_key)
    warnings.warn(
        f"Requested haystack {ratio_name} is not achievable for this sample pool. "
        f"sample={sample_key}, requested_ratio={requested_ratio:.3f}, "
        f"budget={distractor_budget}, available_text={available_text_count}, "
        f"available_image={available_image_count}, achieved_text={achieved_text_count}, "
        f"achieved_image={achieved_image_count}, achieved_ratio={achieved_ratio:.3f}",
        stacklevel=2,
    )


def estimate_text_tokens(text):
    # The tokenizer is not available in this raw-data transform; char/4 is only a stable sizing heuristic.
    return max(1, round(len(text) / 4))


def build_text_distractor_units(ctxs, text_indices, target_tokens):
    if target_tokens is None:
        return [
            {
                "sort_idx": idx,
                "indices": [idx],
                "entry": ctxs[idx],
            }
            for idx in text_indices
        ]

    units = []
    current_indices = []
    current_texts = []
    current_tokens = 0

    def flush():
        nonlocal current_indices, current_texts, current_tokens
        if not current_indices:
            return
        merged_entry = copy.deepcopy(ctxs[current_indices[0]])
        merged_entry["type"] = "text"
        merged_entry["text"] = " ".join(text for text in current_texts if text).strip()
        units.append(
            {
                "sort_idx": current_indices[0],
                "indices": list(current_indices),
                "entry": merged_entry,
            }
        )
        current_indices = []
        current_texts = []
        current_tokens = 0

    for idx in text_indices:
        text = ctxs[idx].get("text", "")
        current_indices.append(idx)
        current_texts.append(text)
        current_tokens += estimate_text_tokens(text)
        if current_tokens >= target_tokens:
            flush()

    flush()
    return units


def apply_mm_niah_ratio_variant(sample, variant_cfg):
    sample = copy.deepcopy(sample)
    ctxs = sample.get("ctxs", [])
    image_list = list(sample.get("image_list", []))

    if not ctxs:
        return sample

    image_paths_by_ctx_idx = {}
    image_ptr = 0
    for idx, entry in enumerate(ctxs):
        if entry.get("type") == "image":
            if image_ptr >= len(image_list):
                raise ValueError("image_list is shorter than the number of image entries in ctxs")
            image_paths_by_ctx_idx[idx] = image_list[image_ptr]
            image_ptr += 1

    if image_ptr != len(image_list):
        raise ValueError("image_list is longer than the number of image entries in ctxs")

    distractor_text_indices = []
    distractor_image_indices = []
    preserved_indices = set()

    for idx, entry in enumerate(ctxs):
        if entry.get("has_answer") or entry.get("has_ans"):
            preserved_indices.add(idx)
            continue

        if entry.get("type") == "image":
            distractor_image_indices.append(idx)
        else:
            distractor_text_indices.append(idx)

    text_units = build_text_distractor_units(
        ctxs,
        distractor_text_indices,
        variant_cfg.get("text_segment_token_budget"),
    )

    total_distractors = len(text_units) + len(distractor_image_indices)
    if total_distractors == 0:
        return sample

    sample_key = sample.get("id", sample.get("idx", sample.get("question", "")))
    distractor_budget = min(variant_cfg["distractor_budget"], total_distractors)
    requested_ratio_name = variant_cfg["requested_ratio_name"]
    requested_ratio = variant_cfg["requested_ratio"]
    text_ratio = variant_cfg["target_text_ratio"]

    if distractor_budget == total_distractors:
        achieved_text_count = len(text_units)
        achieved_image_count = len(distractor_image_indices)
        achieved_ratio = achieved_image_count / max(distractor_budget, 1)
        if abs(achieved_ratio - requested_ratio) > 1e-9:
            warn_unachievable_ratio(
                sample_key=sample_key,
                ratio_name=requested_ratio_name,
                requested_ratio=requested_ratio,
                distractor_budget=distractor_budget,
                available_text_count=len(text_units),
                available_image_count=len(distractor_image_indices),
                achieved_text_count=achieved_text_count,
                achieved_image_count=achieved_image_count,
            )
        kept_text_unit_indices = set(range(len(text_units)))
        kept_image_indices = set(distractor_image_indices)
    else:
        desired_text_count = int(round(distractor_budget * text_ratio))
        desired_image_count = distractor_budget - desired_text_count

        keep_text_count = min(desired_text_count, len(text_units))
        keep_image_count = min(desired_image_count, len(distractor_image_indices))

        remaining_budget = distractor_budget - keep_text_count - keep_image_count
        remaining_text_capacity = len(text_units) - keep_text_count
        remaining_image_capacity = len(distractor_image_indices) - keep_image_count

        while remaining_budget > 0:
            if remaining_text_capacity >= remaining_image_capacity and remaining_text_capacity > 0:
                keep_text_count += 1
                remaining_text_capacity -= 1
            elif remaining_image_capacity > 0:
                keep_image_count += 1
                remaining_image_capacity -= 1
            else:
                break
            remaining_budget -= 1

        if keep_text_count != desired_text_count or keep_image_count != desired_image_count:
            warn_unachievable_ratio(
                sample_key=sample_key,
                ratio_name=requested_ratio_name,
                requested_ratio=requested_ratio,
                distractor_budget=distractor_budget,
                available_text_count=len(text_units),
                available_image_count=len(distractor_image_indices),
                achieved_text_count=keep_text_count,
                achieved_image_count=keep_image_count,
            )

        rng = random.Random(
            f"{variant_cfg.get('seed', 42)}:{sample_key}:{requested_ratio_name}:{requested_ratio}:{distractor_budget}"
        )

        kept_text_unit_indices = set(rng.sample(range(len(text_units)), keep_text_count)) if keep_text_count > 0 else set()
        kept_image_indices = set(rng.sample(distractor_image_indices, keep_image_count)) if keep_image_count > 0 else set()

    kept_units = []
    for idx in preserved_indices:
        kept_units.append(
            {
                "sort_idx": idx,
                "indices": [idx],
                "entry": ctxs[idx],
            }
        )
    for unit_idx in kept_text_unit_indices:
        kept_units.append(text_units[unit_idx])
    for idx in kept_image_indices:
        kept_units.append(
            {
                "sort_idx": idx,
                "indices": [idx],
                "entry": ctxs[idx],
            }
        )

    new_ctxs = []
    new_image_list = []
    for unit in sorted(kept_units, key=lambda item: item["sort_idx"]):
        new_ctxs.append(unit["entry"])
        for idx in unit["indices"]:
            if idx in image_paths_by_ctx_idx:
                new_image_list.append(image_paths_by_ctx_idx[idx])

    sample["ctxs"] = new_ctxs
    sample["image_list"] = new_image_list
    return sample
