import os
from itertools import product
from transformers import set_seed
from collections import defaultdict
import json
import time

from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import DataLoader

from .arguments import parse_arguments
from .data_transforms import haystack_variant_suffix
from .vlm_model import load_LLM

from .data import (
    load_data, 
    TestItemDataset,
)

import logging
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def run_test(args, model, dataset, test_file):
    logger.info(f"running test on {dataset} with test {test_file}")

    test_name = os.path.splitext(os.path.basename(test_file))[0]
    variant_suffix = haystack_variant_suffix(args)
    output_path = os.path.join(args.output_dir, f"{dataset}_{test_name}_in{args.input_max_length}_size{args.max_test_samples}_samp{args.do_sample}_addnull{args.add_null_score}{variant_suffix}_{args.seed}.json")
    print("output path:", output_path)
    if os.path.exists(output_path) and not args.overwrite and not args.debug:
        logger.info(f"{output_path} already exists, skipping...")
        return output_path

    set_seed(args.seed)
    data = load_data(args, dataset, test_file)

    if args.dry_run:
        logger.info(f"Dry run mode, loaded {len(data['data'])} samples from {dataset}")
        return None
    else:
        logger.info(f"loaded {len(data['data'])} samples from {dataset}")

    dataloader = DataLoader(
        TestItemDataset(data, model, model.processor),
        batch_size=1, 
        shuffle=False, 
        collate_fn=lambda x: x,
        num_workers=args.num_workers if not args.debug else 0,
    )

    metrics = defaultdict(list)
    instance_head_score_list = []
    activation_argmax_in_gold_list = []
    activation_argmax_token_index_list = []
    activation_argmax_attention_value_list = []
    activation_example_ids = []
    results = []
    start_time = time.time()
    with torch.inference_mode():
        for idx, inputs in enumerate(tqdm(dataloader)):
            test_item = data["data"][idx]
            inputs = inputs[0] # batch size is just 1
            if args.count_tokens:
                metrics["input_len"].append(inputs.input_ids.shape[1])
                continue

            output = model.get_attention_score(
                inputs=inputs,
                save_activation_frequency=args.save_activation_frequency,
            )
            if output is None:
                logger.info(f"skipping example {idx+1} because the model returned None")
                continue

            # in output, we get the (n_layers, n_heads) score tensors for every gold span
            curr_instance_head_score = output["gold_span_score_sum"] # (n_layers, n_heads)
            
            instance_head_score_list.append(curr_instance_head_score)

            if args.save_activation_frequency and "argmax_in_gold" in output:
                activation_argmax_in_gold_list.append(output["argmax_in_gold"].detach().cpu())
                activation_argmax_token_index_list.append(output["argmax_token_index"].detach().cpu())
                activation_argmax_attention_value_list.append(output["argmax_attention_value"].detach().cpu())
                activation_example_ids.append(str(test_item.get("id", idx)))

            metrics["input_len"].append(output["input_len"])

            result = {**test_item, **output}
            result.pop("gold_span_score_sum", None)
            result.pop("argmax_in_gold", None)
            result.pop("argmax_token_index", None)
            result.pop("argmax_attention_value", None)
            result.pop("doc_span", None)
            result.pop("context", None)
            result.pop("input_ids", None)
            input_text = result['input_text']
            results.append(result)

            # print out some examples, we also limit how much we print out since it can get really long
            if idx < 2 or args.debug:
                logger.info(f"Example {idx+1}: ")
                logger.info(f"Decoder inputs:\n{input_text}\n")

                logger.info(f"Input length: {output['input_len']}")
                # currently we hardcode somethings to print out, but you may change these to print out other things
                logger.info(f"Question: {test_item['question'] if 'question' in test_item else ''}")
                logger.info(f"Answer: {test_item['answer'] if 'answer' in test_item else ''}")
            
            if args.debug:
                import pdb; pdb.set_trace()

            output = None

    end_time = time.time()
    mem_usage = sum([torch.cuda.max_memory_allocated(i) for i in range(torch.cuda.device_count())])
    logger.info(f"Memory usage: {mem_usage/1000**3:.02f} GB")
    logger.info(f"Throughput: {len(results) / (end_time - start_time):.02f} samples/s")

    if args.count_tokens:
        logger.info(f"----{dataset}----\nAverage input length: {np.mean(metrics['input_len']):.02f}, std input length: {np.std(metrics['input_len']):.02f}, max input length: {max(metrics['input_len'])}, min input length: {min(metrics['input_len'])}\n----returning----")
        return output_path

    if len(results) == 0:
        logger.error("No results to evaluate, something went wrong, returning...")
        return output_path

    # step 2: average the instance head scores over all samples
    averaged_metrics = {k: np.mean(v)*(100 if "_len" not in k else 1) for k, v in metrics.items()}
    avg_head_score = torch.stack(instance_head_score_list, dim=0)
    avg_head_score = avg_head_score.mean(dim=0) # (n_layers, n_heads)

    # save the heads in list format
    avg_head_score = avg_head_score.tolist()
    head_score_list = [
        (f"{layer}-{head}", score)
        for layer, row in enumerate(avg_head_score)
        for head, score in enumerate(row)
    ]

    head_score_list = sorted(head_score_list, key=lambda x: x[1], reverse=True)

    logger.info("Averaged metrics:")
    for k, v in averaged_metrics.items():
        logger.info(f"{k}: {v:.02f}")
    logger.info(f"Averaged instance head scores: {head_score_list}")

    activation_frequency_path = None
    activation_frequency_head_list = None
    if args.save_activation_frequency and activation_argmax_in_gold_list:
        activation_hits = torch.stack(activation_argmax_in_gold_list, dim=0)
        activation_token_indexes = torch.stack(activation_argmax_token_index_list, dim=0).to(torch.int32)
        activation_attention_values = torch.stack(activation_argmax_attention_value_list, dim=0).to(torch.float16)
        activation_frequency = activation_hits.float().mean(dim=0)
        activation_frequency_values = activation_frequency.tolist()
        activation_frequency_head_list = [
            (f"{layer}-{head}", score)
            for layer, row in enumerate(activation_frequency_values)
            for head, score in enumerate(row)
        ]
        activation_frequency_head_list = sorted(activation_frequency_head_list, key=lambda x: x[1], reverse=True)
        activation_frequency_path = os.path.splitext(output_path)[0] + "_activation_frequency.npz"
        np.savez_compressed(
            activation_frequency_path,
            argmax_in_gold=activation_hits.numpy(),
            argmax_token_index=activation_token_indexes.numpy(),
            argmax_attention_value=activation_attention_values.numpy(),
            activation_frequency=activation_frequency.numpy(),
            example_ids=np.array(activation_example_ids, dtype=str),
        )
        logger.info(f"Activation frequency sidecar written to {activation_frequency_path}")

    output = {
        "args": args.__dict__,
        "head_score_list": head_score_list,
        "activation_frequency_head_list": activation_frequency_head_list,
        "activation_frequency_path": activation_frequency_path,
        "activation_frequency_definition": "fraction of examples where at least one query token's max-attended source token for a head is inside any gold span",
        "data": results,
        "metrics": metrics,
        "averaged_metrics": averaged_metrics,
        "memory_usage": mem_usage,
        "throughput": len(results) / (end_time - start_time),
    }

    if args.output_dir is not None:
        with open(output_path, "w") as f:
            json.dump(output, f, indent=4)
        logger.info(f"done, results are written to {output_path}")

    return output_path


def main():
    args = parse_arguments()

    logger.info(f"Arguments: {args}")
    assert args.model_name_or_path is not None
    os.makedirs(args.output_dir, exist_ok=True)

    if not args.do_sample:
        if args.temperature != 0.0:
            logger.warning("do_sample is set to false but temperature is not 0, do_sample will overwrite temperature")

    model = load_LLM(args)

    datasets = args.datasets.split(",")
    test_files = args.test_files.split(",")
    max_lengths = ([int(args.input_max_length)] * len(datasets)) if isinstance(args.input_max_length, int) or len(args.input_max_length.split(",")) == 1 else [int(l) for l in args.input_max_length.split(",")]
    gen_lengths = ([int(args.generation_max_length)] * len(datasets)) if isinstance(args.generation_max_length, int) or len(args.generation_max_length.split(",")) == 1 else [int(l) for l in args.generation_max_length.split(",")]
    assert len(test_files) == len(max_lengths)
    test_length_list = [int(l) * 1024 for l in args.test_length.split(",")]

    for dataset, test_file, max_length, gen_length in zip(datasets, test_files, max_lengths, gen_lengths):
        if max_length not in test_length_list:
            continue
        args.datasets = dataset
        args.test_files = test_file
        args.input_max_length = max_length
        args.generation_max_length = gen_length
        model.max_length = max_length
        model.generation_max_length = gen_length

        try: 
            run_test(args, model, dataset, test_file)
        except Exception as e:
            # in case we run into some kind of error 
            logger.exception(e)
            logger.error(f"Error in {dataset}, continuing...")
            if args.debug:
                raise e

if __name__ == "__main__":
    main()
