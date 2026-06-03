import os 
import glob
import json
import hashlib
import re
import traceback
from pathlib import Path
import sys
from transformers import AutoProcessor, AutoConfig
import ast
import random

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from mmrethead.data import head_mask_load_image_task
from mmrethead.vlm_model.head_mask_qwen3_vl import Qwen3VLModel

import numpy as np
import argparse

random.seed(42)

DEFAULT_MMLONGBENCH_ROOT = REPO_ROOT / "data" / "MMLongBench"
DEFAULT_MMLONGBENCH_DATA_ROOT = DEFAULT_MMLONGBENCH_ROOT / "mmlb_data"
DEFAULT_MMLONGBENCH_IMAGE_ROOT = DEFAULT_MMLONGBENCH_ROOT / "mmlb_image"
DEFAULT_SAVE_PATH = "results/causal_masking"

#from openai import OpenAI
from datetime import datetime, timezone
from collections import defaultdict
import time
import torch

def reset_rope(model, model_max_train_len, scaling_factor):
    for l in model.model.layers:
        l.self_attn.rotary_emb.scaling_factor = scaling_factor
        l.self_attn.rotary_emb._set_cos_sin_cache(seq_len=model_max_train_len, device=l.self_attn.rotary_emb.inv_freq.device, dtype=torch.float32)
    return


def context_length_to_k(context_length):
    mapping = {
        8192: "8",
        16384: "16",
        32768: "32",
        65536: "64",
        131072: "128",
    }
    if context_length in mapping:
        return mapping[context_length]
    if context_length % 1024 == 0:
        return str(context_length // 1024)
    raise ValueError(f"Cannot infer MM-NIAH K value from context length {context_length}")


def default_task_data_path(dataset_name, max_context_len):
    name = (dataset_name or "mm_image").lower()
    if any(key in name for key in ["mm_image", "mm_niah_image", "retrieval-image", "identical"]):
        return DEFAULT_MMLONGBENCH_DATA_ROOT / "NIAH" / f"retrieval-image_test_K{context_length_to_k(max_context_len)}_dep6.jsonl"
    if any(key in name for key in ["viquae", "infoseek", "vrag"]):
        raise ValueError("--task_data_path is required for VRAG-style tasks.")
    raise ValueError(f"Cannot infer default task data path for dataset_name={dataset_name!r}")


def load_example_ids(path):
    with open(path, "r", encoding="utf-8") as handle:
        raw = handle.read()

    stripped = raw.strip()
    if not stripped:
        return []

    if stripped.startswith("["):
        values = json.loads(stripped)
        if not isinstance(values, list):
            raise ValueError(f"Expected a JSON list of example ids in {path}")
        return [str(value) for value in values]

    return [line.strip() for line in raw.splitlines() if line.strip()]


def safe_suffix(value):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def bounded_suffix(value, max_len):
    safe = safe_suffix(value).strip("._")
    if not safe:
        safe = "example"
    if len(safe) <= max_len:
        return safe
    digest = hashlib.sha1(value.encode("utf-8")).hexdigest()[:12]
    prefix_len = max(1, max_len - len(digest) - 1)
    prefix = safe[:prefix_len].rstrip("._")
    if not prefix:
        prefix = safe[:prefix_len]
    return f"{prefix}_{digest}"


def build_save_name(base_save_name, example_id, max_len=180):
    base = base_save_name.strip("._")
    if not base:
        base = "run"
    available = max_len - len(base) - 1
    if available < 16:
        digest = hashlib.sha1(base.encode("utf-8")).hexdigest()[:12]
        prefix_len = max(1, max_len - len(digest) - 17)
        base = f"{base[:prefix_len].rstrip('._')}_{digest}"
        available = max_len - len(base) - 1
    suffix = bounded_suffix(example_id, max(16, available))
    return f"{base}_{suffix}"

class LLMNeedleHaystackTester:
    """
    This class is used to test the LLM Needle Haystack.
    """
    def __init__(self,
                 task_data_path=None,
                 task_image_dir=None,
                 min_context_len = 1024,
                 max_context_len = 131072,
                 ctx_len_intervals = 40,
                 context_lengths = None,
                 document_depth_percent_min = 0,
                 document_depth_percent_max = 100,
                 document_depth_percent_intervals = 10,
                 document_depth_percent_interval_type = "linear",
                 mask_topk=0,
                 model_name_or_path='',
                 task_suffix=None,
                 save_results = True,
                 save_contexts = True,
                 context_buffer = 200,
                 print_ongoing_status = True,
                 head_score_path=None,
                 save_path=None,
                 dataset_name=None,
                 example_id=None,
                 example_idx=-1,
                 use_yarn=False,
                 attn_implementation=None,
                 vision_batch_size=None,
                 max_image_num=None,
                 mask_prefill=True,
                 ):
        """        
        :param haystack_dir: The directory of text files to use as background context (or a haystack) in which the needle is to be found. Default is Paul Graham Essays.
        :param save_results: Whether or not you would like to save your contexts to file. Warning: These will get long! Default = True
        :param save_contexts: Whether or not you would like to save your contexts to file. Warning: These will get long! Default is True.
        :param context_buffer: The amount of cushion you'd like to leave off the input context to allow for the output context. Default 200 tokens
        :param min_context_len: The minimum length of the context. Default is 1000.
        :param max_context_len: The maximum length of the context. Default is 200000.
        :param ctx_len_intervals: The number of intervals for the context length. Default is 35.
        :param document_depth_percent_min: The minimum depth percent of the document. Default is 0.
        :param document_depth_percent_max: The maximum depth percent of the document. Default is 100.
        :param document_depth_percent_intervals: The number of intervals for the document depth percent. Default is 35.
        :param document_depth_percent_interval_type: The type of interval for the document depth percent. Must be either 'linear' or 'sigmoid'. Default is 'linear'.
        :param model_name_or_path: The name of the model. Default is 'gpt-4-1106-preview'.
        :param print_ongoing_status: Whether or not to print the ongoing status. Default is True.
        """
        if not task_data_path or not task_image_dir:
            raise ValueError("data path and image dir must be provided.")
        
        self.task_data_path = task_data_path
        self.task_image_dir = task_image_dir
        self.save_results = save_results
        self.context_buffer = context_buffer
        self.save_contexts = save_contexts
        self.print_ongoing_status = print_ongoing_status
        self.testing_results = []
        self.mask_topk = mask_topk
        self.save_path=save_path
        self.min_context_len = min_context_len
        self.max_context_len = max_context_len
        self.dataset_name = dataset_name
        self.example_id = example_id
        self.example_idx = example_idx
        self.mask_prefill = mask_prefill

        if("/" in model_name_or_path):
            self.model_version = model_name_or_path.split("/")[-1]
        else: 
            self.model_version = model_name_or_path

        if task_suffix is not None: 
            self.model_version += "_" + task_suffix

        if min_context_len is None or max_context_len is None or ctx_len_intervals is None:
            raise ValueError("Either context_lengths_min, context_lengths_max, ctx_len_intervals need to be filled out OR the context_lengths_list needs to be supplied.")
        
        self.context_lengths = np.round(np.linspace(min_context_len, max_context_len, num=ctx_len_intervals + 1, endpoint=True)).astype(int)


        if document_depth_percent_interval_type not in [None, "linear", "sigmoid"]:
            raise ValueError("document_depth_percent_interval_type must be either None, 'linear' or 'sigmoid'. If you'd like your own distribution give a list of ints in via document_depth_percent_intervals")

        if document_depth_percent_min is None or document_depth_percent_max is None or document_depth_percent_intervals is None:
            raise ValueError("Either document_depth_percent_min, document_depth_percent_max, document_depth_percent_intervals need to be filled out OR the document_depth_percents needs to be supplied.")
        
        if document_depth_percent_interval_type == 'linear':
            self.document_depth_percents = np.round(np.linspace(document_depth_percent_min, document_depth_percent_max, num=document_depth_percent_intervals + 1, endpoint=True)).astype(int)
        elif document_depth_percent_interval_type == 'sigmoid':
            self.document_depth_percents = [self.logistic(x) for x in np.linspace(document_depth_percent_min, document_depth_percent_max, document_depth_percent_intervals)]
        
        self.model_name_or_path = model_name_or_path
        config = AutoConfig.from_pretrained(model_name_or_path)
        self.layer_num, self.head_num = config.text_config.num_hidden_layers, config.text_config.num_attention_heads
        print(f"layer number: {self.layer_num}, head number {self.head_num}")
        
        
        if "Qwen3-VL" in self.model_version:
            model_cls = Qwen3VLModel
        elif "Qwen2.5-VL" in self.model_version:
            from mmrethead.vlm_model.head_mask_qwen2_5_vl import Qwen2_5VLModel
            model_cls = Qwen2_5VLModel
        elif "gemma-3" in self.model_version.lower():
            from mmrethead.vlm_model.head_mask_gemma3 import Gemma3VLModel
            model_cls = Gemma3VLModel
        else:
            raise ValueError(f"model version {self.model_version} is not supported.")
        
        model_kwargs = {
            "use_yarn": use_yarn,
            "vision_batch_size": vision_batch_size,
            "max_image_num": max_image_num,
        }
        if attn_implementation is not None:
            model_kwargs["attn_implementation"] = attn_implementation
        self.model = model_cls(model_name_or_path, **model_kwargs)

        self.init_head_mask_with_save_name(head_score_path)
        self.load_data()

    def logistic(self, x, L=100, x0=50, k=.1):
        if x == 0:
            return 0
        if x == 100:
            return 100
        return np.round(L / (1 + np.exp(-k * (x - x0))), 3)

    def init_head_mask_with_save_name(self, head_score_path):
        self.block_list = []
        if self.mask_topk > 0:
            with open(head_score_path, "r") as file:
                stable_block_list = json.load(file)
                if isinstance(stable_block_list, dict):
                    stable_block_list = stable_block_list["head_score_list"]
                stable_block_list = sorted(stable_block_list, key=lambda x: x[1], reverse=True) 
            self.block_list = [[int(ll) for ll in l[0].split("-")] for l in stable_block_list]

        if self.mask_topk > 0:
            print(f"masking out top {self.mask_topk} retrieval heads")
        else:
            print(f"masking out random {-self.mask_topk}  heads")

        if self.mask_topk > 0:
            self.block_list = self.block_list[:self.mask_topk]
            self.save_name = f"{self.model_version}_mask_top{self.mask_topk}"
        elif self.mask_topk == 0:
            self.block_list = None
            self.save_name = self.model_version
        else:
            self.block_list = self.construct_random_head(-self.mask_topk)
            self.save_name = f"{self.model_version}_mask_random{-self.mask_topk}"

        self.save_name += f"_len{self.min_context_len // 1024}-{self.max_context_len // 1024}K"
        if self.mask_prefill:
            self.save_name += "_prefillmask"


    def generate(
        self,
        q_outputs,
        inp,
        decode_len,
        cache_position=None,
        block_list=None,
    ):
        output = []
        past_kv = q_outputs.past_key_values
        eos_token_id = self.model.tokenizer.eos_token_id
        for _ in range(decode_len):
            inp = inp.view(1, 1)
            outputs = self.model.model(input_ids=inp, past_key_values=past_kv, use_cache=True, \
                 output_attentions=False, cache_position=cache_position, block_list=block_list)
            past_kv = outputs.past_key_values
            inp = outputs.logits[0, -1].argmax()
            # step_token = self.model.tokenizer.convert_ids_to_tokens(inp.item())
            output.append(inp.item())
            cache_position += 1 
            if inp.item()==eos_token_id: break
            
        return output

    def decode_from_prefill(self, inputs, decode_len, block_list=None):
        prefill_inputs = {
            k: v[:, :-1] if k in ["input_ids", "attention_mask"] else v
            for k, v in inputs.items()
        }
        prefill_block_list = block_list if self.mask_prefill else None
        q_outputs = self.model.model(
            **prefill_inputs,
            use_cache=True,
            return_dict=True,
            logits_to_keep=1,
            block_list=prefill_block_list,
        )
        cache_position = torch.tensor([inputs["input_ids"].size(1) - 1], dtype=torch.int64, device=self.model.device)
        return self.generate(q_outputs, inputs["input_ids"][:, -1], decode_len, cache_position=cache_position, block_list=block_list)
    
    def construct_random_head(self, n):
        head_pool = [
            (l, h) for l in range(self.layer_num) 
                   for h in range(self.head_num)
        ]
        results = random.sample(head_pool, n)
        
        return results

    def run_test(self, args):
        # Run through each iteration of context_lengths and depths
        for context_length in self.context_lengths:
            for depth_percent in self.document_depth_percents:
                self.evaluate_and_log(context_length, depth_percent)

    def evaluate_and_log(self, context_length, depth_percent):
        content_with_needle = self.model.build_context(self.test_item, self.data, context_length, depth_percent, self.context_buffer)

        test_item_with_needle = self.test_item.copy()
        context = [c["text"] if c["type"] == "text" else "<image>" for c in content_with_needle]
        context = "".join(context)
        image_list = [c["image"] for c in content_with_needle if c["type"] == "image"]
        image_list += self.test_item["choices_image"]
        test_item_with_needle["context"] = context
        test_item_with_needle["image_list"] = image_list

        inputs = self.model.prepare_inputs(test_item_with_needle, self.data)
 
        test_start_time = time.time()
        inputs = inputs.to(self.model.device)

        with torch.no_grad():
            output = self.decode_from_prefill(
                inputs,
                50,
                block_list=self.block_list,
            )
            response = self.model.tokenizer.decode(output, skip_special_tokens=True).strip()

            # official_output_ids = self.model.model.generate(
            #                             **inputs, 
            #                             max_new_tokens=50, 
            #                             min_new_tokens=0,
            #                             pad_token_id=self.model.tokenizer.eos_token_id,
            #                             eos_token_id=self.model.tokenizer.eos_token_id,
            #                             use_cache=True,
            #                             return_dict_in_generate=True,
            #                             do_sample=False,
            #                         )
            # input_len = inputs["input_ids"].size(1)
            # official_response = self.model.tokenizer.decode(official_output_ids['sequences'][0, input_len:], skip_special_tokens=True)

        test_end_time = time.time()
        test_elapsed_time = test_end_time - test_start_time
        
        score = self.metric(self.test_item["answer"], response)
        results = {
            'model' : self.model_name_or_path,
            'context_length' : int(context_length),
            'depth_percent' : float(depth_percent),
            'needle' : self.test_item["needle"],
            'model_response' : response,
            'score' : score,
            'mask_prefill': self.mask_prefill,
            'test_duration_seconds' : test_elapsed_time,
            'test_timestamp_utc' : datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S%z')
        }

        self.testing_results.append(results)

        if self.print_ongoing_status:
            print (f"-- Test Summary -- ")
            print (f"Duration: {test_elapsed_time:.1f} seconds")
            print (f"Context: {context_length} tokens")
            print (f"Depth: {depth_percent}%")
            print (f"Score: {score}")
            print (f"Response: {response}\n")

        context_file_location = f'{self.model_version.replace(".", "_")}_len_{context_length}_depth_{int(depth_percent*100)}'
        
        if self.save_results:
            # Save the context to file for retesting
            if not os.path.exists(f'{self.save_path}/{self.save_name}'):
                os.makedirs(f'{self.save_path}/{self.save_name}')
            
    
            # Save the result to file for retesting
            p = f'{self.save_path}/{self.save_name}/{context_file_location}_results.json'
            print("Writing at %s" % p)
            with open(p, 'w') as f:
                json.dump(results, f)

    def load_data(self):
        max_context_length = max(self.context_lengths)

        result = head_mask_load_image_task(
            self.task_data_path,
            self.task_image_dir,
            dataset_name=self.dataset_name,
            example_idx=self.example_idx,
            example_id=self.example_id,
        )
        self.data = result
        self.test_item = self.data["data"][0]
        self.metric = self.data.pop("metric")

        self.test_item["content"] = self.model.precompute_content_segment_lengths(self.test_item)

        # if self.model.prepare_inputs(self.test_item, self.data)["input_ids"].size(1) < max_context_length:
        #   print("the context length is still not enough, please check the haystack_dir")
        
    def decode_with_length(self, tokens, context_length=None):
        return self.enc.decode(tokens[:context_length])
    
    def print_start_test_summary(self):
        print ("\n")
        print ("Starting Needle In A Haystack Testing...")
        print (f"- Model: {self.model_name_or_path}")
        print (f"- Context Lengths: {len(self.context_lengths)}, Min: {min(self.context_lengths)}, Max: {max(self.context_lengths)}")
        print (f"- Document Depths: {len(self.document_depth_percents)}, Min: {min(self.document_depth_percents)}%, Max: {max(self.document_depth_percents)}%")
        print (f"- Needle: {self.test_item['needle']}")
        print ("\n\n")

    def start_test(self, args):
        if self.print_ongoing_status:
            self.print_start_test_summary()
        self.run_test(args)


if __name__ == "__main__":
    # Tons of defaults set, check out the LLMNeedleHaystackTester's init for more info
    parser = argparse.ArgumentParser()
    parser.add_argument('--min_context_len', metavar='N', default=1024, type=int, help='a number')
    parser.add_argument('--max_context_len', metavar='N', default=32768, type=int, help='a number')
    parser.add_argument('--task_data_path', type=str, default=None, help='path to one task JSONL; defaults to repo-local MMLongBench MM-NIAH image file inferred from --max_context_len')
    parser.add_argument('--task_image_dir', type=str, default=str(DEFAULT_MMLONGBENCH_IMAGE_ROOT), help='path to task-style image directory')
    parser.add_argument('--dataset_name', type=str, default="mm_image", help='optional dataset name for loader dispatch')
    parser.add_argument('--example_id', type=str, default=None, help='optional exact example id for loader selection')
    parser.add_argument('--example_ids_file', type=str, default=None, help='optional JSON or newline-delimited list of example ids to run in-process')
    parser.add_argument('--example_idx', metavar='N', default=-1, type=int, help='example index for loader selection')
    parser.add_argument('--model_name_or_path', type=str, default=None, help='path to model')
    parser.add_argument('--task_suffix', type=str, default=None, help='the suffix of current task to distinguish different results')
    parser.add_argument('--ctx_len_intervals', type=int, default=20, help='Number of intervals of context lengths to evaluate')
    parser.add_argument('--document_depth_percent_min', type=float, default=0, help='minimum needle depth percent')
    parser.add_argument('--document_depth_percent_max', type=float, default=100, help='maximum needle depth percent')
    parser.add_argument('--document_depth_percent_intervals', type=int, default=10, help='number of depth intervals')
    parser.add_argument('--document_depth_percent_interval_type', type=str, default='linear', choices=['linear', 'sigmoid'], help='depth schedule')
    parser.add_argument('--mask_topk', type=int, default=0, help='mask topk heads, input a negative value to mask random heads')
    parser.add_argument('--head_score_path', type=str, default=None, help='path to head score file')
    parser.add_argument('--save_path', type=str, default=DEFAULT_SAVE_PATH, help='path to save results')
    parser.add_argument('--use_yarn', action='store_true', help='use YaRN rope scaling for supported models')
    parser.add_argument('--attn_implementation', type=str, default=None, help='attention implementation passed to the model loader')
    parser.add_argument('--vision_batch_size', type=int, default=None, help='vision tower batch size for Gemma-style loaders')
    parser.add_argument('--max_image_num', type=int, default=None, help='optional max number of images for Gemma-style loaders')
    parser.add_argument('--mask_prefill', dest='mask_prefill', action='store_true', default=True, help='apply head mask during the prefill pass as well as decoding; enabled by default')
    parser.add_argument('--no_mask_prefill', dest='mask_prefill', action='store_false', help='apply head mask during decoding only')
    # parser = add_args(parser)
    args = parser.parse_args()

    if args.task_data_path is None:
        args.task_data_path = str(default_task_data_path(args.dataset_name, args.max_context_len))

    print(args)

    example_ids = None
    initial_example_id = args.example_id
    if args.example_ids_file:
        example_ids = load_example_ids(args.example_ids_file)
        if not example_ids:
            raise ValueError("No example ids found")
        initial_example_id = example_ids[0]

    ht = LLMNeedleHaystackTester(
        model_name_or_path=args.model_name_or_path,
        task_suffix=args.task_suffix,
        save_contexts=True,
        save_results=True,
        mask_topk=args.mask_topk,
        min_context_len=args.min_context_len,
        max_context_len=args.max_context_len,
        ctx_len_intervals=args.ctx_len_intervals,
        document_depth_percent_min=args.document_depth_percent_min,
        document_depth_percent_max=args.document_depth_percent_max,
        document_depth_percent_intervals=args.document_depth_percent_intervals,
        document_depth_percent_interval_type=args.document_depth_percent_interval_type,
        head_score_path=args.head_score_path,
        task_data_path=args.task_data_path,
        task_image_dir=args.task_image_dir,
        save_path=args.save_path,
        dataset_name=args.dataset_name,
        example_id=initial_example_id,
        example_idx=args.example_idx,
        use_yarn=args.use_yarn,
        attn_implementation=args.attn_implementation,
        vision_batch_size=args.vision_batch_size,
        max_image_num=args.max_image_num,
        mask_prefill=args.mask_prefill,
    )

    if not example_ids:
        ht.start_test(args)
    else:
        base_save_name = ht.save_name
        status = 0
        for index, example_id in enumerate(example_ids, start=1):
            print(f"[{index}/{len(example_ids)}] example_id={example_id}", flush=True)
            try:
                ht.example_id = example_id
                ht.example_idx = -1
                ht.load_data()
                ht.testing_results = []
                ht.save_name = build_save_name(base_save_name, example_id)
                ht.start_test(args)
            except Exception:
                status = 1
                traceback.print_exc()
        raise SystemExit(status)
