import re
import torch
from .model_utils import LLM
from .model_utils import find_nth_match
from .model_utils import compute_attention_weights
from .model_utils import get_span_indices_by_text
from .model_utils import get_inverse_offset_mapping
from qwen_vl_utils import process_vision_info
from transformers import AutoConfig
from transformers import AutoProcessor
from copy import deepcopy
from .custom_modeling.cache_with_query import DynamicCacheWithQuery
from .custom_modeling.custom_modeling_qwen3_vl import Qwen3VLForConditionalGeneration


class Qwen3VLModel(LLM):
    def __init__(
            self,
            model_name,
            temperature=0.9,
            top_p=0.9,
            max_length=32768,
            generation_max_length=2048,
            generation_min_length=0,
            do_sample=True,
            stop_newline=False,
            use_chat_template=False,
            **kwargs,
    ):
        super().__init__(
            model_name,
            temperature=temperature,
            top_p=top_p,
            max_length=max_length,
            generation_max_length=generation_max_length,
            generation_min_length=generation_min_length,
            do_sample=do_sample,
            stop_newline=stop_newline,
            use_chat_template=use_chat_template,
        )

        model_kwargs = {}
        model_kwargs["offload_state_dict"] = kwargs.get("offload_state_dict", False)
        model_kwargs["attn_implementation"] = kwargs.get("attn_implementation", "flash_attention_2")
        self.do_prefill = kwargs.get("do_prefill", True) # Note: We overrided the Qwen2VL and Qwen2.5VL code and reduce the num_logits_to_keep to the last one token.
        self.max_length = max_length
        self.add_null_score = kwargs.get("add_null_score", True)

        self.processor = AutoProcessor.from_pretrained(model_name)

        tokenizer = self.processor.tokenizer
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.truncation_side = "left" # we truncate elder history than recent one
        tokenizer.padding_side = "left" # batch generation needs left padding

        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=kwargs.get("torch_dtype", torch.bfloat16),
            device_map="auto",
            trust_remote_code=True,
            **model_kwargs
        )

        self.start_layer = 0
        self.end_layer = self.model.config.text_config.num_hidden_layers - 1

        # use the default if possible, append if necessary
        stop_token_ids = self.model.generation_config.eos_token_id
        stop_token_ids = [stop_token_ids] if not isinstance(stop_token_ids, list) else stop_token_ids
        if stop_newline:
            stop = list(set(["\n", "Ċ", "ĊĊ", "<0x0A>"]))
            stop_token_ids = list(
                set([tokenizer.convert_tokens_to_ids(stop_token) for stop_token in stop] + stop_token_ids))
            if tokenizer.unk_token_id is not None and tokenizer.unk_token_id in stop_token_ids:
                stop_token_ids.remove(tokenizer.unk_token_id)
            stop_token_ids = [x for x in stop_token_ids if x is not None]
        self.stop_token_ids = stop_token_ids
        self.device = self.model.device

    def format_chat(self, text, image_list, system_prompt=None):
        content = re.split(r'(<image>)', text)
        image_idx, new_content = 0, []
        for c in content:
            if c == "<image>":
                new_content.append({
                    "type": "image",
                    "image": image_list[image_idx]
                })
                image_idx += 1
            else:
                new_content.append({
                    "type": "text",
                    "text": c
                })
        assert image_idx == len(image_list)
        messages = [{"role": "user", "content": new_content},]
        
        if system_prompt is not None:
            messages.append({"role": "assistant", "content": [{"type": "text", "text": system_prompt}]})

        return messages
    
    
    def expand_visual_tokens(self, text, inputs):
        assert not isinstance(text, list), "The get_processed_text method doesn't support multiple input"
        text = [text]

        text = text.copy()  # below lines change text in-place
        if 'pixel_values' in inputs:
            merge_length = self.processor.image_processor.merge_size**2
            image_grid_thw = inputs["image_grid_thw"]

            index = 0
            for i in range(len(text)):
                while self.processor.image_token in text[i]:
                    num_image_tokens = image_grid_thw[index].prod() // merge_length
                    text[i] = text[i].replace(self.processor.image_token, "<|placeholder|>" * num_image_tokens, 1)
                    index += 1
                text[i] = text[i].replace("<|placeholder|>", self.processor.image_token)

        assert "pixel_values_videos" not in inputs, "We don't include video inputs now."
        return text
    
    def get_span_indices_by_image_id(self, full_prompt: str, char_offset_to_token_idx: dict, img_idx: int):
        image_char_start = find_nth_match(full_prompt, self.processor.vision_start_token, img_idx)
        assert image_char_start != -1, f"image_char_start not found in full_prompt: {img_idx}"
            
        image_char_end = full_prompt.find(self.processor.vision_end_token, image_char_start)

        token_start = char_offset_to_token_idx[image_char_start]
        token_end = char_offset_to_token_idx[image_char_end]

        return token_start, token_end
        
    def build_prompt(self, test_item, data):
        text = data["user_template"].format(**test_item)
        image_list = test_item["image_list"]
        messages = self.format_chat(text, image_list, data["system_template"])

        text = self.processor.apply_chat_template(
            messages, tokenize=False, continue_final_message=True
        )
        image_inputs, video_inputs = process_vision_info(
            messages, 
            image_patch_size=self.processor.image_processor.patch_size,)
            
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
            return_offsets_mapping=True,
        )

        # step 1 get full input prompt
        expanded_text = self.expand_visual_tokens(text, inputs)[0]
        # step 2 get offset mapping
        offset_mapping = inputs.pop("offset_mapping")[0]

        # step 3 turn offset format from `token_index: char_range` to `char_index: token_index`
        inverse_offset_mapping = get_inverse_offset_mapping(offset_mapping)

        # step 4 get_span_idx for both gold doc and query
        gold_span_list, query = test_item["gold_span"], test_item["query_span"]

        # gold_span can be text or image
        # TODO only tested "text" span
        gold_span_indices = [get_span_indices_by_text(expanded_text, inverse_offset_mapping, gs["text"])
                             if gs["type"] == "text" else
                             self.get_span_indices_by_image_id(expanded_text, inverse_offset_mapping, gs["img_idx"])
                             for gs in gold_span_list]

        # query can be text-only and text + image. We need to first process <image> in query and then use get_span_indices_by_text
        if query["type"] == "text-image":
            query_image_idx_list = query["img_idx"]
            query_inputs = {"pixel_values": inputs["pixel_values"], 
                            "image_grid_thw": torch.stack([inputs["image_grid_thw"][idx] for idx in query_image_idx_list], dim=0)}
        else:
            query_inputs = {}
        query_special_token = query["text"].replace("<image>", f"{self.processor.vision_start_token}{self.processor.image_token}{self.processor.vision_end_token}")

        expanded_query = self.expand_visual_tokens(query_special_token, query_inputs)[0]
        query_indices = get_span_indices_by_text(expanded_text, inverse_offset_mapping, expanded_query)

        inputs["gold_span_indices"] = gold_span_indices
        inputs["query_indices"] = query_indices

        return inputs
    
    def prepare_inputs(self, test_item, data):
        query_inputs = self.build_prompt(test_item, data)
        if self.add_null_score:
            null_test_item = deepcopy(test_item)
            query_span = test_item["query_span"]
            if query_span.get("type") == "text-image":
                image_count = len(query_span.get("image", []))
                null_question = "N/A N/A N/A N/A N/A" + "".join(
                    [f"\n{chr(c_idx + ord('A'))}. <image>" for c_idx in range(image_count)]
                )
                null_test_item["query_span"] = {
                    **query_span,
                    "text": null_question,
                }
                null_test_item["question"] = null_question
            else:
                null_query = "N/A N/A N/A N/A N/A"
                null_test_item["query_span"] = {"type": "text", "text": null_query, "image": [], "img_idx": []}
                null_test_item["question"] = null_query
            null_query_inputs = self.build_prompt(null_test_item, data)
            query_inputs["null_inputs"] = null_query_inputs
        return query_inputs

    def _compute_per_token_scores(self, inputs, query_indices, kv_cache=None):
        """
        Helper function to compute per-token Xattention scores from a given input dictionary.
        Returns a tensor of per-token scores and other relevant info.
        """
        # 2. Prepare cache and move to device
        if kv_cache is None:
            query_full_indices = list(range(query_indices[0], query_indices[1] + 1))
            kv_cache = DynamicCacheWithQuery(query_indices=query_full_indices)
            inputs = inputs.to(self.model.device)
        else:
            # use kv_cache from first query to speed up forward() for the calibration query.
            for i in range(len(kv_cache.layers)):
                kv_cache.layers[i].keys = kv_cache.layers[i].keys[:,:,:query_indices[0],:]
                kv_cache.layers[i].values = kv_cache.layers[i].values[:,:,:query_indices[0],:]
                kv_cache.layers[i].query = torch.tensor([], dtype=kv_cache.layers[i].dtype, device=kv_cache.layers[i].device)
            # kv_cache._seen_tokens = query_indices[0]
            start_idx = query_indices[0]

            inputs = {"input_ids": inputs.input_ids[:, start_idx:].to(self.model.device),
                      "attention_mask": inputs.attention_mask.to(self.model.device),
                      "cache_position": torch.arange(start_idx, start_idx + inputs.input_ids[:, start_idx:].shape[1], dtype=torch.int64, device=self.model.device)}

            query_full_indices = list(range(query_indices[0]-start_idx, query_indices[1]-start_idx+1))
            kv_cache._query_indices = query_full_indices
        
        # 3. Model forward pass
        with torch.no_grad():
            outputs = self.model(
                **inputs,
                use_cache=True,
                past_key_values=kv_cache,
                compute_logits=False,
                logits_to_keep=1,
            )
        # 4. Compute attention weights
        kv_cache = outputs.past_key_values
        per_token_scores, per_query_argmax_token_indexes, per_query_argmax_attention_values, query_end_idx = [], [], [], query_indices[-1]
        for i in range(self.start_layer, self.end_layer + 1):
            attn_weights = compute_attention_weights(kv_cache.layers[i].keys[:, :, :query_end_idx + 1], kv_cache.layers[i].query).to(
                self.device).squeeze(0)
            per_query_argmax_attention_value, per_query_argmax_token_index = attn_weights.max(dim=-1)
            attn_weights = attn_weights.mean(1)  # average over query tokens
            per_query_argmax_attention_values.append(per_query_argmax_attention_value)
            per_query_argmax_token_indexes.append(per_query_argmax_token_index)
            per_token_scores.append(attn_weights.squeeze(0))
        per_token_scores = torch.stack(per_token_scores, dim=0) # Shape: (n_layers, n_heads, n_tokens)
        per_query_argmax_token_indexes = torch.stack(per_query_argmax_token_indexes, dim=0) # Shape: (n_layers, n_heads, n_query_tokens)
        per_query_argmax_attention_values = torch.stack(per_query_argmax_attention_values, dim=0) # Shape: (n_layers, n_heads, n_query_tokens)

        return per_token_scores, per_query_argmax_token_indexes, per_query_argmax_attention_values, kv_cache

    @torch.no_grad()
    def get_attention_score(self, inputs=None, prompt=None, **kwargs):
        # step 5 prepare cache with query
        null_inputs = inputs.pop("null_inputs", None)
        query_indices = inputs.pop('query_indices')
        gold_span_indices = inputs.pop('gold_span_indices')
        
        per_token_scores, per_query_argmax_token_index, per_query_argmax_attention_value, kv_cache = self._compute_per_token_scores(inputs, query_indices)

        if null_inputs is not None:
            null_query_indices = null_inputs.pop('query_indices')
            null_gold_span_indices = null_inputs.pop('gold_span_indices')
            null_query_per_token_scores, _, _, _ = self._compute_per_token_scores(null_inputs, null_query_indices, kv_cache)

            min_length = min(per_token_scores.shape[-1], null_query_per_token_scores.shape[-1])
            per_token_scores_CAL = per_token_scores[:,:,:min_length] - null_query_per_token_scores[:,:,:min_length]
        else:
            per_token_scores_CAL = per_token_scores

        activation_outputs = {}
        if kwargs.get("save_activation_frequency", False):
            per_query_argmax_in_gold = torch.zeros_like(per_query_argmax_token_index, dtype=torch.bool)
            for gold_span in gold_span_indices:
                per_query_argmax_in_gold |= (per_query_argmax_token_index >= gold_span[0]) & (per_query_argmax_token_index <= gold_span[1])
            argmax_in_gold = per_query_argmax_in_gold.any(dim=-1)
            argmax_attention_value, best_query_index = per_query_argmax_attention_value.max(dim=-1)
            argmax_token_index = torch.gather(per_query_argmax_token_index, -1, best_query_index.unsqueeze(-1)).squeeze(-1)
            activation_outputs = {
                "argmax_token_index": argmax_token_index,
                "argmax_attention_value": argmax_attention_value,
                "argmax_in_gold": argmax_in_gold,
            }
        
        gold_span_score_list = []
        for i, gold_span in enumerate(gold_span_indices):
            curr_gold_span_token_scores = per_token_scores_CAL[:, :, gold_span[0] : gold_span[1]+1]
            curr_gold_span_score = curr_gold_span_token_scores.sum(dim=-1)
            gold_span_score_list.append(curr_gold_span_score)
        
        gold_span_score_list = torch.stack(gold_span_score_list, dim=0) # (n_gold_spans, n_layers, n_heads)

        # summ along all gold spans
        gold_span_score_sum = gold_span_score_list.sum(dim=0) # (n_layers, n_heads)

        input_len = inputs['input_ids'].size(1)
        if input_len > 1500:
            save_prompt = self.processor.decode(inputs["input_ids"][0][:500]) + " <skip> " + self.processor.decode(
                inputs["input_ids"][0][-500:])
        else:
            save_prompt = self.processor.decode(inputs["input_ids"][0])
        return {
            "gold_span_score_sum": gold_span_score_sum,
            "gold_span_indices": gold_span_indices,
            "input_len": input_len,
            "input_text": save_prompt,
            **activation_outputs,
        }
