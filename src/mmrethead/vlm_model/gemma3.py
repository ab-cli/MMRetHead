import re
import torch
from functools import partial

from .model_utils import LLM, truncate_images
from .model_utils import find_nth_match
from .model_utils import compute_attention_weights
from .model_utils import get_span_indices_by_text
from .model_utils import get_inverse_offset_mapping

from transformers import AutoConfig
from transformers import AutoProcessor

from .custom_modeling.custom_modelling_gemma3 import Gemma3ForConditionalGeneration
from copy import deepcopy
from .custom_modeling.cache_with_query import DynamicCacheWithQuery

def get_image_features_batch(self, pixel_values: torch.Tensor, vision_batch_size=32) -> torch.Tensor:
    """
    Projects the last hidden state from the vision model into language model space.

    Args:
        pixel_values (`torch.FloatTensor]` of shape `(batch_size, channels, height, width)`)
           The tensors corresponding to the input images.
        vision_batch_size (int): Number of images to process in each batch.
    Returns:
        image_features (`torch.Tensor`): Image feature tensor of shape `(num_images, image_length, embed_dim)`).
    """
    all_image_features = []
    num_images = pixel_values.shape[0]

    for start_idx in range(0, num_images, vision_batch_size):
        end_idx = min(start_idx + vision_batch_size, num_images)
        batch_pixel_values = pixel_values[start_idx: end_idx]
        batch_vision_outputs = self.vision_tower(pixel_values=batch_pixel_values).last_hidden_state
        batch_image_features = self.multi_modal_projector(batch_vision_outputs)
        all_image_features.append(batch_image_features)
    all_image_features = torch.cat(all_image_features, dim=0)

    return all_image_features


class Gemma3VLModel(LLM):
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
        self.vision_batch_size = kwargs.get("vision_batch_size", 32)
        self.max_image_num = kwargs.get("max_image_num", None)
        self.max_length = max_length
        self.processor = AutoProcessor.from_pretrained(model_name, use_fast=True)

        self.add_null_score = kwargs.get("add_null_score", True) # Modified : Added this


        tokenizer = self.processor.tokenizer
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.truncation_side = "left" # we truncate elder history than recent one
        tokenizer.padding_side = "left" # batch generation needs left padding

        self.model = Gemma3ForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=kwargs.get("torch_dtype", torch.bfloat16),
            device_map="auto",
            trust_remote_code=True,
            **model_kwargs
        )

        self.start_layer = 0
        self.end_layer = self.model.config.text_config.num_hidden_layers - 1

        import types
        self.model.get_image_features = types.MethodType(
            partial(get_image_features_batch, vision_batch_size=self.vision_batch_size), self.model)

        if kwargs.get("torch_compile", False): # Changed to False because it would break DynamicCache with query
            self.model = torch.compile(self.model)

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

    def format_chat(self, text, image_list, system_prompt):
        content = re.split(r'(<image>)', text)
        image_idx, new_content = 0, []
        for c in content:
            if c == "<image>":
                new_content.append({
                    "type": "image",
                    "url": image_list[image_idx]
                })
                image_idx += 1
            else:
                new_content.append({
                    "type": "text",
                    "text": c
                })
        assert image_idx == len(image_list)
        messages = [{"role": "user", "content": new_content},
                    {"role": "assistant", "content": [
                     {"type": "text", "text": system_prompt}
                     ]}]
        return messages

    # Modified : Removed the bbox thing and copied from qwen3
    def get_span_indices_by_image_id(self, full_prompt: str, char_offset_to_token_idx: dict, img_idx: int):
        image_char_start = find_nth_match(full_prompt, "<start_of_image>", img_idx)
        assert image_char_start != -1, f"image_char_start not found in full_prompt: {img_idx}"
            
        image_char_end = full_prompt.find("<end_of_image>", image_char_start)

        token_start = char_offset_to_token_idx[image_char_start]
        token_end = char_offset_to_token_idx[image_char_end]

        return token_start, token_end

    # Modified : copied build prompt 
    def build_prompt(self, test_item, data):
        text = data["user_template"].format(**test_item)
        image_list = test_item["image_list"]

        if self.max_image_num is not None:
            text, image_list = truncate_images(text, image_list, self.max_image_num)

        messages = self.format_chat(text, image_list, data["system_template"])

        # Need more explainations here
        # Step 1 get full input prompt
        # Apply chat template applies image token expansion
        inputs = self.processor.apply_chat_template(
            messages, tokenize=True, continue_final_message=True,
            return_dict=True, return_tensors="pt",
            return_offsets_mapping=True,
        )

        # Get expanded text by decoding the input_ids
        expanded_text = self.processor.tokenizer.decode(
            inputs["input_ids"][0], skip_special_tokens=False
        )
        
        # Step 2 get offset mapping
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

        # Things weird after this

        # query span
        if query["type"] == "text-image":                                                            
            # TODO: verify exact token strings from apply_chat_template output
            start_img = "<start_of_image>"
            end_img = "<end_of_image>"
            image_token = "<image_soft_token>"
            num_tokens_per_image = 256
            expanded_image = f"{start_img}{image_token * num_tokens_per_image}{end_img}"

            query_text = query["text"].replace(" <image>", "\n\n" + expanded_image + "\n")
            if query_text.startswith("<image>"):
                query_text = query_text.replace("<image>", expanded_image + "\n\n", 1)

            # Image expansion is direct now. No need to specificly find image thw for each image
        else:
            query_text = query["text"]

        # The expansion here is also removed as a result of the above change

        query_indices = get_span_indices_by_text(expanded_text, inverse_offset_mapping, query_text)

        inputs["gold_span_indices"] = gold_span_indices
        inputs["query_indices"] = query_indices

        return inputs

    # Like Qwen2.5 the content of this were moved build prompt
    def prepare_inputs(self, test_item, data):
        query_inputs = self.build_prompt(test_item, data)
        if self.add_null_score:
            null_test_item = deepcopy(test_item)
            query_span = test_item["query_span"]
            if query_span.get("type") == "text-image":
                image_count = len(query_span.get("image", []))
                if image_count == 0:
                    image_count = len(query_span.get("img_idx", []))
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
                null_test_item["query_span"] = {"type": "text", "text": null_query}
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
        attention_token_type_ids = inputs.token_type_ids
        if kv_cache is None:
            query_full_indices = list(range(query_indices[0], query_indices[1] + 1))
            kv_cache = DynamicCacheWithQuery(query_indices=query_full_indices)
            inputs = inputs.to(self.model.device)
            attention_token_type_ids = inputs.token_type_ids
        else:
            # use kv_cache from first query to speed up forward() for the calibration query.
            for i in range(len(kv_cache.layers)):
                kv_cache.layers[i].keys = kv_cache.layers[i].keys[:,:,:query_indices[0],:]
                kv_cache.layers[i].values = kv_cache.layers[i].values[:,:,:query_indices[0],:]
                kv_cache.layers[i].query = torch.tensor([], dtype=kv_cache.layers[i].dtype, device=kv_cache.layers[i].device)
            # kv_cache._seen_tokens = query_indices[0]
            start_idx = query_indices[0]
            attention_token_type_ids = attention_token_type_ids.to(self.model.device)

            inputs = {"input_ids": inputs.input_ids[:, start_idx:].to(self.model.device),
                      "token_type_ids": inputs.token_type_ids[:, start_idx:].to(self.model.device), # Gemma forward pass expects token type
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
                logits_to_keep=1, # No Compute logits
            )
        # 4. Compute attention weights
        kv_cache = outputs.past_key_values
        per_token_scores, per_query_argmax_token_indexes, per_query_argmax_attention_values, query_end_idx = [], [], [], query_indices[-1]
        text_config = self.model.config.text_config
        layer_types = getattr(text_config, "layer_types", None)
        model_sliding_window = getattr(text_config, "sliding_window", None)
        for i in range(self.start_layer, self.end_layer + 1):
            sliding_window = (
                model_sliding_window
                if layer_types is not None and layer_types[i] == "sliding_attention"
                else None
            )
            attn_weights = compute_attention_weights(
                kv_cache.layers[i].keys[:, :, :query_end_idx + 1],
                kv_cache.layers[i].query,
                query_start_idx=query_indices[0],
                sliding_window=sliding_window,
                token_type_ids=attention_token_type_ids[:, :query_end_idx + 1],
            ).to(self.device).squeeze(0)
            per_query_argmax_attention_value, per_query_argmax_token_index = attn_weights.max(dim=-1)
            attn_weights = attn_weights.mean(1)  # average over query tokens
            per_query_argmax_attention_values.append(per_query_argmax_attention_value)
            per_query_argmax_token_indexes.append(per_query_argmax_token_index)
            per_token_scores.append(attn_weights.squeeze(0))
        per_token_scores = torch.stack(per_token_scores, dim=0) # Shape: (n_layers, n_heads, n_tokens)
        per_query_argmax_token_indexes = torch.stack(per_query_argmax_token_indexes, dim=0) # Shape: (n_layers, n_heads, n_query_tokens)
        per_query_argmax_attention_values = torch.stack(per_query_argmax_attention_values, dim=0) # Shape: (n_layers, n_heads, n_query_tokens)

        return per_token_scores, per_query_argmax_token_indexes, per_query_argmax_attention_values, kv_cache

    # Removed generate function
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
