import re
import torch
from functools import partial

from .model_utils import LLM, truncate_images
from .model_utils import find_nth_match
from .model_utils import get_span_indices_by_text

from transformers import AutoConfig
from transformers import AutoProcessor

from .custom_modeling.custom_modelling_gemma3 import Gemma3ForConditionalGeneration
from copy import deepcopy

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
        self.tokenizer = self.processor.tokenizer

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

    # Model specific: Tokenizers work differently
    def get_image_token_count(self, curr_content):
        image = curr_content["image"]
        messages = [{"role": "user", "content": [{"type": "image", "url": image}]}]
        inputs = self.processor.apply_chat_template(messages, tokenize=True, return_dict=True, return_tensors="pt")
        return inputs["input_ids"].size(1)

    def get_text_token_count(self, curr_content):
        text = curr_content["text"]
        inputs = self.processor(text=text, add_special_tokens=False, return_tensors="pt")
        return inputs.input_ids.size(1)

    # Model specific: List style content for Gemma. String style content for Qwen2.5
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
        messages = [{"role": "user", "content": new_content},
                    {"role": "assistant", "content": [
                     {"type": "text", "text": system_prompt}
                     ]}]
        return messages

    # Model specific: No need for process vision info like Qwen2.5
    def prepare_inputs(self, test_item, data):
        text = data["user_template"].format(**test_item)
        image_list = test_item["image_list"]
        messages = self.format_chat(text, image_list, data["system_template"])

        for m in messages:
            if isinstance(m["content"], list):
                for c in m["content"]:
                    if c["type"] == "image":
                        c["url"] = c.pop("image")

        inputs = self.processor.apply_chat_template(
            messages, tokenize=True, continue_final_message=True,
            return_dict=True, return_tensors="pt",
        )
        return inputs

    def precompute_content_segment_lengths(self, test_item):
        context = test_item["context"]
        image_list = test_item["image_list"]

        parts = re.split(r'(<image>)', context)
        image_idx, content = 0, []
        for p in parts:
            if p == "<image>":
                content.append({"type": "image", "image": image_list[image_idx]})
                image_idx += 1
            else:
                content.append({"type": "text", "text": p})

        for c in content:
            if c["type"] == "text":
                c["len"] = self.get_text_token_count(c)
            elif c["type"] == "image":
                c["len"] = self.get_image_token_count(c)
        return content

    def truncate_content(self, content, context_length):
        truncated_content = []
        curr_len = 0
        for c in content:
            truncated_content.append(c.copy())
            curr_len += c["len"]
            if curr_len >= context_length:
                break
        
        more_token_count = curr_len - context_length
        if more_token_count > 0:
            curr_len -= truncated_content[-1]["len"]
            if truncated_content[-1]["type"] == "text":
                new_segment = truncated_content[-1]
                new_tokens = self.tokenizer.encode(new_segment["text"], add_special_tokens=False)
                new_segment["text"] = self.tokenizer.decode(new_tokens[:-more_token_count])
                new_segment["len"] = self.get_text_token_count(new_segment)
                curr_len += truncated_content[-1]["len"]
            else:
                # if the last one is an image or video, we remove it
                truncated_content.pop()
        
        return truncated_content, curr_len

    def build_context(self, test_item, data, context_length, depth_percent, context_buffer=200):
        content = test_item["content"]

        # truncate message
        truncated_content, curr_len = self.truncate_content(content, context_length)

        # insert needle
        context_length -= context_buffer

        needle_copy = test_item["needle"].copy()
        if needle_copy["type"] == "text": # TODO
            needle_copy["text"] = " " + needle_copy["text"].strip() + " "

        if needle_copy["type"] == "text":
            needle_len = self.get_text_token_count(needle_copy)
        elif needle_copy["type"] == "image":
            needle_len = self.get_image_token_count(needle_copy)
    
        if curr_len + needle_len > context_length:
            truncated_content, curr_len = self.truncate_content(truncated_content, context_length - needle_len)

        needle_position = int(curr_len * (depth_percent / 100))
        insert_curr_len, insert_segment = 0, None
        for i, c in enumerate(truncated_content):
            if insert_curr_len + c["len"] > needle_position:
                insert_segment = i
                segment_needle_position = needle_position - insert_curr_len
                break
            insert_curr_len += c["len"]
    
        if insert_segment is None:
            truncated_content.append(needle_copy)
        else:
            if truncated_content[insert_segment]["type"] == "text":
                old_segment = truncated_content[insert_segment]
                segment_ids = self.tokenizer.encode(old_segment["text"], add_special_tokens=False)
                period_tokens = self.tokenizer.convert_tokens_to_ids(['.', 'Ġ.'])

                tokens_new_segment = segment_ids[:segment_needle_position]
                while tokens_new_segment and tokens_new_segment[-1] not in period_tokens:
                    tokens_new_segment.pop()
                    segment_needle_position -= 1

                new_segment_1 = {"type": "text", "text": self.tokenizer.decode(segment_ids[:segment_needle_position])}
                new_segment_2 = {"type": "text", "text": self.tokenizer.decode(segment_ids[segment_needle_position:])}
            
                # truncated_content = truncated_content[:insert_segment] + [new_segment_1, needle_copy, new_segment_2] + truncated_content[insert_segment + 1: ]
                truncated_content[insert_segment:insert_segment+1] = [new_segment_1, needle_copy, new_segment_2]
                
            else:
                # If insert segment is a image or a video, just insert before it
                truncated_content.insert(insert_segment, needle_copy)
        return truncated_content

    @torch.no_grad()
    def generate(self, inputs=None, prompt=None, **kwargs):
        inputs = inputs.to(self.model.device)
        input_len = inputs.input_ids.size(1)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=self.generation_max_length,
            min_new_tokens=self.generation_min_length,
            do_sample=self.do_sample,
            temperature=self.temperature if self.do_sample else None,
            top_p=self.top_p if self.do_sample else None,
            top_k = None,
            eos_token_id=self.stop_token_ids,
            pad_token_id=self.tokenizer.pad_token_id,
            return_dict_in_generate=True,
            output_scores=False,
        )
        text = self.processor.decode(outputs['sequences'][0, input_len:], skip_special_tokens=True)

        if input_len > 1500:
            save_prompt = self.processor.decode(inputs["input_ids"][0][:500]) + " <skip> " + self.processor.decode(
                inputs["input_ids"][0][-500:])
        else:
            save_prompt = self.processor.decode(inputs["input_ids"][0])
        return {
            "output": text,
            "input_len": input_len,
            "output_len": outputs['sequences'].size(1) - input_len,
            "input_text": save_prompt,
        }  
