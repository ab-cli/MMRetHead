import re
import torch
from .model_utils import LLM
from transformers import AutoConfig
from transformers import AutoProcessor
from .custom_modeling.custom_modeling_qwen2_5_vl import Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info

class Qwen2_5VLModel(LLM):
    def __init__(
            self,
            model_name,
            **kwargs,
    ):
        super().__init__(model_name)

        model_kwargs = {}
        model_kwargs["offload_state_dict"] = kwargs.get("offload_state_dict", False)
        model_kwargs["attn_implementation"] = kwargs.get("attn_implementation", "flash_attention_2")
        self.do_prefill = kwargs.get("do_prefill", True) # Note: We overrided the Qwen2VL and Qwen2.5VL code and reduce the num_logits_to_keep to the last one token.

        self.processor = AutoProcessor.from_pretrained(model_name, use_fast=True)
        self.tokenizer = self.processor.tokenizer

        tokenizer = self.tokenizer
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.truncation_side = "left" # we truncate elder history than recent one
        tokenizer.padding_side = "left" # batch generation needs left padding


        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        config.max_position_embeddings = 32768
        if kwargs.get("use_yarn", False):
            config.rope_scaling = {
                "type": "yarn",
                "mrope_section": [16, 24, 24],
                "factor": 4,
                "original_max_position_embeddings": 32768,
            }

        model_cls = Qwen2_5_VLForConditionalGeneration

        self.model = model_cls.from_pretrained(
            model_name,
            torch_dtype=kwargs.get("torch_dtype", torch.bfloat16),
            device_map="auto",
            trust_remote_code=True,
            config=config,
            **model_kwargs
        )

        # use the default if possible, append if necessary
        self.device = self.model.device

    def get_image_token_count(self, curr_content):
        image = curr_content["image"]
        messages = [{"role": "user", "content": [{"type": "image", "image": image}]}]
        text = self.processor.apply_chat_template(messages, tokenize=False)
        image_inputs, _ = process_vision_info(
            messages,
            image_patch_size=self.processor.image_processor.patch_size,
        )
        inputs = self.processor(text=[text], images=image_inputs, return_tensors="pt")
        return inputs.input_ids.size(1)

    def get_text_token_count(self, curr_content):
        text = curr_content["text"]
        inputs = self.processor(text=text, add_special_tokens=False, return_tensors="pt")
        return inputs.input_ids.size(1)

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
            messages.append({"role": "assistant", "content": system_prompt})

        return messages

    def prepare_inputs(self, test_item, data):
        text = data["user_template"].format(**test_item)
        image_list = test_item["image_list"]
        messages = self.format_chat(text, image_list, data["system_template"])

        text = self.processor.apply_chat_template(
            messages, tokenize=False, continue_final_message=True
        )
        image_inputs, video_inputs = process_vision_info(
            messages,
            image_patch_size=self.processor.image_processor.patch_size,
        )
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        return inputs

    def precompute_content_segment_lengths(self, test_item):
        context = test_item["context"]
        image_list = test_item["image_list"]
        messages = self.format_chat(context, image_list)

        content = messages[0]["content"]

        for c in content:
            if c["type"] == "text":
                c["len"] = self.get_text_token_count(c)
            elif c["type"] == "image":
                c["len"] = self.get_image_token_count(c)
            else:
                raise ValueError(f"Unknown message type: {m['type']}")
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
