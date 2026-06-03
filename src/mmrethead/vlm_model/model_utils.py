import re
import io
import time
import math
import torch
import base64
from PIL import Image

import logging
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# def eager_attention_forward(
#     module: nn.Module,
#     query: torch.Tensor,
#     key: torch.Tensor,
#     attention_mask: Optional[torch.Tensor],
#     scaling: float,
#     dropout: float = 0.0,
#     **kwargs,
# ):
#     key_states = repeat_kv(key, module.num_key_value_groups) # 1. repeat
#
#     attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling # 2. matmul
#     if attention_mask is not None: # 3. causal mask
#         causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
#         attn_weights = attn_weights + causal_mask
#     # 4. softmax
#     attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
#
#     return attn_weights


################# model attn #################
def _get_causal_mask(attn_weights):
    query_len, seq_len = attn_weights.size(-2), attn_weights.size(-1)
    causal_mask = torch.ones_like(attn_weights.transpose(-1, -2).squeeze(0))
    causal_mask = torch.triu(causal_mask, diagonal=-(seq_len - query_len))
    causal_mask = causal_mask.transpose(-1, -2)
    causal_mask = (1 - causal_mask) * torch.finfo(causal_mask.dtype).min
    return causal_mask


def _get_causal_mask_simplified(attn_weights):
    query_len, seq_len = attn_weights.size(-2), attn_weights.size(-1)
    causal_mask = torch.ones_like(attn_weights.squeeze(0))
    causal_mask = torch.tril(causal_mask, diagonal=(seq_len - query_len))
    causal_mask = (1 - causal_mask) * torch.finfo(causal_mask.dtype).min
    return causal_mask


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def _get_gemma3_image_bidirectional_mask(token_type_ids, query_positions, key_positions):
    if token_type_ids is None:
        return None

    token_type_ids = token_type_ids.to(query_positions.device)
    seq_len = token_type_ids.shape[1]
    if int(query_positions.max().item()) >= seq_len or int(key_positions.max().item()) >= seq_len:
        raise ValueError(
            "token_type_ids must cover the absolute key/query positions used for attention recomputation"
        )

    is_image = token_type_ids == 1
    previous_is_image = torch.nn.functional.pad(is_image, (1, 0), value=0)[:, :-1]
    new_image_start = is_image & ~previous_is_image
    image_group_ids = torch.cumsum(new_image_start.int(), dim=1) - 1
    image_group_ids = torch.where(is_image, image_group_ids, -1)

    query_token_types = token_type_ids[:, query_positions]
    key_token_types = token_type_ids[:, key_positions]
    query_image_groups = image_group_ids[:, query_positions]
    key_image_groups = image_group_ids[:, key_positions]

    query_is_image = query_token_types == 1
    key_is_image = key_token_types == 1
    same_image_group = query_image_groups[:, :, None] == key_image_groups[:, None, :]
    return query_is_image[:, :, None] & key_is_image[:, None, :] & same_image_group


def compute_attention_weights(key_states, query_states, query_start_idx=None, sliding_window=None, token_type_ids=None):
    bsz, num_heads, q_len, head_dim = query_states.size()
    num_key_value_heads = key_states.size(1)
    num_key_value_groups = num_heads // num_key_value_heads
    kv_seq_len = key_states.size(-2)

    key_states = repeat_kv(key_states, num_key_value_groups)
    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(head_dim)

    if attn_weights.size() != (bsz, num_heads, q_len, kv_seq_len):
        raise ValueError(
            f"Attention weights should be of size {(bsz, num_heads, q_len, kv_seq_len)}, but is {attn_weights.size()}")

    if query_start_idx is None:
        query_start_idx = kv_seq_len - q_len

    query_positions = torch.arange(
        query_start_idx,
        query_start_idx + q_len,
        device=attn_weights.device,
    )
    key_positions = torch.arange(kv_seq_len, device=attn_weights.device)

    allowed_mask = key_positions[None, :] <= query_positions[:, None]

    if sliding_window is not None:
        allowed_mask &= (query_positions[:, None] - key_positions[None, :]) < sliding_window

    image_bidirectional_mask = _get_gemma3_image_bidirectional_mask(
        token_type_ids,
        query_positions,
        key_positions,
    )
    if image_bidirectional_mask is not None:
        allowed_mask = allowed_mask[None, :, :] | image_bidirectional_mask
    else:
        allowed_mask = allowed_mask[None, :, :]

    attn_weights = attn_weights.masked_fill(
        ~allowed_mask[:, None, :, :],
        torch.finfo(attn_weights.dtype).min,
    )

    attn_lses = torch.logsumexp(attn_weights, dim=-1,
                                keepdim=True)  # Log-sum-exp of attention weights for numerical stability in softmax.
    attn_weights = torch.exp(attn_weights - attn_lses)  # softmax
    return attn_weights


def find_nth_match(haystack: str, needle: str, n: int) -> int:
    """
    0-indexed
    """
    if not needle:
        return -1
    pos = -1
    for _ in range(n + 1):
        pos = haystack.find(needle, pos + 1)
        if pos == -1:
            return -1
    return pos

def get_span_indices_by_text(full_prompt: str, char_offset_to_token_idx: dict, span: str):
    if span not in full_prompt:
        raise ValueError(f"Content not found in the prompt.")
    
    assert full_prompt.count(span) == 1, f"Span {span} appears more than once in the prompt."

    content_char_start = full_prompt.index(span)
    content_char_end = content_char_start + len(span) - 1  # inclusive end index

    # find the token indices that correspond to the character indices
    start_idx = char_offset_to_token_idx[content_char_start]
    end_idx = char_offset_to_token_idx[content_char_end]
    return start_idx, end_idx


def get_inverse_offset_mapping(offset_mapping):
    # create mapping from character offset to token index
    char_offset_to_token_idx = {}
    for i, (start, end) in enumerate(offset_mapping):
        for j in range(start, end):
            char_offset_to_token_idx[j] = i
    return char_offset_to_token_idx


def resize_image(image_list, image_resize):
    new_image_list = []
    for img in image_list:
        width, height = img.size
        new_width = int(width * image_resize)
        new_height = int(height * image_resize)
        img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

        new_image_list.append(img)
    return new_image_list


def resize_image_max_size(image_list, max_image_size):
    new_image_list = []
    for img in image_list:
        width, height = img.size
        if width <= max_image_size and height <= max_image_size:
            new_image_list.append(img)
            continue

        if width > height:
            new_width = max_image_size
            new_height = min(int(max_image_size / width * height), max_image_size)
        else:
            new_height = max_image_size
            new_width = min(int(max_image_size / height * width), max_image_size)

        img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

        new_image_list.append(img)

    return new_image_list


def image_to_io(image: Image.Image, format: str = 'PNG') -> io.BytesIO:
    img_io = io.BytesIO()
    image.save(img_io, format=format)
    img_io.seek(0)
    return img_io


def encode_image_base64(pil_image, format="PNG"):
    buffer = io.BytesIO()
    pil_image.save(buffer, format=format)
    img_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return img_str


def truncate_images(text, image_list, max_image_num=None):
    """
    keep the last max_image_num images in the example. Truncate image_list and remove beginning <image> marker in text.

    Args:
        text (str): query with <image>
        image_list (list): list of image path (or PIL.Image)
        max_image_num (int, optional): Max number of kept images

    Returns:
        tuple: revised text and image_list
    """
    if max_image_num is None or len(image_list) <= max_image_num:
        return text, image_list

    segments = re.split(r'(<image>)', text)

    # compute remove number
    keep_count = max_image_num
    remove_count = len(image_list) - keep_count

    # compute <image> marker numbers
    image_tags_count = segments.count('<image>')

    # safe check
    assert image_tags_count == len(image_list), f"Warning: Number of <image> tags ({image_tags_count}) doesn't match image_list length ({len(image_list)})"

    # build new text
    new_segments = []
    removed = 0
    for segment in segments:
        if segment == '<image>' and removed < remove_count:
            # replace with ""
            new_segments.append('')
            removed += 1
        else:
            new_segments.append(segment)

    # join all segments
    new_text = ''.join(new_segments)

    # only keep last kepp_count images
    new_image_list = image_list[-keep_count:]

    return new_text, new_image_list


def format_chat(text, image_list, system_prompt):
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
                {"role": "assistant", "content": system_prompt}]
    return messages


def call_api(func, limit: int=5, pause: int=10):
    """
    Call the API function with retries and rate limit handling.
    TODO: more error handling?
    """
    count = 0
    while True:
        try:
            output = func()
            break
        except Exception as e:
            logger.info(f"Exception while using api: {e}")
            msg = str(e).lower()

            if "rate limit" in msg or "rate_limit" in msg or "quota" in msg or "429" in msg or ("overloaded" in msg and count >= limit):
                logger.info(f"Rate limit exceeded, waiting {pause} secs and retrying...")
            count += 1
            if count < limit:
                logger.info(f"Encountered error {e}, retrying...")
                time.sleep(pause)
            else:
                logger.info("Skipping generation due to unknown error")
                raise e
    return output


class LLM:
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
    ):
        self.model_name = model_name
        self.temperature = temperature
        self.top_p = top_p
        self.max_length = max_length
        self.generation_max_length = generation_max_length
        self.generation_min_length = generation_min_length
        self.do_sample = do_sample
        self.use_chat_template = use_chat_template
        self.stops = None
        if stop_newline:
            self.stops = ["\n", "\n\n"]

    def prepare_inputs(self, test_item, data):
        raise NotImplementedError("prepare_inputs not implemented for LLM")
    
    def generate(self, inputs=None, prompt=None, **kwargs):
        raise NotImplementedError("generate not implemented for LLM")
