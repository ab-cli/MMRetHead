import torch
import functools
from transformers.cache_utils import DynamicLayer
from transformers.cache_utils import DynamicCache, Cache
from transformers.cache_utils import DynamicSlidingWindowLayer
from transformers.configuration_utils import PreTrainedConfig
# from transformers.cache_utils import apply_processors # Don't use this to decorate update() anymore. We need add query_states.
from typing import Any, Optional, Iterable, Callable
from transformers.utils import is_torch_greater_or_equal


class DynamicLayerWithQuery(DynamicLayer):
    """
    A cache layer that grows dynamically as more tokens are generated. This is the default for generative models.
    It stores the key and value states as tensors of shape `[batch_size, num_heads, seq_len, head_dim]`.

    The DynamicLayerWithQuery includes query embeddings of specified indices, allowing computation of attention scores.
    """

    def lazy_initialization(self, key_states: torch.Tensor):
        self.dtype, self.device = key_states.dtype, key_states.device
        self.query = torch.tensor([], dtype=self.dtype, device=self.device)
        self.keys = torch.tensor([], dtype=self.dtype, device=self.device)
        self.values = torch.tensor([], dtype=self.dtype, device=self.device)
        self.is_initialized = True

    def update(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        cache_kwargs: Optional[dict[str, Any]] = None,
        store_values: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Update the key and value caches in-place, and return the necessary keys and value states.
        We add `query_states` of specified tokens in this new class.

        Args:
            query_states (`torch.Tensor`): The new query states to cache.
            key_states (`torch.Tensor`): The new key states to cache.
            value_states (`torch.Tensor`): The new value states to cache.
            cache_kwargs (`dict[str, Any]`, *optional*): Additional arguments for the cache.

        Returns:
            tuple[`torch.Tensor`, `torch.Tensor`]: The key and value states.
        """
        # Lazy initialization
        if not self.is_initialized:
            self.lazy_initialization(key_states)

        if query_states is not None:
            if self.query.numel() > 0:
                raise ValueError("Query_states should be empty when cache is initialized. We cannot cache two queries.")
            self.query = query_states
        self.keys = torch.cat([self.keys, key_states], dim=-2)
        if store_values:
            self.values = torch.cat([self.values, value_states], dim=-2)
        return self.keys, self.values


class DynamicCacheWithQuery(DynamicCache):
    """
    A cache that grows dynamically as more tokens are generated. This is the default for generative models.
    It stores the key and value states as a list of `CacheLayer`, one for each layer. The expected shape for each tensor
    in the `CacheLayer`s is `[batch_size, num_heads, seq_len, head_dim]`.
    If a config is passed, it will additionally check for sliding or hybrid cache structure, greatly reducing the
    memory requirement of the cached tensors to `[batch_size, num_heads, min(seq_len, sliding_window), head_dim]`.

    See `Cache` for details on common methods that are implemented by all cache classes.

    Args:
        ddp_cache_data (`Iterable[tuple[torch.Tensor, torch.Tensor]]`, *optional*):
            It was originally added for compatibility with `torch.distributed` (DDP). In a nutshell, it is
            `map(gather_map, zip(*caches))`, i.e. each item in the iterable contains the key and value states
            for a layer gathered across replicas by torch.distributed (shape=[global batch size, num_heads, seq_len, head_dim]).
            Note: it needs to be the 1st arg as well to work correctly
        config (`PreTrainedConfig`, *optional*):
            The config of the model for which this Cache will be used. If passed, it will be used to check for sliding
            or hybrid layer structure, greatly reducing the memory requirement of the cached tensors to
            `[batch_size, num_heads, min(seq_len, sliding_window), head_dim]`.
        offloading (`bool`, *optional*, defaults to `False`):
            Whether to perform offloading of the layers to `cpu`, to save GPU memory.
        offload_only_non_sliding (`bool`, *optional*, defaults to `False`):
            If `offloading` is `True`, this further decides if only the non-sliding layers will be offloaded (because
            usually the sliding layers are small in size, so there is no need to offload them, and skipping it is faster).
    """

    def __init__(
        self,
        ddp_cache_data: Optional[Iterable[tuple[Optional[torch.Tensor], ...]]] = None,
        config: Optional[PreTrainedConfig] = None,
        offloading: bool = False,
        offload_only_non_sliding: bool = False,
        query_indices=None,
        selected_layers: Optional[Iterable[int]] = None,
        store_values: bool = True,
        detach_for_attention: bool = False,
    ):
        self._query_indices = query_indices
        self.selected_layers = None if selected_layers is None else {int(layer) for layer in selected_layers}
        self.store_values = bool(store_values)
        self.detach_for_attention = bool(detach_for_attention)

        layers = []
        # If a config is passed, use it to infer the layer types and initialize accordingly
        if config is not None:
            decoder_config = config.get_text_config(decoder=True)
            sliding_window = getattr(decoder_config, "sliding_window", None) or getattr(
                decoder_config, "attention_chunk_size", None
            )
            layer_types = getattr(decoder_config, "layer_types", None)
            if layer_types is None:
                layer_types = [
                    "sliding_attention" if sliding_window is not None else "full_attention"
                    for _ in range(decoder_config.num_hidden_layers)
                ]
            # Some models have shared layers thus no cache is needed for them (e.g. Gemma3n)
            if hasattr(decoder_config, "num_kv_shared_layers"):
                layer_types = layer_types[: -decoder_config.num_kv_shared_layers]

            for layer_type in layer_types:
                # From a cache point of view, both sliding and chunked are the same in how they should behave and how many
                # states they should return - only the mask changes to make them different at the end!
                if layer_type in ("sliding_attention", "chunked_attention"):
                    assert False, "We don't support sliding window cache with query." # TODO support sliding window cache with query if needed
                    layers.append(DynamicSlidingWindowLayer(sliding_window=sliding_window))
                else:
                    layers.append(DynamicLayerWithQuery())

        # In this case, use the passed data to already fill in the Cache
        if ddp_cache_data is not None:
            # Init all the layers with the data
            for layer_idx, kv_and_optional_sliding in enumerate(ddp_cache_data):
                # If the config was not passed above, initialize a new cache layer for each entry of the ddp_data
                if config is None:
                    # kv_and_optional_sliding contains at least two elements: the key and value states. It can also
                    # contain a third element, which is an optional sliding window tensor.
                    sliding_window_tensor = kv_and_optional_sliding[2] if len(kv_and_optional_sliding) == 3 else None
                    # If there is a sliding window tensor, use it to initialize the layer
                    if sliding_window_tensor is not None:
                        # Since the same layer is dispatched across replicas, sliding_window is the same for all
                        sliding_window = sliding_window_tensor[0].item()
                        assert False, "We don't support sliding window cache with query." # TODO support sliding window cache with query if needed
                        layers.append(DynamicSlidingWindowLayer(sliding_window=sliding_window))
                    else:
                        layers.append(DynamicLayerWithQuery())
                # Update the layer with the data
                # TODO: we don't support this call.
                _, _ = layers[layer_idx].update(None, kv_and_optional_sliding[0], kv_and_optional_sliding[1])

        # If neither of config nor ddp_data was passed, then simply lazy init a full cache of DynamicLayer
        if len(layers) == 0:
            Cache.__init__(
                self,
                layer_class_to_replicate=DynamicLayerWithQuery,
                offloading=offloading,
                offload_only_non_sliding=offload_only_non_sliding,
            )
        else:
            Cache.__init__(self, layers=layers, offloading=offloading, offload_only_non_sliding=offload_only_non_sliding)
    
    def update(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[dict[str, Any]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Updates the cache with the new `key_states` and `value_states` for the layer `layer_idx`.

        Parameters:
            query_states (`torch.Tensor`):
                The new query states to cache.
            key_states (`torch.Tensor`):
                The new key states to cache.
            value_states (`torch.Tensor`):
                The new value states to cache.
            layer_idx (`int`):
                The index of the layer to cache the states for.
            cache_kwargs (`dict[str, Any]`, *optional*):
                Additional arguments for the cache subclass. These are specific to each subclass and allow new types of
                cache to be created.

        Return:
            A tuple containing the updated key and value states.
        """
        should_store_layer = self.selected_layers is None or int(layer_idx) in self.selected_layers
        if not should_store_layer:
            if self.detach_for_attention:
                return key_states.detach(), value_states.detach()
            return key_states, value_states

        # In this case, the `layers` were not provided, and we must append as much as `layer_idx`
        if self.layer_class_to_replicate is not None:
            while len(self.layers) <= layer_idx:
                self.layers.append(self.layer_class_to_replicate())

        if self.offloading:
            # Wait for the stream to finish if needed, and start prefetching the next layer
            torch.cuda.default_stream(key_states.device).wait_stream(self.prefetch_stream)
            self.prefetch(layer_idx + 1, self.only_non_sliding)

        if query_states is not None and self._query_indices is not None:
            expected_query_len = len(self._query_indices)
            if query_states.shape[-2] != expected_query_len:
                raise ValueError(
                    f"Expected query_states sequence length {expected_query_len}, "
                    f"got {query_states.shape[-2]}. Slice query_states before calling cache.update()."
                )

        keys, values = self.layers[layer_idx].update(
            query_states,
            key_states,
            value_states,
            cache_kwargs,
            store_values=self.store_values,
        )

        if self.offloading:
            self.offload(layer_idx, self.only_non_sliding)

        if self.detach_for_attention:
            return key_states.detach(), value_states.detach()
        return keys, values
    
