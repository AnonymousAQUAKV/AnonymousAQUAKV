import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from transformers import QuantoQuantizedCache, CacheConfig, QuantizedCacheConfig, QuantizedCache
from typing import Optional, Dict, Any, Tuple, List
from higgs_cache import HiggsQuantizedCache, HiggsQuantizedCacheConfig
from prekv.quantizers import  HiggsQuantizer
from prekv.cache_utils import TreatPrefixSeparately, PredictorHiggsCache, SingleChunkQuantizedCacheWithPredictors
from functools import partial


class Predictor(nn.Module):
    def __init__(self, inp_dim=3072):
        super().__init__()
        self.lin1 = nn.Linear(inp_dim, inp_dim * 2)
        self.lin2 = nn.Linear(inp_dim * 2, inp_dim * 4)
        self.lin3 = nn.Linear(inp_dim * 4, inp_dim)
        self.bn1 = nn.LayerNorm(inp_dim * 2)
        self.bn2 = nn.LayerNorm(inp_dim * 4)
        self.act = nn.GELU()


    def forward(self, x):
        x = self.bn1(self.act(self.lin1(x)))
        x = self.bn2(self.act(self.lin2(x)))
        x = self.lin3(x)
        return x


class QuantoPredictionCache(QuantoQuantizedCache):
    def __init__(self, cache_config: CacheConfig, nn_keys, nn_values, quant_k=True, quant_v=True) -> None:
        super().__init__(cache_config)
        self.nn_keys = nn_keys
        self.nn_values = nn_values
        self.quant_k = quant_k
        self.quant_v = quant_v
    def update(self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if layer_idx == 0:
            self._seen_tokens += key_states.shape[-2]

        if layer_idx % 2 == 0:
            if len(self.key_cache) <= layer_idx:
                self.key_cache.append(key_states)
                self.value_cache.append(value_states)
            else:
                self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states], dim=-2)
                self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_states], dim=-2)
            return (self.key_cache[layer_idx], self.value_cache[layer_idx])
        else:
            k_shape = self.key_cache[layer_idx - 1].shape
            if self.quant_k:
                predicted_key = self.nn_keys(self.key_cache[layer_idx - 1].transpose(1, 2).reshape(k_shape[0], k_shape[2], k_shape[1] * k_shape[3])).reshape(k_shape[0], k_shape[2], k_shape[1], k_shape[3]).transpose(1, 2)
                residual_key  = self._quantize(key_states - predicted_key[:, :, -key_states.shape[2]:, :], axis=0)
            else:
                residual_key = key_states
            if self.quant_v:
                predicted_value = self.nn_values(self.value_cache[layer_idx - 1].transpose(1, 2).reshape(k_shape[0], k_shape[2], k_shape[1] * k_shape[3])).reshape(k_shape[0], k_shape[2], k_shape[1], k_shape[3]).transpose(1, 2)
                residual_value = self._quantize(value_states - predicted_value[:, :, -key_states.shape[2]:, :], axis=0)
            else:
                residual_value = value_states
            if len(self.key_cache) <= layer_idx:
                self.key_cache.append(residual_key)
                self.value_cache.append(residual_value)
            else:
                self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], residual_key], dim=-2)
                self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], residual_value], dim=-2)
            if self.quant_k:
                result_key = self._dequantize(self.key_cache[layer_idx]).bfloat16() + predicted_key
            else:
                result_key = self.key_cache[layer_idx]

            if self.quant_v:
                result_value = self._dequantize(self.value_cache[layer_idx]).bfloat16() + predicted_value
            else:
                result_value = self.value_cache[layer_idx]
            return (result_key, result_value)


class TestQuantoPredictionCache(QuantoQuantizedCache):
    def __init__(self, cache_config: CacheConfig) -> None:
        super().__init__(cache_config)

    def update(self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if layer_idx == 0:
            self._seen_tokens += key_states.shape[-2]
        if layer_idx % 2 == 0:
            if len(self.key_cache) <= layer_idx:
                self.key_cache.append(key_states)
                self.value_cache.append(value_states)
            else:
                self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states], dim=-2)
                self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_states], dim=-2)
            return (self.key_cache[layer_idx], self.value_cache[layer_idx])
        else:
            k_shape = self.key_cache[layer_idx - 1].shape
            res_key  = self._quantize(key_states, axis=0)
            res_value = self._quantize(value_states, axis=0)
            if len(self.key_cache) <= layer_idx:
                self.key_cache.append(res_key)
                self.value_cache.append(res_value)
            else:
                self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], res_key], dim=-2)
                self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], res_value], dim=-2)
            return (self._dequantize(self.key_cache[layer_idx]).bfloat16(), self._dequantize(self.value_cache[layer_idx]).bfloat16())


def get_quanto_prediction_cache(device):
    cache_config = QuantizedCacheConfig(nbits=4)

    nn_keys, nn_values = Predictor(1024), Predictor(1024)
    nn_keys.load_state_dict(torch.load('../saved_models/keys_model_sen_all_layers_rot_ln.pt', weights_only=False))
    nn_values.load_state_dict(torch.load('../saved_models/values_model_sen_all_layers_rot_ln.pt', weights_only=False))

    past_key_values = QuantoPredictionCache(cache_config, nn_keys.bfloat16().to(device), nn_values.bfloat16().to(device))
    return past_key_values


def get_test_quanto_prediction_cache(device):
    cache_config = QuantizedCacheConfig(nbits=4)
    past_key_values = TestQuantoPredictionCache(cache_config)
    return past_key_values

def get_rht_cache(device, quantization_group_size, hadamard_groupsize, edenn_n, edenn_d):
    cache_config = HiggsQuantizedCacheConfig(
        quantization_group_size=quantization_group_size,
        hadamard_groupsize=hadamard_groupsize,
        edenn_n=edenn_n,
        edenn_d=edenn_d,
    )
    cache = HiggsQuantizedCache(cache_config)
    return cache

def get_higgs_predictors(device, hadamard_groupsize: int, edenn_n: int, edenn_d: int,
                         quantization_group_size: int, config: transformers.PretrainedConfig,
                         prefix_size:int = 4,
                         key_predictors: Optional[Dict[int, nn.Module]] = None,
                         value_predictors: Optional[Dict[int, nn.Module]] = None,
                         quantizer_type: str = "higgs"):

    # transfering predictors on to correct device
    if key_predictors:
        # maybe need to clone if several devices acceptable
        for i in key_predictors:
            key_predictors[i].to(device)
    if value_predictors:
        # maybe need to clone if several devices acceptable
        for i in value_predictors:
            value_predictors[i].to(device)

    # creating higgs quantizer
    if quantizer_type=="higgs":
        quantizer = HiggsQuantizer(hadamard_groupsize=hadamard_groupsize,
                                   edenn_d=edenn_d, edenn_n=edenn_n)
    else:
        # for the future
        raise NotImplementedError

    # creating cache with predictors
    cache = TreatPrefixSeparately(prefix_size=prefix_size,
                                  prefix_cache=transformers.DynamicCache(),
                                  suffix_cache=PredictorHiggsCache(
                                                 config=config, min_buffer_size=quantization_group_size,
                                                 save_dequantized_values=True,
                                                 make_quantized_cache=partial(
                                                   SingleChunkQuantizedCacheWithPredictors,
                                                   quantizer=quantizer,
                                                   key_predictors=key_predictors,
                                                   value_predictors=value_predictors
                                                 )
                                            ))
    return cache