import math
from typing import List, Tuple

import torch
import torch.nn.functional as F
from transformers.cache_utils import QuantizedCacheConfig, QuantizedCache
from fast_hadamard_transform import hadamard_transform

from prekv.edenn import HadLinear, higgs_quantize, higgs_dequantize


# Init device
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def pad_to_block(tensor, dims, had_block_size, value=0):
    pad_dims = [0 for _ in range(2 * len(tensor.shape))]
    for dim in dims:
        size = tensor.shape[dim]
        next_multiple_of_1024 = ((size - 1) // had_block_size + 1) * had_block_size
        delta = next_multiple_of_1024 - size
        pad_dims[-2 * dim - 1] = delta

    return F.pad(tensor, pad_dims, "constant", value)


def quantize_tensor(
    weight: torch.Tensor,
    hadamard_groupsize: int,
    edenn_d: int,
    edenn_n: int,
    quantization_buffer_size: int = 64,
) -> List[torch.Tensor]:

    weight = weight.float()
    weight = pad_to_block(weight, [1], hadamard_groupsize)

    mult = weight.shape[1] // hadamard_groupsize
    weight = weight.view(-1, mult, hadamard_groupsize)
    scales = torch.linalg.norm(weight, axis=-1)
    weight = hadamard_transform(weight) / scales[:, :, None]

    weight = pad_to_block(weight, [2], edenn_d)
    weight = weight.view(weight.shape[0], mult, -1, edenn_d)

    codes = list()
    for i in range(0, weight.shape[0], quantization_buffer_size):
        local_weight = weight[i : i + quantization_buffer_size]
        local_codes = higgs_quantize(local_weight, edenn_d, edenn_n)
        codes.append(local_codes)

    return codes, scales


def dequantize_from_tensor_list(
    codes: List[torch.Tensor],
    scales: torch.Tensor,
    edenn_d: int,
    edenn_n: int,
    hadamard_groupsize: int
) -> torch.Tensor:
    quantization_buffer_size = codes[0].size(0)
    mult = codes[0].size(1)

    full_shape = list(codes[0].size())
    full_shape[0] = sum(code.size(0) for code in codes)

    weight = torch.empty(*full_shape, edenn_d, dtype=torch.half, device=device)
    codes = iter(codes)
    for i in range(0, weight.size(0), quantization_buffer_size):
        idx = next(codes)
        dequantized = higgs_dequantize(idx, edenn_d, edenn_n)
        weight[i : i + quantization_buffer_size] = dequantized

    weight = weight.view(weight.shape[0], mult, -1)
    weight = weight[..., :hadamard_groupsize]
    weight = weight * scales[:, :, None]
    weight = weight.view(weight.shape[0], -1)

    had_linear = HadLinear(weight, hadamard_groupsize)
    eye = torch.eye(weight.size(1), device=device, dtype=weight.dtype)
    return had_linear.forward(eye).T.contiguous()


class HiggsQuantizedCacheConfig(QuantizedCacheConfig):
    def __init__(
        self,
        quantization_group_size: int = 64,
        hadamard_groupsize: int = 64,
        edenn_d: int = 1,
        edenn_n: int = 16,
    ) -> None:
        nbits = math.log2(edenn_n) / edenn_d + 16 / hadamard_groupsize
        super().__init__(
            backend="higgs",
            nbits=nbits,
            axis_key=0,
            axis_value=0,
            q_group_size=quantization_group_size,
        )
        self.hadamard_groupsize = hadamard_groupsize
        self.edenn_d = edenn_d
        self.edenn_n = edenn_n


class HiggsQuantizedCache(QuantizedCache):
    def __init__(self, cache_config: HiggsQuantizedCacheConfig) -> None:
        super().__init__(cache_config)
        self.hadamard_groupsize = cache_config.hadamard_groupsize
        self.edenn_d = cache_config.edenn_d
        self.edenn_n = cache_config.edenn_n

    def _quantize(self, reference: torch.Tensor, axis: int) -> Tuple[List[torch.Tensor], torch.Tensor, int]:
        original_size = reference.size()
        dtype = reference.dtype
        batch_size, _, sequence_length, _ = reference.size()
        reference_rearr = reference.transpose(1, 2).contiguous().view(batch_size, sequence_length, -1).flatten(0, 1)
        codes, scales = quantize_tensor(reference_rearr, self.hadamard_groupsize, self.edenn_d, self.edenn_n)
        return (codes, scales, original_size, dtype)

    def _dequantize(self, q_tensor: Tuple[List[torch.Tensor], torch.Tensor, int]) -> torch.Tensor:
        codes, scales, (batch_size, n_heads, sequence_length, hidden_dim), dtype = q_tensor
        dequantized = dequantize_from_tensor_list(codes, scales, self.edenn_d, self.edenn_n, self.hadamard_groupsize)
        out = (
            dequantized.view(batch_size, sequence_length, n_heads, hidden_dim)
            .transpose(1, 2)
            .contiguous()
            .to(dtype)
        )
        return out
