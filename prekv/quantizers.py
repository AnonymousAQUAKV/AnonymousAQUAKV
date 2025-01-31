"""
Base vector quantizer class to be used for training and inference with KV cache predictors and its instances (e.g HIGGS)
"""
from typing import TypeVar
import torch
from fast_hadamard_transform import hadamard_transform
from .edenn import higgs_quantize_dequantize, pad_to_block, HadLinear


class QuantizerBase:
    QuantizedState = TypeVar('QuantizedState')

    def quantize(self, x: torch.Tensor) -> QuantizedState: ...

    def dequantize(self, quantized: QuantizedState) -> torch.Tensor: ...

    def quantize_dequantize(self, x: torch.Tensor) -> torch.Tensor:
        return self.dequantize(self.quantize(x)).to(dtype=x.dtype, device=x.device)


class HiggsQuantizer(QuantizerBase):
    def __init__(self, hadamard_groupsize: int, edenn_d: int, edenn_n: int):
        super().__init__()
        self.hadamard_groupsize, self.edenn_d, self.edenn_n = hadamard_groupsize, edenn_d, edenn_n

    @torch.no_grad()
    def quantize(self, x: torch.Tensor):
        return quantize_linear_weight(x, self.hadamard_groupsize, self.edenn_d, self.edenn_n)

    @torch.no_grad()
    def dequantize(self, quantized: HadLinear) -> torch.Tensor:
        device = quantized.weight.device if quantized.weight.device.type == 'cuda' else 'cuda:0'
        return quantized(torch.eye(quantized.weight.shape[1], device=device).half()).T.contiguous()

    def quantize_dequantize(self, x: torch.Tensor) -> torch.Tensor:  # note: this shortcut is likely useless :D
        output_layer = quantize_linear_weight(x, self.hadamard_groupsize, self.edenn_d, self.edenn_n)
        device = x.device if x.device.type == 'cuda' else 'cuda:0'
        return output_layer(torch.eye(x.shape[1], device=device).half()
                            ).T.detach().contiguous().clone().to(device=x.device, dtype=x.dtype)


@torch.no_grad()
def quantize_linear_weight(weight: torch.Tensor, hadamard_groupsize: int, edenn_d: int, edenn_n: int):
    """HIGGS quantization code for weights"""
    weight = weight.to(dtype=torch.float32, device='cuda' if weight.device.type != 'cuda' else weight.device)
    # Pad to Hadamard transform size
    weight = pad_to_block(weight, [1], hadamard_groupsize)

    # Scale and Hadamard transform
    mult = weight.shape[1] // hadamard_groupsize
    weight = weight.reshape(-1, mult, hadamard_groupsize)
    scales = torch.linalg.norm(weight, axis=-1)
    weight = hadamard_transform(weight) / scales[:, :, None]

    # Pad to edenn_d and project
    weight = pad_to_block(weight, [2], edenn_d).reshape(weight.shape[0], mult, -1, edenn_d)

    for i in range(0, weight.shape[0], 64):
        weight[i: i + 64] = higgs_quantize_dequantize(weight[i:i + 64], edenn_d, edenn_n)
    weight = weight.reshape(weight.shape[0], mult, -1)

    # Cut the padded values
    weight = weight[..., :hadamard_groupsize]

    # Unscale
    weight = (weight * scales[:, :, None]).reshape(weight.shape[0], -1)

    return HadLinear(weight.half(), hadamard_groupsize)