"""Narrow compatibility fix for compressed-tensors 0.12.2 packed asymmetric ZP.

Uses the library's public packing/dequantization primitives. No weights on disk
are changed. Inference is bf16 on the reconstructed quantization grid, not an
int4 speed/memory benchmark. Upstream reference:
https://github.com/vllm-project/compressed-tensors/blob/0.12.2/src/compressed_tensors/compressors/quantized_compressors/pack_quantized.py
"""
from importlib.metadata import version


def install():
    if version('compressed-tensors') != '0.12.2':
        raise RuntimeError('This compatibility patch is tested only for compressed-tensors==0.12.2')
    import torch
    from compressed_tensors.compressors.quantized_compressors.pack_quantized import PackedQuantizationCompressor, unpack_from_int32
    from compressed_tensors.quantization.lifecycle.forward import dequantize
    cls = PackedQuantizationCompressor
    if getattr(cls, '_hw7_zp_fixed', False):
        return
    original = cls.decompress_weight

    def decompress(self, compressed_data, quantization_args=None):
        args = quantization_args
        if args is None or args.symmetric or args.strategy not in ('group', 'channel'):
            return original(self, compressed_data, quantization_args)
        shape = tuple(int(v) for v in compressed_data['weight_shape'])
        scales = compressed_data['weight_scale']
        zp = compressed_data['weight_zero_point']
        if zp.dtype != torch.int32:
            raise ValueError('Expected packed int32 asymmetric zero points')
        unpacked_zp = unpack_from_int32(zp, args.num_bits, torch.Size((shape[0], scales.shape[-1])), packed_dim=0)


        compressed_data['weight_zero_point'] = unpacked_zp
        weights = unpack_from_int32(compressed_data['weight_packed'], args.num_bits, torch.Size(shape))
        return dequantize(x_q=weights, scale=scales, zero_point=unpacked_zp,
                          g_idx=compressed_data.get('weight_g_idx'))

    cls.decompress_weight = decompress
    cls._hw7_zp_fixed = True
