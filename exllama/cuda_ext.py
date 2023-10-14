# from abc import ABC
import os
import platform
import sys

import torch
from exllama_ext import (apply_rep_penalty, half_matmul, half_matmul_cublas,
                         make_q4, q4_matmul, q4_matmul_lora, rep_penalty,
                         rms_norm, rope_)

library_dir = os.path.dirname(os.path.abspath(__file__))
extension_name = "exllama_ext"
verbose = False

if os.getenv("COMPILE_JIT"):
    print("Compiling!")
    from torch.cuda.amp import custom_bwd, custom_fwd
    from torch.utils.cpp_extension import load
    exllama_ext = load(
        name=extension_name,
        sources=[
            os.path.join(library_dir, "exllama_ext/exllama_ext.cpp"),
            os.path.join(library_dir, "exllama_ext/cuda_buffers.cu"),
            os.path.join(library_dir, "exllama_ext/cuda_func/q4_matrix.cu"),
            os.path.join(library_dir, "exllama_ext/cuda_func/q4_matmul.cu"),
            os.path.join(library_dir, "exllama_ext/cuda_func/column_remap.cu"),
            os.path.join(library_dir, "exllama_ext/cuda_func/rms_norm.cu"),
            os.path.join(library_dir, "exllama_ext/cuda_func/rope.cu"),
            os.path.join(library_dir, "exllama_ext/cuda_func/half_matmul.cu"),
            os.path.join(library_dir, "exllama_ext/cuda_func/q4_attn.cu"),
            os.path.join(library_dir, "exllama_ext/cuda_func/q4_mlp.cu"),
            os.path.join(library_dir, "exllama_ext/cpu_func/rep_penalty.cpp")
        ],
        extra_include_paths=[os.path.join(library_dir, "exllama_ext")],
        verbose=verbose,
        extra_ldflags=(["cublas.lib"] + ([f"/LIBPATH:{os.path.join(sys.base_prefix, 'libs')}"]
                       if sys.base_prefix != sys.prefix else [])) if windows else [],
        extra_cuda_cflags=[
            "-lineinfo"] + (["-U__HIP_NO_HALF_CONVERSIONS__", "-O3"] if torch.version.hip else []),
        extra_cflags=["-O3"]
        # extra_cflags = ["-ftime-report", "-DTORCH_USE_CUDA_DSA"]
    )
else:
    import exllama_ext

# from exllama_ext import set_tuning_params
# from exllama_ext import prepare_buffers
# from exllama_ext import q4_mlp


# Dummy tensor to pass instead of g_idx since there is no way to pass "None" to a C++ extension

none_tensor = torch.empty((1, 1), device="meta")


# Construct Q4Matrix, return handle

def ext_make_q4(qweight, qzeros, scales, g_idx, device):

    return make_q4(qweight,
                   qzeros,
                   scales,
                   g_idx if g_idx is not None else none_tensor,
                   device)


# Matrix multiplication, returns x @ q4

def ext_q4_matmul(x, q4, q4_width, lora_A=None, lora_B=None):

    outshape = x.shape[:-1] + (q4_width,)
    x = x.view(-1, x.shape[-1])
    output = torch.empty((x.shape[0], q4_width),
                         dtype=torch.float16, device=x.device)

    if lora_A is None:
        q4_matmul(x, q4, output)
    else:
        lora_temp = torch.empty(
            (x.shape[0], lora_A.shape[1]), dtype=torch.float16, device=x.device)
        q4_matmul_lora(x, q4, output, lora_A, lora_B, lora_temp)

    return output.view(outshape)


# Matrix multiplication, returns x @ w, both half-precision tensors

def ext_half_matmul(x, w, cublas=False):

    outshape = x.shape[:-1] + (w.shape[1],)
    x = x.view(-1, x.shape[-1])

    if cublas:
        output = torch.empty(
            (x.shape[0], w.shape[1]), dtype=torch.float16, device=x.device)
        half_matmul_cublas(x, w, output)
    else:
        output = torch.zeros(
            (x.shape[0], w.shape[1]), dtype=torch.float16, device=x.device)
        half_matmul(x, w, output)

    return output.view(outshape)


# RoPE embeddings, in_place

def ext_rope_(x, sin, cos, past_len, num_heads, head_dim):

    rope_(x, sin, cos, past_len, num_heads, head_dim)


# RMS norm: x = x * w / sqrt(row_mean(x * x) + epsilon)

def ext_rms_norm(x, w, epsilon):

    outshape = x.shape
    x = x.view(-1, x.shape[-1])
    output = torch.empty_like(x)
    rms_norm(x, w, output, epsilon)

    return output.view(outshape)


def ext_rms_norm_(x, w, epsilon):

    outshape = x.shape
    x = x.view(-1, x.shape[-1])
    rms_norm(x, w, x, epsilon)


# Repetition penalty

def ext_rep_penalty_mask_cpu(vocab_size, sequence, penalty_max, sustain, decay):

    rep_mask = torch.empty(vocab_size, dtype=torch.float32)
    rep_penalty(sequence, rep_mask, penalty_max, sustain, decay)
    return rep_mask


def ext_apply_rep_penalty_mask_cpu(sequence, penalty_max, sustain, decay, logits):

    apply_rep_penalty(sequence, penalty_max, sustain, decay, logits)
