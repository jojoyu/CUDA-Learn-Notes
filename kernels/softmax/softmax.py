import torch
import time 
from torch.utils.cpp_extension import load
from functools import partial
from typing import Optional

torch.set_grad_enabled(False)

import os
os.environ['TORCH_CUDA_ARCH_LIST'] = '8.9' # for nv l20

# Load the CUDA kernel as a python module
lib = load(name='softmax_lib', 
           sources=['softmax.cu'], 
           extra_cuda_cflags=[
               "-O3",
                "-U__CUDA_NO_HALF_OPERATORS__",
                "-U__CUDA_NO_HALF_CONVERSIONS__",
                "-U__CUDA_NO_HALF2_OPERATORS__",
                "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
                "--expt-relaxed-constexpr",
                "--expt-extended-lambda",
                "--use_fast_math"
            ], 
           extra_cflags=['-std=c++17'])


def run_benchmark(perf_func: callable, x: torch.Tensor, 
                  tag: str, out: Optional[torch.Tensor] = None, 
                  warmup: int = 10, iters: int = 100,
                  show_all: bool = False):
    if out is not None: 
        out.fill_(0)      
    if out is not None:
        for i in range(warmup):
            perf_func(x, out)
    else:
        for i in range(warmup):
            _ = perf_func(x) 
    torch.cuda.synchronize()
    start = time.time()
    # iters
    if out is not None:
        for i in range(iters):
            perf_func(x, out)
    else:
        for i in range(iters):
            out = perf_func(x) 
    torch.cuda.synchronize()
    end = time.time()
    total_time = (end - start) * 1000 # ms
    mean_time = total_time / iters
    out_info = f"out_{tag}"
    out_val = out.flatten().detach().cpu().numpy().tolist()[:3]
    out_val = [round(v, 8) for v in out_val]
    out_val = [f"{v:<12}" for v in out_val]
    print(f"{out_info:>24}: {out_val}, time:{mean_time:.8f}ms")
    if show_all: print(out)
    return out, mean_time

# # grid memory fence
# print("-" * 100)
# N = 128 * 128
# print(" " * 45 + f"N={N}")
# print("-" * 100)
# x = torch.randn((N), device="cuda").cuda().float()
# out = torch.zeros_like(x).cuda().float().contiguous()
# run_benchmark(lib.softmax_f32,                        x, "f32(fence)",   out, show_all=True)
# run_benchmark(lib.softmax_f32x4,                      x, "f32x4(fence)", out, show_all=True)
# run_benchmark(partial(torch.softmax, dim=0, out=out), x, "f32_th", show_all=True)

###########################################
print("#" * 100)
import torch

# 创建一个2×3的随机tensor
S, H = 64, 64
# x = torch.randn((S, H), device="cuda").cuda().float().contiguous()
x = torch.arange(S*H, dtype=torch.float32, device="cuda").reshape(S, H).contiguous()
out = torch.zeros_like(x).cuda().float().contiguous()
print("Original tensor:")
print(x)

# 在dim=1(行)方向上计算softmax
output = torch.softmax(x, dim=1)
print("\nSoftmax result (dim=1):")
# 设置打印选项以显示完整tensor
torch.set_printoptions(profile="full")
print(output)

# 验证每行和为1
print("\nSum of each row:")
print(output.sum(dim=1))

run_benchmark(lib.online_safe_softmax_f32_per_token,              x, "f32(per)",         out, warmup=0, iters=1, show_all=True)
# run_benchmark(partial(torch.softmax, dim=1, out=out), x, "f32_th(per)", warmup=0, iters=1, show_all=True)
# 恢复默认打印选项
torch.set_printoptions(profile="default")
print("#" * 100)

###########################################

# # per token softmax
# print("-" * 100)
# S, H = 4096, 256
# print(" " * 45 + f"S={S}, H={H}")
# print("-" * 100)
# x = torch.randn((S, H), device="cuda").cuda().float().contiguous()
# out = torch.zeros_like(x).cuda().float().contiguous()
# run_benchmark(lib.softmax_f32_per_token,              x, "f32(per)",         out)
# run_benchmark(lib.softmax_f32x4_per_token,            x, "f32x4(per)",       out)
# run_benchmark(lib.safe_softmax_f32_per_token,         x, "f32(safe)",        out) 
# run_benchmark(lib.online_safe_softmax_f32_per_token,  x, "f32(safe+online)", out)
# run_benchmark(lib.online_safe_softmax_f32x4_pack_per_token,  x, "f32x4(safe+online)", out)
# run_benchmark(lib.safe_softmax_f32x4_per_token,       x, "f32x4(safe)",      out) 
# run_benchmark(partial(torch.softmax, dim=1, out=out), x, "f32_th(per)")

# print("-" * 100)
# x_f16 = x.half().contiguous()
# out_f16 = out.half().contiguous()
# run_benchmark(lib.safe_softmax_f16_f32_per_token,         x_f16, "f16f32(safe)",       out_f16) 
# run_benchmark(lib.safe_softmax_f16x2_f32_per_token,       x_f16, "f16x2f32(safe)",     out_f16) 
# run_benchmark(lib.safe_softmax_f16x8_pack_f32_per_token,  x_f16, "f16x8packf32(safe)", out_f16) 
# run_benchmark(partial(torch.softmax, dim=1, out=out_f16), x_f16, "f16_th(per)")
# print("-" * 100)

# # per token softmax
# print("-" * 100)
# S, H = 4096, 512
# print(" " * 45 + f"S={S}, H={H}")
# print("-" * 100)
# x = torch.randn((S, H), device="cuda").cuda().float().contiguous()
# out = torch.zeros_like(x).cuda().float().contiguous()
# run_benchmark(lib.softmax_f32_per_token,              x, "f32(per)",         out)
# run_benchmark(lib.softmax_f32x4_per_token,            x, "f32x4(per)",       out)
# run_benchmark(lib.safe_softmax_f32_per_token,         x, "f32(safe)",        out) 
# run_benchmark(lib.online_safe_softmax_f32_per_token,  x, "f32(safe+online)", out)
# run_benchmark(lib.online_safe_softmax_f32x4_pack_per_token,  x, "f32x4(safe+online)", out)
# run_benchmark(lib.safe_softmax_f32x4_per_token,       x, "f32x4(safe)",      out) 
# run_benchmark(partial(torch.softmax, dim=1, out=out), x, "f32_th(per)")

# print("-" * 100)
# x_f16 = x.half().contiguous()
# out_f16 = out.half().contiguous()
# run_benchmark(lib.safe_softmax_f16_f32_per_token,         x_f16, "f16f32(safe)",       out_f16) 
# run_benchmark(lib.safe_softmax_f16x2_f32_per_token,       x_f16, "f16x2f32(safe)",     out_f16) 
# run_benchmark(lib.safe_softmax_f16x8_pack_f32_per_token,  x_f16, "f16x8packf32(safe)", out_f16) 
# run_benchmark(partial(torch.softmax, dim=1, out=out_f16), x_f16, "f16_th(per)")
# print("-" * 100)

# # per token softmax
# print("-" * 100)
# S, H = 4096, 1024
# print(" " * 45 + f"S={S}, H={H}")
# print("-" * 100)
# x = torch.randn((S, H), device="cuda").cuda().float().contiguous()
# out = torch.zeros_like(x).cuda().float().contiguous()
# run_benchmark(lib.softmax_f32_per_token,              x, "f32(per)",         out)
# run_benchmark(lib.softmax_f32x4_per_token,            x, "f32x4(per)",       out)
# run_benchmark(lib.safe_softmax_f32_per_token,         x, "f32(safe)",        out) 
# run_benchmark(lib.online_safe_softmax_f32_per_token,  x, "f32(safe+online)", out)
# run_benchmark(lib.online_safe_softmax_f32x4_pack_per_token,  x, "f32x4(safe+online)", out)
# run_benchmark(lib.safe_softmax_f32x4_per_token,       x, "f32x4(safe)",      out) 
# run_benchmark(partial(torch.softmax, dim=1, out=out), x, "f32_th(per)")

# print("-" * 100)
# x_f16 = x.half().contiguous()
# out_f16 = out.half().contiguous()
# run_benchmark(lib.safe_softmax_f16_f32_per_token,         x_f16, "f16f32(safe)",       out_f16) 
# run_benchmark(lib.safe_softmax_f16x2_f32_per_token,       x_f16, "f16x2f32(safe)",     out_f16) 
# run_benchmark(lib.safe_softmax_f16x8_pack_f32_per_token,  x_f16, "f16x8packf32(safe)", out_f16) 
# run_benchmark(partial(torch.softmax, dim=1, out=out_f16), x_f16, "f16_th(per)")
# print("-" * 100)

# # per token softmax
# print("-" * 100)
# S, H = 4096, 2048
# print(" " * 45 + f"S={S}, H={H}")
# print("-" * 100)
# x = torch.randn((S, H), device="cuda").cuda().float().contiguous()
# out = torch.zeros_like(x).cuda().float().contiguous()
# run_benchmark(lib.softmax_f32x4_per_token,            x, "f32x4(per)",  out)
# run_benchmark(lib.safe_softmax_f32x4_per_token,       x, "f32x4(safe)", out) 
# run_benchmark(lib.online_safe_softmax_f32x4_pack_per_token,  x, "f32x4(safe+online)", out)
# run_benchmark(partial(torch.softmax, dim=1, out=out), x, "f32_th(per)")

# print("-" * 100)
# x_f16 = x.half().contiguous()
# out_f16 = out.half().contiguous()
# run_benchmark(lib.safe_softmax_f16x2_f32_per_token,       x_f16, "f16x2f32(safe)",     out_f16) 
# run_benchmark(lib.safe_softmax_f16x8_pack_f32_per_token,  x_f16, "f16x8packf32(safe)", out_f16) 
# run_benchmark(partial(torch.softmax, dim=1, out=out_f16), x_f16, "f16_th(per)")
# print("-" * 100)

# # per token softmax
# print("-" * 100)
# S, H = 4096, 4096
# print(" " * 45 + f"S={S}, H={H}")
# print("-" * 100)
# x = torch.randn((S, H), device="cuda").cuda().float().contiguous()
# out = torch.zeros_like(x).cuda().float().contiguous()
# run_benchmark(lib.softmax_f32x4_per_token,            x, "f32x4(per)",  out)
# run_benchmark(lib.safe_softmax_f32x4_per_token,       x, "f32x4(safe)", out) 
# run_benchmark(lib.online_safe_softmax_f32x4_pack_per_token,  x, "f32x4(safe+online)", out)
# run_benchmark(partial(torch.softmax, dim=1, out=out), x, "f32_th(per)")

# print("-" * 100)
# x_f16 = x.half().contiguous()
# out_f16 = out.half().contiguous()
# run_benchmark(lib.safe_softmax_f16x8_pack_f32_per_token,  x_f16, "f16x8packf32(safe)", out_f16) 
# run_benchmark(partial(torch.softmax, dim=1, out=out_f16), x_f16, "f16_th(per)")
# print("-" * 100)

# # per token softmax
# print("-" * 100)
# S, H = 4096, 8192
# print(" " * 45 + f"S={S}, H={H}")
# print("-" * 100)
# x = torch.randn((S, H), device="cuda").cuda().float().contiguous()
# out = torch.zeros_like(x).cuda().float().contiguous()
# x_f16 = x.half().contiguous()
# out_f16 = out.half().contiguous()
# run_benchmark(lib.safe_softmax_f16x8_pack_f32_per_token,  x_f16, "f16x8packf32(safe)", out_f16) 
# run_benchmark(partial(torch.softmax, dim=1, out=out_f16), x_f16, "f16_th(per)")

# # per token softmax
# print("-" * 100)
# S, H = 8192, 8192
# print(" " * 45 + f"S={S}, H={H}")
# print("-" * 100)
# x = torch.randn((S, H), device="cuda").cuda().float().contiguous()
# out = torch.zeros_like(x).cuda().float().contiguous()
# x_f16 = x.half().contiguous()
# out_f16 = out.half().contiguous()
# run_benchmark(lib.safe_softmax_f16x8_pack_f32_per_token,  x_f16, "f16x8packf32(safe)", out_f16) 
# run_benchmark(partial(torch.softmax, dim=1, out=out_f16), x_f16, "f16_th(per)")
# print("-" * 100)
