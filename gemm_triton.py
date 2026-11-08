
import argparse
from dataclasses import dataclass
from typing import Optional

import torch
import triton
import triton.language as tl
import triton.testing


DTYPES = {
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
    "fp32": torch.float32,
}


@dataclass(frozen=True)
class GemmConfig:
    M: int
    N: int
    K: int
    BM: int
    BN: int
    BK: int
    num_warps: int
    num_stages: int
    dtype: torch.dtype


# --------------------------------------------------
# Triton kernel: implement this yourself
# --------------------------------------------------

default_warps, default_stages = 4, 1
default_warp_expr = tl.constexpr(default_warps)
default_stage_expr = tl.constexpr(default_stages)

@triton.jit
def gemm_kernel(
    A, B, C,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    num_warps: tl.constexpr = default_warp_expr,
    num_stages: tl.constexpr = default_stage_expr,
):
    pid_row, pid_col = tl.program_id(0), tl.program_id(1)
    a_desc = tl.make_tensor_descriptor(A, [M,K], [K,1], [BM,BK])
    b_desc = tl.make_tensor_descriptor(B, [K,N], [N,1], [BK,BN])
    c_desc = tl.make_tensor_descriptor(C, [M,N], [N,1], [BM,BN])
    acc = tl.zeros([BM,BN], dtype=tl.float32)
    for k_tile in range(tl.cdiv(K,BK)):
        a_tile = a_desc.load([pid_row*BM, k_tile*BK])
        b_tile = b_desc.load([k_tile*BK, pid_col*BN])
        acc = tl.dot(a_tile, b_tile, acc)
    c_desc.store([pid_row*BM, pid_col*BN], acc)

# --------------------------------------------------
# Kernel launcher
# --------------------------------------------------

def gemm(
    a: torch.Tensor,
    b: torch.Tensor,
    config: GemmConfig,
) -> torch.Tensor:
    assert a.shape == (config.M, config.K)
    assert b.shape == (config.K, config.N)
    assert a.dtype == b.dtype == config.dtype
    assert a.is_cuda and b.is_cuda
    assert a.is_contiguous() and b.is_contiguous()

    c = torch.empty(
        (config.M, config.N),
        device=a.device,
        dtype=torch.float32,
    )

    grid = (
        triton.cdiv(config.M, config.BM),
        triton.cdiv(config.N, config.BN),
    )

    compiled = gemm_kernel[grid](
        a, b, c,
        config.M, config.N, config.K,
        config.BM, config.BN, config.BK,
        num_warps=config.num_warps,
        num_stages=config.num_stages,
    )
    print("Global scratch per CTA:", compiled.metadata.global_scratch_size)
    print("Global scratch alignment:", compiled.metadata.global_scratch_align)
    print("Shared memory per CTA:", compiled.metadata.shared)
    return c


def alloc_fn(
    size: int,
    alignment: int,
    stream: Optional[int],
) -> torch.Tensor:
    return torch.empty(size, device="cuda", dtype=torch.uint8)

triton.set_allocator(alloc_fn)

# --------------------------------------------------
# Correctness
# --------------------------------------------------

def verify(
    a: torch.Tensor,
    b: torch.Tensor,
    actual: torch.Tensor,
) -> None:
    # Reference uses the quantized input values
    # but performs the multiplication in FP32.
    reference = a.float() @ b.float()
    rtol = 1e-2
    atol = 1e-2
    torch.testing.assert_close(
        actual, reference,
        rtol=rtol,
        atol=atol,
    )
    max_error = (actual - reference).abs().max().item()
    print(f"PASS: max absolute error = {max_error:.6g}")


# --------------------------------------------------
# Command-line
# --------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Triton GEMM runner"
    )

    parser.add_argument("--M", type=int, default=512)
    parser.add_argument("--N", type=int, default=512)
    parser.add_argument("--K", type=int, default=512)

    parser.add_argument("--bm", type=int, default=64)
    parser.add_argument("--bn", type=int, default=64)
    parser.add_argument("--bk", type=int, default=32)

    parser.add_argument("--num-warps", type=int, default=4)
    parser.add_argument("--num-stages", type=int, default=1)

    parser.add_argument(
        "--dtype",
        choices=DTYPES.keys(),
        default="fp16",
    )

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--bench", action="store_true")

    args = parser.parse_args()

    for name in ("M", "N", "K", "bm", "bn", "bk",
                 "num_warps", "num_stages"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name} must be positive")

    for name in ("bm", "bn", "bk"):
        v = getattr(args, name)
        if v & (v - 1):
            parser.error(f"--{name} must be a power of 2")

    return args


def main() -> None:
    args = parse_args()

    config = GemmConfig(
        M=args.M,
        N=args.N,
        K=args.K,
        BM=args.bm,
        BN=args.bn,
        BK=args.bk,
        num_warps=args.num_warps,
        num_stages=args.num_stages,
        dtype=DTYPES[args.dtype],
    )

    torch.manual_seed(args.seed)
    torch.set_float32_matmul_precision("highest")

    a = torch.randn(
        (config.M, config.K),
        device="cuda",
        dtype=config.dtype,
    )
    b = torch.randn(
        (config.K, config.N),
        device="cuda",
        dtype=config.dtype,
    )

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GEMM: ({config.M}, {config.K}) "
          f"x ({config.K}, {config.N})")
    print(f"Input dtype: {args.dtype}")
    print(f"Output dtype: fp32")
    print(f"Tile: ({config.BM}, {config.BN}, {config.BK})")
    print(f"Warps: {config.num_warps}")
    print(f"Stages: {config.num_stages}")

    # First invocation also triggers JIT compilation.
    c = gemm(a, b, config)
    torch.cuda.synchronize()

    if args.check:
        verify(a, b, c)

    if args.bench:
        ms = triton.testing.do_bench(
            lambda: gemm(a, b, config),
            warmup=100,
            rep=500,
        )
        flops = 2 * config.M * config.N * config.K
        tflops = flops / (ms * 1e-3) / 1e12

        print(f"Latency: {ms:.4f} ms")
        print(f"Throughput: {tflops:.2f} TFLOP/s")


if __name__ == "__main__":
    main()
