
import argparse
from dataclasses import dataclass
from typing import Optional

import torch
import triton
from triton.experimental import gluon
import triton.language as tl

from triton.experimental import gluon as g
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.nvidia.hopper import TensorDescriptor
from triton.experimental.gluon.language.nvidia.hopper import (
    tma,
    mbarrier,
    mma_v2,
)
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
    acc = tl.zeros([BM,BN], dtype=tl.float32)
    for k_tile in range(tl.cdiv(K,BK)):
        a_tile = a_desc.load([pid_row*BM, k_tile*BK])
        b_tile = b_desc.load([k_tile*BK, pid_col*BN])
        acc = tl.dot(a_tile, b_tile, acc)
    rows = pid_row*BM + tl.arange(0,BM)
    cols = pid_col*BN + tl.arange(0,BN)
    c_ptrs = C + rows[:,None]*N + cols[None,:]
    tl.store(c_ptrs, acc, mask=(rows[:,None] < M) & (cols[None,:] < N))


@g.jit
def gemm_gluon_kernel(
    a_desc, b_desc, C,
    M: gl.constexpr,
    N: gl.constexpr,
    K: gl.constexpr,
    BM: gl.constexpr,
    BN: gl.constexpr,
    BK: gl.constexpr,
):
    pid_row, pid_col = gl.program_id(0), gl.program_id(1)
    row_start, col_start = pid_row * BM, pid_col * BN
    smem_a = gl.allocate_shared_memory(a_desc.dtype, a_desc.block_type.shape, a_desc.layout)
    smem_b = gl.allocate_shared_memory(b_desc.dtype, b_desc.block_type.shape, b_desc.layout)

    mma_layout: gl.constexpr = gl.NVMMADistributedLayout(version=[2,0], warps_per_cta=[2,2], instr_shape=[16,8])
    a_reg_layout: gl.constexpr = gl.DotOperandLayout(operand_index=0, parent=mma_layout, k_width=2)
    b_reg_layout: gl.constexpr = gl.DotOperandLayout(operand_index=1, parent=mma_layout, k_width=2)

    barrier = gl.allocate_shared_memory(gl.int64, [1], mbarrier.MBarrierLayout())
    mbarrier.init(barrier, count=1)
    phase = 0

    acc = gl.zeros(shape=(BM,BN), dtype=gl.float32, layout=mma_layout)
    for k_tile in range(gl.cdiv(K,BK)):
        k_start = k_tile * BK
        mbarrier.expect(barrier, a_desc.block_type.nbytes + b_desc.block_type.nbytes)
        tma.async_load(a_desc, [row_start, k_start], barrier, smem_a)
        tma.async_load(b_desc, [k_start, col_start], barrier, smem_b)
        mbarrier.wait(barrier, phase=phase, deps=[smem_a, smem_b])
        phase ^= 1

        a_tile = smem_a.load(a_reg_layout)
        b_tile = smem_b.load(b_reg_layout)
        acc = mma_v2(a_tile, b_tile, acc)
        gl.barrier()

    mbarrier.invalidate(barrier)

    store_layout: gl.constexpr = gl.BlockedLayout(
        [1,4], # Elements per thread
        [4,8], # Thread layout per warp
        [2,2], # Warps per CTA
        [1,0], # Prioritize columns(0) and then rows(1)
    )
    out = gl.convert_layout(acc, store_layout)
    rows = pid_row*BM + gl.arange(0, BM, layout=gl.SliceLayout(1, store_layout))
    cols = pid_col*BN + gl.arange(0, BN, layout=gl.SliceLayout(0, store_layout))
    ptrs = C + rows[:,None]*N + cols[None,:]
    gl.store(ptrs, out, mask=(rows[:,None] < M) & (cols[None,:] < N))


def gemm(
    a: torch.Tensor,
    b: torch.Tensor,
    config: GemmConfig,
    kernel: str = 'triton'
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
    grid = (triton.cdiv(config.M, config.BM), triton.cdiv(config.N, config.BN))
    M,N,K = config.M,config.N,config.K
    BM,BN,BK = config.BM,config.BN,config.BK
    if kernel == 'triton':
        compiled = gemm_kernel[grid](
            a, b, c,
            M,N,K,BM,BN,BK,
            num_warps=config.num_warps,
            num_stages=config.num_stages,
        )
        print("Global scratch per CTA:", compiled.metadata.global_scratch_size)
        print("Global scratch alignment:", compiled.metadata.global_scratch_align)
        print("Shared memory per CTA:", compiled.metadata.shared)
        torch.cuda.synchronize()
    elif kernel == 'gluon':
        a_layout = gl.NVMMASharedLayout.get_default_for([BM,BK], gl.float16)
        b_layout = gl.NVMMASharedLayout.get_default_for([BK,BN], gl.float16)
        a_desc = TensorDescriptor.from_tensor(a, [BM,BK], a_layout)
        b_desc = TensorDescriptor.from_tensor(b, [BK,BN], b_layout)
        compiled = gemm_gluon_kernel[grid](
            a_desc, b_desc, c,
            M,N,K,BM,BN,BK,
        )
        torch.cuda.synchronize()
    else:
        raise Exception(f"unknown kernel: {kernel}")
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

    parser.add_argument(
        "--kernel",
        choices=["triton", "gluon"],
        default="triton",
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
    c = gemm(a, b, config, kernel=args.kernel)
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
