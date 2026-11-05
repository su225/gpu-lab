import argparse

import torch
import triton
import triton.language as tl


@triton.jit
def vector_add_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    N,
    BLOCK_SIZE: tl.constexpr = 256,
    num_warps: int = 4,
):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    a = tl.load(a_ptr + offsets, mask=mask, other=0.0)
    b = tl.load(b_ptr + offsets, mask=mask, other=0.0)
    c = a + b
    tl.store(c_ptr + offsets, c, mask=mask)


def vector_add(
    a: torch.Tensor,
    b: torch.Tensor,
    block_size: int = 256,
    num_warps: int = 4,
):
    if not a.is_cuda or not b.is_cuda:
        raise ValueError("Inputs must be CUDA tensors")
    if a.shape != b.shape:
        raise ValueError("Input shapes must match")
    if a.dtype != torch.float32 or b.dtype != torch.float32:
        raise TypeError("Only float32 is supported")
    if not a.is_contiguous() or not b.is_contiguous():
        raise ValueError("Inputs must be contiguous")
    if block_size < 1 or block_size & (block_size - 1):
        raise ValueError("block_size must be a power of two")

    n = a.numel()
    c = torch.empty_like(a)
    if n == 0:
        return c
    grid = (triton.cdiv(n, block_size),)
    vector_add_kernel[grid](a, b, c, n,
        BLOCK_SIZE=block_size, num_warps=num_warps)
    return c


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--N", type=int, default=20)
    parser.add_argument("--block-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    assert args.N >= 0
    assert args.block_size > 0
    assert triton.next_power_of_2(args.block_size) == args.block_size

    torch.manual_seed(args.seed)

    a = torch.rand(args.N, device="cuda", dtype=torch.float32)
    b = torch.rand(args.N, device="cuda", dtype=torch.float32)
    c = vector_add(a, b, args.block_size)
    expected = a + b

    torch.testing.assert_close(c, expected, rtol=0, atol=0)

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"N: {args.N}")
    print(f"BLOCK_SIZE: {args.block_size}")
    print(f"Program instances: {triton.cdiv(args.N, args.block_size)}")

    from pprint import pprint
    pprint(f"A: {a.cpu().tolist()[:16]}")
    pprint(f"B: {b.cpu().tolist()[:16]}")
    pprint(f"C: {c.cpu().tolist()[:16]}")
    pprint("PASS: Triton output matches PyTorch")


if __name__ == "__main__":
    main()
