import torch
import triton
import triton.language as tl


@triton.jit
def vector_add_kernel(
    a_ptr,
    b_ptr,
    out_ptr,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N

    a = tl.load(a_ptr + offsets, mask=mask, other=0)
    b = tl.load(b_ptr + offsets, mask=mask, other=0)

    tl.store(out_ptr + offsets, a + b, mask=mask)


print("PyTorch:", torch.__version__)
print("Triton:", triton.__version__)
print("CUDA:", torch.version.cuda)
print("GPU:", torch.cuda.get_device_name(0))
print("Compute capability:", torch.cuda.get_device_capability(0))

assert torch.cuda.is_available()

# 1. PyTorch CUDA test
print("\n[1] Testing PyTorch CUDA...")

a = torch.randn(1024, 1024, device="cuda")
b = torch.randn(1024, 1024, device="cuda")
c = a @ b

torch.cuda.synchronize()
assert c.device.type == "cuda"
print("PASS: PyTorch CUDA matrix multiplication")

# 2. Triton JIT test
print("\n[2] Testing Triton JIT...")

N = 100_003
BLOCK = 256

a = torch.randn(N, device="cuda")
b = torch.randn(N, device="cuda")
out = torch.empty_like(a)

grid = (triton.cdiv(N, BLOCK),)

kernel = vector_add_kernel[grid](
    a, b, out,
    N,
    BLOCK=BLOCK,
)

torch.cuda.synchronize()
torch.testing.assert_close(out, a + b)

print("PASS: Triton JIT compilation and GPU execution")
print("Registers/thread:", kernel.n_regs)
print("Shared memory (bytes):", kernel.metadata.shared)

# 3. Inspect compiler output
print("\n[3] Checking generated code...")

for key in ("ttir", "ttgir", "llir", "ptx", "sass"):
    if key in kernel.asm:
        print(f"Available: {key}")

print("\nALL TESTS PASSED!")

