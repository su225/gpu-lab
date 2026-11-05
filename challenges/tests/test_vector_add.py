
import pytest
import torch

from vector_add import vector_add


@pytest.mark.parametrize("n", [
    1,
    7,
    8,
    9,
    31,
    32,
    33,
    255,
    256,
    257,
    1024,
    100_003,
])
@pytest.mark.parametrize("block_size", [8, 128, 256])
@pytest.mark.parametrize("num_warps", [1, 4])
def test_vector_add(n, block_size, num_warps):
    generator = torch.Generator(device="cuda").manual_seed(42)
    a = torch.randn(n, device="cuda", generator=generator)
    b = torch.randn(n, device="cuda", generator=generator)
    expected = torch.add(a, b)
    actual = vector_add(
        a, b,
        block_size=block_size,
        num_warps=num_warps,
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("block_size", [8, 128, 256])
def test_empty_input(block_size):
    a = torch.empty(0, device="cuda")
    b = torch.empty(0, device="cuda")
    actual = vector_add(a, b, block_size=block_size)
    assert actual.shape == (0,)
    assert actual.numel() == 0


def test_inputs_not_modified():
    a = torch.randn(100, device="cuda")
    b = torch.randn(100, device="cuda")
    a_before = a.clone()
    b_before = b.clone()
    vector_add(a, b)
    torch.testing.assert_close(a, a_before, rtol=0, atol=0)
    torch.testing.assert_close(b, b_before, rtol=0, atol=0)


def test_mismatched_shapes():
    a = torch.randn(10, device="cuda")
    b = torch.randn(11, device="cuda")
    with pytest.raises(ValueError, match="shapes"):
        vector_add(a, b)


def test_unsupported_dtype():
    a = torch.randn(10, device="cuda", dtype=torch.float16)
    b = torch.randn(10, device="cuda", dtype=torch.float16)
    with pytest.raises(TypeError, match="float32"):
        vector_add(a, b)


def test_invalid_block_size():
    a = torch.randn(10, device="cuda")
    b = torch.randn(10, device="cuda")
    with pytest.raises(ValueError, match="power of two"):
        vector_add(a, b, block_size=13)
