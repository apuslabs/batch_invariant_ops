import torch
import torch.nn.functional as F
from batch_invariant_ops import set_batch_invariant_mode
torch.set_default_device('cuda')

# Just to get the logging out of the way haha
with set_batch_invariant_mode(True):
    pass

# When the batch size changes, the underlying implementation (e.g., cuBLAS) may
# choose a different tiling or reduction strategy to optimize performance. This
# alters the order of floating-point operations, and due to non-associativity,
# leads to numerically different results for the same input row. This test
# verifies that our custom kernel produces the exact same output for a given row
# regardless of the batch size.
def test_mm_invariance():
    B, D = 2048, 4096
    a = torch.linspace(-100, 100, B*D).reshape(B, D)
    b = torch.linspace(-100, 100, D*D).reshape(D, D)
    
    # Method 1: Matrix-vector multiplication (batch size 1)
    out1 = torch.mm(a[:1], b)
    
    # Method 2: Matrix-matrix multiplication, then slice (full batch)
    out2 = torch.mm(a, b)[:1]
    
    # Check if results are identical
    diff = (out1 - out2).abs().max()
    print(f"Difference: {diff.item()}")
    return diff.item() == 0

# `addmm` (add matrix multiply) is subject to the same batch-invariance issues as `mm`.
# The underlying matmul computation can change its internal algorithm based on the
# total size of the input tensors (M, N, K). A different algorithm means a
# different floating-point summation order, leading to different results. This
# test confirms our `addmm` override is batch-invariant.
def test_addmm_invariance():
    B, D = 2048, 4096
    a = torch.linspace(-100, 100, B*D).reshape(B, D)
    b = torch.linspace(-100, 100, D*D).reshape(D, D)
    bias = torch.linspace(-1, 1, D)

    # Method 1: addmm with batch size 1
    out1 = torch.addmm(bias, a[:1], b)

    # Method 2: addmm with full batch then slice first row
    out2 = torch.addmm(bias, a, b)[:1]

    # Check if results are identical
    diff = (out1 - out2).abs().max()
    print(f"Difference: {diff.item()}")
    return diff.item() == 0

# Unlike matmul, `log_softmax` is a row-wise operation. Its computation for one
# row is independent of other rows. Standard PyTorch implementations are already
# batch-invariant for this reason, as changing the batch size doesn't alter the
# reduction strategy for any given row. This test confirms both the standard
# implementation and our custom kernel are batch-invariant.
def test_log_softmax_invariance():
    B, D = 2048, 4096
    x = torch.linspace(-100, 100, B * D).reshape(B, D)

    # Method 1: log_softmax with batch size 1
    out1 = F.log_softmax(x[:1], dim=-1)

    # Method 2: log_softmax with full batch then slice first row
    out2 = F.log_softmax(x, dim=-1)[:1]

    # Check if results are identical
    diff = (out1 - out2).abs().max()
    print(f"Difference: {diff.item()}")
    return diff.item() == 0

# `mean` involves a reduction. When parallelizing, the strategy can change with
# the input size. For small batches, a kernel might split the reduction for a
# single row across multiple cores to improve utilization. For large batches, it
# might assign one row per core (data-parallel). This change in reduction
# strategy alters the summation order, causing batch-variance. This test
# ensures our custom kernel avoids this.
def test_mean_invariance():
    B, D = 2048, 4096
    x = torch.linspace(-100, 100, B * D).reshape(B, D)

    # Method 1: mean along last dim with batch size 1
    out1 = torch.mean(x[:1], dim=-1)

    # Method 2: mean along last dim with full batch then slice first row
    out2 = torch.mean(x, dim=-1)[:1]

    diff = (out1 - out2).abs().max()
    print(f"Difference: {diff.item()}")
    return diff.item() == 0

# Attention uses two matmuls (QK^T and P@V) and a softmax over the sequence.
# The fused kernel implementation can change reduction/tiling strategies with
# batch size and sequence length, leading to batch-variance in standard kernels.
# This test compares the first batch element computed alone vs. within a larger
# batch using PyTorch's scaled_dot_product_attention.
def test_attention_invariance():
    torch.manual_seed(0)
    B, H, T, Dh = 64, 8, 512, 64  # keep memory reasonable while stressing kernels
    dtype = torch.float16
    q = torch.randn(B, H, T, Dh, dtype=dtype, device='cuda')
    k = torch.randn(B, H, T, Dh, dtype=dtype, device='cuda')
    v = torch.randn(B, H, T, Dh, dtype=dtype, device='cuda')

    # Method 1: attention with batch size 1
    out1 = F.scaled_dot_product_attention(q[:1], k[:1], v[:1], is_causal=False)

    # Method 2: attention with full batch then slice first element
    out2 = F.scaled_dot_product_attention(q, k, v, is_causal=False)[:1]

    diff = (out1 - out2).abs().max()
    print(f"Difference: {diff.item()}")
    return diff.item() == 0

# Test with standard PyTorch (likely to show differences)
print("Standard PyTorch:")
with set_batch_invariant_mode(False):
    is_deterministic = test_mm_invariance()
    print(f"Deterministic: {is_deterministic}")
    print("addmm:")
    is_deterministic_addmm = test_addmm_invariance()
    print(f"Deterministic: {is_deterministic_addmm}")
    print("log_softmax:")
    is_deterministic_log_softmax = test_log_softmax_invariance()
    print(f"Deterministic: {is_deterministic_log_softmax}")
    print("mean.dim:")
    is_deterministic_mean = test_mean_invariance()
    print(f"Deterministic: {is_deterministic_mean}")
    print("attention (sdpa):")
    is_deterministic_attn = test_attention_invariance()
    print(f"Deterministic: {is_deterministic_attn}")

# Test with batch-invariant operations
print("\nBatch-Invariant Mode:")
with set_batch_invariant_mode(True):
    is_deterministic = test_mm_invariance()
    print(f"Deterministic: {is_deterministic}")
    print("addmm:")
    is_deterministic_addmm = test_addmm_invariance()
    print(f"Deterministic: {is_deterministic_addmm}")
    print("log_softmax:")
    is_deterministic_log_softmax = test_log_softmax_invariance()
    print(f"Deterministic: {is_deterministic_log_softmax}")
    print("mean.dim:")
    is_deterministic_mean = test_mean_invariance()
    print(f"Deterministic: {is_deterministic_mean}")
    # print("attention (flex):")
    # is_deterministic_attn = test_attention_invariance()
    # print(f"Deterministic: {is_deterministic_attn}")

