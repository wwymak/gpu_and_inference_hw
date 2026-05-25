import torch
import numpy as np

# ============================================================================
# Part 1: Implement PyTorch Functions
# ============================================================================
#
# TASK 1a: Implement an operation with the lowest arithmetic intensity.
# Use an op that performs essentially memory traffic with ~0 useful FLOPs
# per element.


def lowest_ai_fn(x: torch.Tensor) -> torch.Tensor:
    """Lowest arithmetic intensity baseline (0 FLOP/Byte)."""
    # TODO (1 line): implement a lowest-AI op
    return x.clone()


# TASK 1b: Implement a function with configurable arithmetic intensity.
# Build an element-wise compute operation where work increases with `num_ops`.
# Design it so fused arithmetic intensity grows roughly linearly with `num_ops`,
# while each element is still read/written once at the kernel boundary.
# Return either the eager function or a compiled version depending on the
# `compiled` flag so we can compare both on the roofline plot.
#
# Use an accumulator variable and implement fused multiply-add (FMA) style work
# explicitly, e.g. `acc = acc * x + x`, so each loop iteration contributes
# about 2 FLOPs per element in a realistic GPU-friendly pattern. We prefer this
# pattern here mainly because it gives clean FLOP accounting and resembles the
# kind of floating-point work GPUs are designed to do; Avoid patterns like repeated
# doubling (`x = x + x`), since long self-dependent pointwise chains can trigger
# very poor Inductor compile-time behavior and are also less useful for this
# roofline exercise.


def make_compute_fn(num_ops: int, compiled: bool = True):
    """Return an eager or compiled function whose work scales with num_ops."""

    def fn(x: torch.Tensor) -> torch.Tensor:
        acc = x
        for _ in range(num_ops):
            acc = acc * x + x
        return acc

    # TODO (1 line): return either `fn` or `torch.compile(fn)` based on `compiled`
    return torch.compile(fn) if compiled else fn


# ============================================================================
# Part 2: Benchmarking
# ============================================================================
#
# TASK 2: Complete the benchmark function using CUDA events.
# CUDA events measure GPU time precisely (not CPU wall time), which avoids
# including kernel launch overhead or CPU-GPU synchronization delays.


def benchmark_fn(fn, *args, warmup=25, rep=100) -> float:
    """Benchmark a GPU function using CUDA events.

    Returns median execution time in milliseconds.
    """
    # Warmup (triggers torch.compile on first call, then warms caches)
    for _ in range(warmup):
        fn(*args)
    torch.cuda.synchronize()
    # TODO: time `rep` runs using CUDA events and return median latency (ms)

    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(rep)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(rep)]

    for i in range(rep):
        start_events[i].record()
        fn(*args)
        end_events[i].record()

    torch.cuda.synchronize()
    times = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
    return float(np.median(times))


# TASK 3: Compute element-wise operation metrics from measured runtime.
# Count every arithmetic operation performed inside the loop (careful: each
# `acc = acc * x + x` iteration does more than one FLOP per element).
#
# Use different byte-traffic models for the two variants:
#   - compiled: assume the operation is fused, so each element is read once and
#     written once at the kernel boundary
#   - eager: estimate the traffic from the separate multiply and add operations
#     launched by PyTorch in each loop iteration, including intermediate tensors
#
# Return a tuple with:
#   - total_flops
#   - arithmetic_intensity  (FLOP / Byte)
#   - achieved_flops        (FLOP / s)


def compute_elementwise_metrics(num_elements, num_ops, bytes_per_element, ms, variant="compiled"):
    # TODO: compute total FLOPs, arithmetic intensity, and achieved FLOP/s
    total_flops = num_elements * num_ops * 2
    if variant == "compiled":
        bytes_moved = bytes_per_element * num_elements * 2
    else:
        #  tmp = acc * x → read acc + read x + write tmp = 3 element-traffics
        #  acc = tmp + x → read tmp + read x + write acc = 3 element-traffics
        bytes_moved = bytes_per_element * num_elements * 6 * num_ops
    ai = total_flops / bytes_moved
    achieved_flops = total_flops / (ms * 1e-3)

    return total_flops, ai, achieved_flops


# ============================================================================
# Part 3: Short Writeup
# ============================================================================
# Answer these after you generate `results/roofline.png` and inspect the points.
#
# Q1. Look at the compiled element-wise operations from `1 ops` through `64 ops`.
# Why does performance rise as arithmetic intensity increases even though the
# measured runtime changes only a little?
#  In 1 - 64 ops, the gpu is memory bound, the data transfer between cpu memory
# and gpu memory dominates, and the gpu spent most of the time being idle while waiting
# for data. However, this memory transfer time doesn't change much from 1-64 ops,
# while the flops increases proportionally a lot more, so performance rises
# operation.

# Q2. In one sample run, `matmul 1024x1024` achieved lower FLOP/s than the
# `128 ops` compiled element-wise operation. Give one or two reasons why that can
# happen on a large GPU like an H100.
# in a large h100, the matmul and element wise ops very much under utilise the gpu
# for the matmul cuda kernel, there is extra overhead (tile loading, tile boundary checks etc), and
# these overheads dominate. In contrast, there
# is none of these for the elementwise op, which compiles to a single streaming loop
# The matmuls are also ran in fp32 without enabling tensor cores -- this means the op did not utilise
# the extra capability offered by the hardware
#
# Q3. Between `64 ops` and `128 ops`, runtime increases more noticeably than it
# did for smaller operations. What does that suggest about what resource is
# becoming the bottleneck?
# This suggests that compute (gpu ) is becoming the bottleneck. In this scenario, the gpu
# is already almost at maximum FLOP/s, adding more ops mean the extra ops need to wait, unlike
# in the memory bound case

# Q4. Why do the eager `ops-K` points look so different from the compiled ones?
# the eager ops-k