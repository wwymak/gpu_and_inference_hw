# HW2 Optimisation Notes

Notes from optimising `optimized_loop` in `hw2_task.py` against the `slow_loop`
baseline on an L40S GPU (tiny random Llama, 2 layers, d_model=2048, 128 new
tokens, prompt batch=1).

## Starting point

The original loop:

```python
def optimized_loop(model, input_ids, n_steps):
    generated_ids = input_ids.clone()
    generated_tokens = []
    for _ in range(n_steps):
        outputs = model(input_ids=generated_ids)
        next_token_id = torch.argmax(outputs.logits[:, -1, :], dim=-1)
        token_value = next_token_id.item()       # forces CUDA sync every step
        generated_tokens.append(token_value)
        generated_ids = torch.cat([generated_ids, next_token_id.unsqueeze(0)], dim=1)
    return generated_tokens
```

Three problems were identified:

1. `.item()` inside the loop forces a CPU/GPU sync every step — serialises the
   pipeline.
2. `torch.cat` re-allocates a `(B, L+1)` tensor each step and copies the old
   contents.
3. The model is re-processing the entire prefix every step (O(L²) total work
   over the loop), instead of caching keys/values.

## The three optimisations

### 1. Remove the per-step sync (`.item()`)

`.item()` blocks the CPU until the GPU has finished computing
`next_token_id`. Keep the tensor on the device, write it into a buffer, and
do a single host transfer at the end.

### 2. Preallocate the output buffer

Replace the growing Python list / `torch.cat` chain with a preallocated
`(B, n_steps)` tensor. One allocation up front, slice-write each step, no
copies.

### 3. KV cache (biggest win)

Pass `past_key_values` and `use_cache=True`, and feed only the new token
each step (`input_ids = next_token_id.unsqueeze(1)`). Turns each step from
O(L) into O(1) attention work, so the whole loop drops from O(L²) to O(L).

### Final version applied to `hw2_task.py`

```python
def optimized_loop(model, input_ids, n_steps):
    batch, _ = input_ids.shape
    new_tokens = torch.empty(
        (batch, n_steps), dtype=input_ids.dtype, device=input_ids.device
    )

    past_key_values = None
    cur = input_ids
    for i in range(n_steps):
        outputs = model(input_ids=cur, past_key_values=past_key_values, use_cache=True)
        past_key_values = outputs.past_key_values
        next_token_id = torch.argmax(outputs.logits[:, -1, :], dim=-1)
        new_tokens[:, i] = next_token_id
        cur = next_token_id.unsqueeze(1)

    return new_tokens[0].tolist()  # single D->H sync at the end
```

## Why bfloat16 alone already gives ~4× on an L40S

Loading the model in bf16 (via `build_model(torch.bfloat16)`) was enough to
get >4× speedup over the fp32 baseline before any loop changes. Three
compounding reasons:

1. **Tensor Core throughput.** L40S Ada tensor cores do bf16 matmul at
   roughly 2× the rate of fp32. LLM forward passes are matmul-dominated, so
   this alone gives ~2×.
2. **Memory bandwidth.** Weights and activations are half the size, so every
   layer reads/writes half the bytes. At batch=1 the model is often
   bandwidth-bound, so halving bytes ≈ halving time on those parts.
3. **KV cache and intermediates** are also half-sized → better cache
   locality.

~2× (compute) × ~2× (bandwidth), compounded, lands around 4× in practice.

## Measured results

All numbers from `time_generation` (128 tokens, single batch).

| Variant                                        | Wall clock | tok/s | Speedup vs slow | Self CUDA total |
|------------------------------------------------|-----------:|------:|----------------:|----------------:|
| Slow (fp32, no opts)                           |     1.49 s |  86.1 |          1.00 × |          105 ms |
| bf16 + preallocation + no `.item()` + KV cache |     0.26 s | 493.1 |          5.73 × |          5.5 ms |

The wall-clock speedup is ~6× but the **GPU-time** speedup is ~19× — the
remaining wall clock is dominated by Python/CPU launch overhead, which
matters a lot at this model size (2 layers, 128 steps).

## A subtle trap when isolating individual optimisations

While trying to measure "preallocation + no `.item()` only" (i.e. *without*
the KV cache), an early attempt looked like this:

```python
for i in range(n_steps):
    outputs = model(input_ids=input_ids)   # always the original prompt!
    next_token_id = torch.argmax(outputs.logits[:, -1, :], dim=-1)
    new_tokens[:, i] = next_token_id
    cur = next_token_id.unsqueeze(1)        # set but never fed back in
```

This *looked* fast (~0.25 s, similar to the fully-optimised version) but
the token preview gave it away: `[775, 775, 775, 775, ...]` — the same
token every step. The loop never actually advanced the input; it was
computing the same prompt-length forward pass 128 times.

The Self CUDA total made this clearer: **17.8 ms**, ~3× more GPU work than
the correctly-optimised version (5.5 ms). Wall clock hid the difference
because both runs were dominated by ~250 ms of CPU/launch overhead.

A similar bug version captured `past_key_values` but never passed it back
in, while feeding `cur = next_token_id.unsqueeze(1)` (a single token with
no history) — that produces wrong output too, even though it runs.

### The correct "preallocation + no `.item()`" only (no KV cache)

To do an apples-to-apples comparison against the slow baseline with *only*
the prealloc + sync-removal optimisations applied (still O(L²) attention),
the loop has to keep growing `generated_ids` the same way the slow loop
does:

```python
def optimized_loop(model, input_ids, n_steps):
    batch, _ = input_ids.shape
    new_tokens = torch.empty(
        (batch, n_steps), dtype=input_ids.dtype, device=input_ids.device
    )
    generated_ids = input_ids
    for i in range(n_steps):
        outputs = model(input_ids=generated_ids)
        next_token_id = torch.argmax(outputs.logits[:, -1, :], dim=-1)
        new_tokens[:, i] = next_token_id
        generated_ids = torch.cat([generated_ids, next_token_id.unsqueeze(1)], dim=1)
    return new_tokens[0].tolist()
```

## Takeaways

- For autoregressive generation on a small model, the **KV cache** is the
  dominant algorithmic win — it changes work per step, not just the
  constant factor.
- `.item()` in a hot loop is a silent latency tax; batch the host transfer
  to the end.
- Preallocate growing tensors when you know the final shape — saves
  per-step allocation/copy.
- When benchmarking individual optimisations, always sanity-check the
  output (token preview) against the reference loop. A "fast" variant that
  produces wrong tokens is meaningless.
- At small scales (few layers, few steps), wall clock is dominated by CPU
  launch overhead. Compare **Self CUDA total** in the profiler for an
  honest picture of GPU work.
