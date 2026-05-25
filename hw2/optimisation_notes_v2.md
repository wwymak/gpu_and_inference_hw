# HW2 Inference Optimization Notes

End-to-end walkthrough of how the optimized generation loop in `hw2_task.py`
evolved, the reasoning behind each change, and the dead ends along the way.

## Starting point

The baseline `slow_loop` in `utils.py` re-encodes the entire growing prompt on
every step, runs in float32, has no KV cache, and pulls every token back to CPU
with `.item()` — about 134 tok/s on H100.

The optimized loop layered up the following changes.

## Layered optimizations (and their impact)

| # | Change | Wall-clock speedup vs baseline |
|---|--------|-------------------------------|
| 1 | `bfloat16` instead of `float32` | ~4.2x |
| 2 | Pre-allocated output tensor, no `.item()` per step | ~5.0x |
| 3 | KV cache via `use_cache=True` + `past_key_values` | ~5.7x |
| 4 | `logits_to_keep=1` (don't materialize all-token logits) | ~6.2x |
| 5 | `torch.compile(model)` (default mode) | ~6.4x |
| 6 | `torch.compile(model, mode="reduce-overhead")` + `StaticCache` | ~15x |

Items 1–5 are mostly self-explanatory. The interesting story is what it took
to get item 6 working.

## The `enable_flash_sdp` red herring

Earlier code included
`torch.backends.cuda.enable_flash_sdp(True)`. Removing it appeared to make the
script faster, which was confusing.

Looking at the profile traces, the attention kernel selected was
`fmha_cutlassF_bf16_aligned_32x128_gmem_sm80` — the **memory-efficient**
SDPA backend — in both cases. `enable_flash_sdp(True)` only *allows* the flash
backend; it doesn't force the dispatcher to pick it. For this workload (tiny
2-layer model, batch=1, single-token decoder queries) the memory-efficient
kernel is the right choice and PyTorch picks it automatically. The flag was
dead code.

The "speedup" from removing it was within run-to-run noise (CUDA totals 11.88 vs
11.92 ms). The conclusion was: leave it out.

(Side note: `torch.backends.cuda.enable_flash_sdp` is the old gate. The modern
API is `torch.nn.attention.sdpa_kernel(SDPBackend.FLASH_ATTENTION)` as a context
manager.)

## Why the initial CPU-bound win mattered

After items 1–5, the v6 profile showed:

- `Self CPU time total: 80ms`
- `Self CUDA time total: 12ms`

That ~7:1 ratio meant the GPU was idle most of the time waiting for the CPU to
launch the next kernel — a classic CPU-bound decoder. The remaining wall-clock
gap was almost entirely Python/dispatcher overhead, not compute.

The right hammer for that is **CUDA graphs**, which capture a fixed sequence of
kernel launches once and then replay them with effectively zero CPU overhead.
`torch.compile(model, mode="reduce-overhead")` enables this automatically.

## The CUDA graphs / KV cache fight

Naively flipping to `mode="reduce-overhead"` broke immediately with:

```
RuntimeError: Error: accessing tensor output of CUDAGraphs that has been
overwritten by a subsequent run.
```

Two root causes interact:

1. **CUDA graphs require static input/output addresses across runs.** The
   default HF `DynamicCache` re-allocates K/V tensors as the sequence grows.
   Every step changes addresses and shapes → cudagraphs gets confused.
2. **Outputs from one captured run get reused as inputs to the next.** Once
   you thread `past_key_values` between iterations, the second `model(...)`
   call sees K/V tensors that "belong" to the previous captured graph.

Even after calling `torch.compiler.cudagraph_mark_step_begin()` before each
invocation (the workaround the error message suggests), cudagraphs still
refuses to capture the prefill graph because `StaticCache.update` mutates
input tensors in place:

```
skipping cudagraphs due to mutated inputs (4 instances). Found from :
   ... self.keys.index_copy_(2, cache_position, key_states)
```

So even with `StaticCache`, the prefill path is incompatible with CUDA graph
capture.

## The fix that worked

Three pieces, all needed:

### a) `StaticCache` with a fixed maximum length

```python
past_key_values = StaticCache(
    config=eager_model.config,
    max_batch_size=batch,
    max_cache_len=MAX_CACHE_LEN,   # PROMPT_LEN + MAX_NEW_TOKENS
    device=device,
    dtype=eager_model.dtype,
)
```

Pre-allocated K/V buffers, written in place via `cache_position`. The tensor
addresses never change.

**Critical:** `max_cache_len` must be a constant across all calls (profile uses
12 steps; `time_generation` uses 128). Sizing the cache to
`prompt_len + n_steps` caused a full recompile + cudagraph re-capture on the
first 128-step call, which dumped 2 seconds of compile time into the timed
run. Sizing it to `PROMPT_LEN + MAX_NEW_TOKENS` everywhere fixed it.

### b) Prefill stays eager, decode is compiled

Because `StaticCache.update`'s in-place mutation blocks cudagraph capture for
the prefill shape, but works fine for the single-token decode shape, the
prefill call goes through the uncompiled `_orig_mod`:

```python
eager_model = model._orig_mod if hasattr(model, "_orig_mod") else model
...
# Prefill (eager — full prompt shape, not graph-captured)
outputs = eager_model(input_ids=input_ids, ...)
# Decode (compiled with cudagraphs — fixed [B, 1] shape)
for i in range(1, n_steps):
    torch.compiler.cudagraph_mark_step_begin()
    outputs = model(input_ids=cur, ...)
```

The decode shape (`[1, 1]`) is what gets called 127+ times — that's where
the cudagraph win lives. Prefill runs once and is cheap relative to decode.

### c) `cudagraph_mark_step_begin()` and `.clone()` on the boundary

Two cudagraph-correctness moves:

1. Call `torch.compiler.cudagraph_mark_step_begin()` before each compiled
   model call. This tells the cudagraph manager that the previous step's
   outputs are no longer live, so the next replay can safely reuse those
   buffers.
2. `.clone()` on `next_token_id` before storing it / feeding it back as
   `cur`. The raw output lives in cudagraph-managed memory that the next
   replay will overwrite; cloning it puts it in fresh memory.

### d) `@torch.inference_mode()`

Disables autograd bookkeeping (slightly stronger than `no_grad`). Small but
free.

## Result

128 tokens, H100:

- Slow baseline: 0.95s (134 tok/s)
- Optimized:     ~0.06s (~2000 tok/s)
- Speedup:       ~15x

The gap between profile-cycle CUDA totals (~12ms / 36 steps) and observed
wall-clock at 128 tokens is now small — i.e., the CPU launch overhead really
did dominate before, and CUDA graphs really did collapse it.

## What didn't help / wasn't worth it

- `enable_flash_sdp(True)` — no-op for this workload, see above.
- `fullgraph=True` on `torch.compile` — HF modeling code has graph breaks
  that fullgraph refuses; not worth chasing.
- Long warmup (10 iterations) — 3 is plenty once compile is stable.
- `mode="max-autotune"` — wasn't tried but would likely be the next thing
  to test for larger batch sizes / longer sequences.

## Possible next steps

- Use `mode="max-autotune"` and compare.
- Try larger batch sizes — many wins above are launch-overhead wins; with
  batch>1 the CUDA-bound work grows linearly while launch overhead is fixed,
  so the optimal recipe may differ.
- Replace the Python `for` loop with a CUDA-graph-captured outer loop, if
  HF ever exposes a clean way to do that.
- For real (non-toy) models, FlashAttention 2 will start mattering at long
  contexts and bigger batches; revisit the SDPA backend choice there.
