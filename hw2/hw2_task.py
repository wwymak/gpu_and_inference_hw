import argparse
import shutil
from pathlib import Path

import torch
from transformers import StaticCache
from utils import (
    build_model,
    get_input_ids,
    slow_loop,
    time_generation,
    MAX_NEW_TOKENS,
    MODEL_NAME,
    PROFILE_STEPS,
    PROMPT_LEN,
    RESULTS_DIR,
)

MAX_CACHE_LEN = PROMPT_LEN + MAX_NEW_TOKENS


@torch.inference_mode()
def optimized_loop(model, input_ids, n_steps):
    """Generate n_steps tokens with a static KV cache; prefill runs eager, decode is compiled."""
    eager_model = model._orig_mod if hasattr(model, "_orig_mod") else model
    batch, prompt_len = input_ids.shape
    device = input_ids.device

    past_key_values = StaticCache(
        config=eager_model.config,
        max_batch_size=batch,
        max_cache_len=MAX_CACHE_LEN,
        device=device,
        dtype=eager_model.dtype,
    )

    new_tokens = torch.empty(
        (batch, n_steps), dtype=input_ids.dtype, device=device
    )

    cache_position = torch.arange(prompt_len, device=device)
    outputs = eager_model(
        input_ids=input_ids,
        past_key_values=past_key_values,
        use_cache=True,
        cache_position=cache_position,
        logits_to_keep=1,
    )
    next_token_id = torch.argmax(outputs.logits[:, -1, :], dim=-1)
    new_tokens[:, 0] = next_token_id
    cur = next_token_id.unsqueeze(1)

    for i in range(1, n_steps):
        cache_position = torch.tensor([prompt_len + i - 1], device=device)
        torch.compiler.cudagraph_mark_step_begin()
        outputs = model(
            input_ids=cur,
            past_key_values=past_key_values,
            use_cache=True,
            cache_position=cache_position,
            logits_to_keep=1,
        )
        next_token_id = torch.argmax(outputs.logits[:, -1, :], dim=-1).clone()
        new_tokens[:, i] = next_token_id
        cur = next_token_id.unsqueeze(1)

    return new_tokens[0].tolist()


def profile(loop_fn, model, input_ids, trace_name: str):
    """Profile loop_fn, export per-cycle TensorBoard traces and copy to named Chrome traces."""
    stem = Path(trace_name).stem
    suffix = Path(trace_name).suffix or ".json"
    tb_dir = RESULTS_DIR / "tensorboard" / stem
    tb_handler = torch.profiler.tensorboard_trace_handler(str(tb_dir))

    def on_trace_ready(prof: torch.profiler.profile) -> None:
        existing = {p.name for p in tb_dir.iterdir()} if tb_dir.exists() else set()
        tb_handler(prof)
        new_files = [p for p in tb_dir.iterdir() if p.name not in existing]
        if new_files:
            latest = max(new_files, key=lambda p: p.stat().st_mtime)
            chrome_path = RESULTS_DIR / f"{stem}_cycle{prof.step_num}{suffix}"
            shutil.copy2(latest, chrome_path)

    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
        on_trace_ready=on_trace_ready,
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    ) as p:
        for _ in range(10):
            with torch.profiler.record_function("model inference"):
                loop_fn(model, input_ids, PROFILE_STEPS)
            p.step()
    print(p.key_averages().table(sort_by="self_cuda_time_total", row_limit=-1))


def generate_optimized(optimized_trace_name: str) -> float:
    """Build the optimized model, profile it, and return the timed elapsed seconds."""
    model = build_model(torch.bfloat16)
    model = torch.compile(model, mode="reduce-overhead")
    input_ids = get_input_ids()
    for _ in range(3):
        optimized_loop(model, input_ids, PROFILE_STEPS)

    profile(optimized_loop, model, input_ids, optimized_trace_name)
    return time_generation(optimized_loop, model, input_ids, "optimised")


def parse_args() -> argparse.Namespace:
    """Parse CLI args for customizing trace file prefixes."""
    parser = argparse.ArgumentParser(description="HW2 LLM inference profiling.")
    parser.add_argument(
        "--slow-prefix",
        default="v0_slow",
        help="Filename prefix for the slow baseline trace (default: v0_slow).",
    )
    parser.add_argument(
        "--optimized-prefix",
        default="v1_optimized",
        help="Filename prefix for the optimized trace (default: v1_optimized).",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("HW2: LLM Inference Optimization")
    print(f"Model: {MODEL_NAME}")
    print("=" * 60)

    print("\n--- Part 1: Slow baseline ---")
    model = build_model(torch.float32)
    input_ids = get_input_ids()
    profile(slow_loop, model, input_ids, f"{args.slow_prefix}_trace.json")
    slow_elapsed = time_generation(slow_loop, model, input_ids, "Slow")
    del model
    torch.cuda.empty_cache()

    print("\n--- Part 2: Optimized ---")
    optimized_elapsed = generate_optimized(
        optimized_trace_name=f"{args.optimized_prefix}_trace.json"
    )

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    if optimized_elapsed is None or optimized_elapsed <= 0:
        print("generate_optimized() did not return a positive elapsed time; "
              "cannot compute speedup.")
    else:
        speedup = slow_elapsed / optimized_elapsed
        print(f"  Slow:      {slow_elapsed:6.2f}s")
        print(f"  Optimized: {optimized_elapsed:6.2f}s")
        print(f"  Speedup:   {speedup:6.2f}x  (vs V0 slow baseline)")


if __name__ == "__main__":
    main()


# ============================================================================
# Writeup
# ============================================================================
#
# Changes made and speedup per fix: (see the logs under results/h100/results)
# logs for best result 'v7b_better_compile_flags.txt'

# | # | Change | Wall-clock speedup vs baseline |
# |---|--------|-------------------------------|
# | 1 | `bfloat16` instead of `float32` | ~4x |
# | 2 | Pre-allocated output tensor, no `.item()` per step | ~5.0x |
# | 3 | KV cache via `use_cache=True` + `past_key_values` | ~5.7x |
# | 4 | `logits_to_keep=1` (don't materialize all-token logits) | ~6.2x |
# | 5 | `torch.compile(model)` (default mode) | ~6.4x |
# | 6 | `torch.compile(model, mode="reduce-overhead")` + `StaticCache` | ~15x |

# Biggest impact and why:
# The biggest impact is item 6, using reduce-overhead in torch.compile
#  `torch.compile(model, mode="reduce-overhead")`
# After items 1–5, the v6 profile showed:
# - `Self CPU time total: 80ms`
# - `Self CUDA time total: 12ms`
#
# That ~7:1 ratio meant the GPU was idle most of the time waiting for the CPU to
# launch the next kernel — a classic CPU-bound decoder. The remaining wall-clock
# gap was almost entirely Python/dispatcher overhead, not compute.
# By using the reduce-overhead option, we make use of CUDA graphs, which capture a fixed sequence of
# kernel launches once and then replay them with effectively zero CPU overhead.
# so we can see that for optimisation, we should not forget about the cpu
