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
# Changes made and speedup per fix:
# 1. loading model in bfloat16 instead of float32 4.2x vs baseline (or if
#  we only look at CUDA total time, 4.8x
# 2 preallocating output tensor and removing the .item() call:  5x vs baseline
# compared to with only bfloat16, in terms of CUDA time, ~1.08x
# 3. making use of kv cache: 5.7x vs baseline (wall clock time), comparing only cuda time to v2,
#  4x times  (ie if we only conisider CUDA time, adding all 3 fixes gives us ~20 speedup in terms
# of cuda time compared to baseline
# 4. using logits_to_keep=1 argument -- The Hugging Face transformers documentation states that only
# the last token's logits are needed for generation, but the deafult is 0, ie using all the token logits
# 5. add torch.compile
#6.  add torch.backends.cuda.enable_flash_sdp(True)
# Biggest impact and why:
#
