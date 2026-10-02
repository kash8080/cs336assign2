"""End-to-end benchmarking of the forward pass, backward pass, and optimizer step.

Examples:
    uv run python -m cs336_systems.benchmarking --size small
    uv run python -m cs336_systems.benchmarking --size small medium large --mode train
    uv run python -m cs336_systems.benchmarking --size small --warmup-steps 0
    uv run python -m cs336_systems.benchmarking --size small --mixed-precision bf16
    uv run python -m cs336_systems.benchmarking --d-model 512 --d-ff 2048 --num-layers 4 --num-heads 8
    nsys profile --trace=cuda,nvtx --capture-range=nvtx --nvtx-capture=measure --capture-range-end=stop \
        python -m cs336_systems.benchmarking --size small --context-length 512 --annotate-attention
    uv run python -m cs336_systems.benchmarking --size xl --mode train --steps 1 --memory-profile
"""

import argparse
import contextlib
import math
import timeit

import pandas as pd
import torch
import torch.cuda.nvtx as nvtx

import cs336_basics.MultiHeadSelfAttention as mhsa
from cs336_basics.AdamW import AdamW
from cs336_basics.CrossEntropy import cross_entropy
from cs336_basics.Softmax import softmax
from cs336_basics.TransformerLM import TransformerLM

# Table 1 of the handout (Section 2.1.2).
MODEL_SIZES = {
    "small": {"d_model": 768, "d_ff": 3072, "num_layers": 12, "num_heads": 12},
    "medium": {"d_model": 1024, "d_ff": 4096, "num_layers": 24, "num_heads": 16},
    "large": {"d_model": 1280, "d_ff": 5120, "num_layers": 36, "num_heads": 20},
    "xl": {"d_model": 2560, "d_ff": 10240, "num_layers": 32, "num_heads": 32},
    "10B": {"d_model": 4608, "d_ff": 12288, "num_layers": 50, "num_heads": 36},
}

# Which phases run in each mode.
MODES = {
    "forward": ("forward",),
    "forward_backward": ("forward", "backward"),
    "train": ("forward", "backward", "optimizer"),
}

# Autocast dtypes selectable with --mixed-precision.
AUTOCAST_DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16}


def resolve_device(requested: str) -> str:
    """Pick a device. "auto" prefers CUDA, then MPS, then CPU."""
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def synchronize(device: str) -> None:
    """Block until all queued GPU work has finished, so the timer measures real execution."""
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    elif device.startswith("mps"):
        torch.mps.synchronize()


def nvtx_range(name: str, device: str):
    """NVTX range on CUDA (visible in nsys); no-op elsewhere since non-CUDA builds lack NVTX."""
    return nvtx.range(name) if device.startswith("cuda") else contextlib.nullcontext()


@contextlib.contextmanager
def record_memory_history(path: str | None):
    """Record CUDA allocations inside the block and save a snapshot for https://pytorch.org/memory_viz.

    No-op if path is None. The snapshot is written even if the block raises (e.g. OOM), which is
    usually when you want it most.
    """
    if path is None:
        yield
        return
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.memory._record_memory_history(max_entries=1_000_000)
    try:
        yield
    finally:
        torch.cuda.memory._dump_snapshot(path)
        torch.cuda.memory._record_memory_history(enabled=None)
        print(f"  peak memory {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB, snapshot saved to {path}")


@nvtx.range("scaled dot product attention")
def annotated_scaled_dot_product_attention(Q, K, V, mask=None):
    """Same math as cs336_basics' scaled_dot_product_attention, split into NVTX ranges (CUDA only)."""
    with nvtx.range("computing attention scores"):
        scores = Q @ K.transpose(-2, -1) / math.sqrt(Q.shape[-1])
        if mask is not None:
            scores = scores.masked_fill(~mask, -torch.inf)
    with nvtx.range("computing softmax"):
        probs = softmax(scores, dim=-1)
    with nvtx.range("final matmul"):
        return probs @ V


def run_step(model, optimizer, x, y, mode: str, device: str, mixed_precision: str | None = None) -> dict[str, float]:
    """Run one step and return the wall-clock time (seconds) of each phase that ran."""
    phases = MODES[mode]
    times = {}
    # Only the forward pass and loss run under autocast; backward reuses the forward dtypes.
    autocast = torch.autocast(
        device_type=torch.device(device).type,
        dtype=AUTOCAST_DTYPES.get(mixed_precision),
        enabled=mixed_precision is not None,
    )

    if "optimizer" in phases:
        optimizer.zero_grad(set_to_none=True)
    elif "backward" in phases:
        model.zero_grad(set_to_none=True)

    # Forward-only runs without autograd (inference); the other modes build the graph.
    start = timeit.default_timer()
    with nvtx_range("forward", device), torch.set_grad_enabled("backward" in phases), autocast:
        logits = model(x)
        loss = cross_entropy(logits, y)
        synchronize(device)
    times["forward"] = timeit.default_timer() - start

    if "backward" in phases:
        start = timeit.default_timer()
        with nvtx_range("backward", device):
            loss.backward()
            synchronize(device)
        times["backward"] = timeit.default_timer() - start

    if "optimizer" in phases:
        start = timeit.default_timer()
        with nvtx_range("optimizer", device):
            optimizer.step()
            synchronize(device)
        times["optimizer"] = timeit.default_timer() - start

    times["total"] = sum(times.values())
    return times


def benchmark(args: argparse.Namespace, name: str, config: dict[str, int], device: str) -> list[dict]:
    """Benchmark one model config. Returns one row per phase with mean/std in ms."""
    model = TransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        rope_theta=args.rope_theta,
        **config,
    ).to(device)
    model.train()
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01, betas=(0.9, 0.999), eps=1e-8)

    # Random batch of data.
    x = torch.randint(0, args.vocab_size, (args.batch_size, args.context_length), device=device)
    y = torch.randint(0, args.vocab_size, (args.batch_size, args.context_length), device=device)

    # Under nsys, --nvtx-capture=measure records only the timed steps and skips the warmup.
    with nvtx_range("warmup", device):
        for _ in range(args.warmup_steps):
            run_step(model, optimizer, x, y, args.mode, device, args.mixed_precision)

    # Memory history covers only the timed steps, so the snapshot isn't padded with warmup.
    snapshot_path = None
    if args.memory_profile is not None:
        precision = args.mixed_precision or "fp32"
        snapshot_path = f"{args.memory_profile}_{name}_ctx{args.context_length}_{args.mode}_{precision}.pickle"

    with nvtx_range("measure", device), record_memory_history(snapshot_path):
        steps = [run_step(model, optimizer, x, y, args.mode, device, args.mixed_precision) for _ in range(args.steps)]

    timings = pd.DataFrame(steps) * 1e3  # seconds -> ms
    num_params = sum(p.numel() for p in model.parameters())
    return [
        {
            "size": name,
            "params_M": round(num_params / 1e6, 1),
            "phase": phase,
            "mean_ms": timings[phase].mean(),
            "std_ms": timings[phase].std(ddof=1) if args.steps > 1 else 0.0,
            "min_ms": timings[phase].min(),
            "max_ms": timings[phase].max(),
        }
        for phase in timings.columns
    ]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Benchmark forward / backward / optimizer step of a TransformerLM.")

    # --- model ---
    p.add_argument("--size", nargs="+", choices=list(MODEL_SIZES), default=None, help="Preset model size(s) from Table 1. Overrides the --d-model etc. flags.")
    p.add_argument("--d-model", type=int, default=768)
    p.add_argument("--d-ff", type=int, default=3072)
    p.add_argument("--num-layers", type=int, default=12)
    p.add_argument("--num-heads", type=int, default=12)
    p.add_argument("--vocab-size", type=int, default=10_000)
    p.add_argument("--context-length", type=int, default=128)
    p.add_argument("--rope-theta", type=float, default=10_000.0)

    # --- data / optimizer ---
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-3)

    # --- benchmarking ---
    p.add_argument("--mode", choices=list(MODES), default="train", help="forward: forward only (no grad). forward_backward: forward + backward. train: forward + backward + optimizer step.")
    p.add_argument("--warmup-steps", type=int, default=5, help="Untimed steps run before measuring.")
    p.add_argument("--steps", type=int, default=10, help="Number of timed steps.")
    p.add_argument("--device", default="auto", help="auto, cuda, mps, or cpu.")
    p.add_argument("--mixed-precision", choices=list(AUTOCAST_DTYPES), default=None, help="Run the forward pass and loss under autocast with this dtype. Off (full FP32) if omitted.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--latex", action="store_true", help="Also print the results as a LaTeX table.")
    p.add_argument("--annotate-attention", action="store_true", help="Swap in the NVTX-annotated attention so nsys splits scores / softmax / final matmul (CUDA only).")
    p.add_argument("--memory-profile", nargs="?", const="memory_snapshot", default=None, metavar="PREFIX", help="Record CUDA memory history over the timed steps and save PREFIX_<size>_ctx<len>_<mode>_<precision>.pickle for pytorch.org/memory_viz (CUDA only). PREFIX defaults to memory_snapshot.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # It fixes PyTorch's random number generator to a known starting point (seed 0 by default), 
    # so every run of the script produces the same "random" values.
    torch.manual_seed(args.seed)
    device = resolve_device(args.device)
    if args.memory_profile is not None and not device.startswith("cuda"):
        raise SystemExit(f"--memory-profile needs CUDA, got device={device}")

    # Patch the name MultiHeadSelfAttention looks up: it did `from ... import scaled_dot_product_attention`,
    # so patching cs336_basics.ScaledDotProductAttention would not reach it.
    if args.annotate_attention:
        mhsa.scaled_dot_product_attention = annotated_scaled_dot_product_attention

    if args.size is not None:
        configs = {name: MODEL_SIZES[name] for name in args.size}
    else:
        configs = {"custom": {"d_model": args.d_model, "d_ff": args.d_ff, "num_layers": args.num_layers, "num_heads": args.num_heads}}

    print(f"device={device} mode={args.mode} mixed_precision={args.mixed_precision} warmup_steps={args.warmup_steps} steps={args.steps} batch_size={args.batch_size} context_length={args.context_length}")

    rows = []
    for name, config in configs.items():
        print(f"benchmarking {name}: {config}")
        try:
            rows.extend(benchmark(args, name, config, device))
        except RuntimeError as e:  # e.g. out of memory on the larger sizes
            print(f"  skipped {name}: {e}")
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
        elif device.startswith("mps"):
            torch.mps.empty_cache()

    if not rows:
        return
    results = pd.DataFrame(rows)
    print()
    print(results.to_string(index=False, float_format=lambda v: f"{v:.2f}"))
    if args.latex:
        print()
        print(results.to_latex(index=False, float_format="%.2f"))


if __name__ == "__main__":
    main()
