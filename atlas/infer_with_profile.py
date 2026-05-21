#!/usr/bin/env python3
"""Profile a small LLM with Nsight-friendly NVTX ranges.

The default path runs an explicit prefill forward pass followed by a
token-by-token decode loop. That makes Nsight Systems timelines easier to
read than a single opaque `model.generate()` range.
"""

import argparse
import time


def nvtx_range(torch, name):
    """Context manager wrapping an NVTX range so Nsight labels phases."""
    class _Range:
        def __enter__(self_):
            torch.cuda.nvtx.range_push(name)
            return self_
        def __exit__(self_, *a):
            torch.cuda.nvtx.range_pop()
    return _Range()


def load_model(model_name, dtype):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
    ).to("cuda").eval()
    return tok, model


def manual_generate(torch, model, inputs, max_new_tokens):
    """Greedy decode with explicit prefill/decode NVTX ranges."""
    input_ids = inputs["input_ids"]
    attention_mask = inputs.get("attention_mask")
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids, device=input_ids.device)

    generated = []

    with nvtx_range(torch, "prefill"):
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=True,
        )
    past = outputs.past_key_values
    next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)
    generated.append(next_token)

    with nvtx_range(torch, "decode_loop"):
        for _ in range(max_new_tokens - 1):
            attention_mask = torch.cat(
                [attention_mask, torch.ones_like(next_token, device=input_ids.device)],
                dim=-1,
            )
            with nvtx_range(torch, "decode_token"):
                outputs = model(
                    input_ids=next_token,
                    attention_mask=attention_mask,
                    past_key_values=past,
                    use_cache=True,
                )
            past = outputs.past_key_values
            next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)
            generated.append(next_token)

    return torch.cat([input_ids] + generated, dim=-1)


def hf_generate(torch, model, inputs, max_new_tokens):
    with nvtx_range(torch, "generate_full"):
        return model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B")
    ap.add_argument("--prompt", default="Explain how attention works in transformers.")
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--dtype", default="fp16", choices=["fp16", "bf16", "fp32"])
    ap.add_argument("--hf-generate", action="store_true",
                    help="use Hugging Face generate() instead of explicit prefill/decode")
    args = ap.parse_args()

    import torch

    assert torch.cuda.is_available(), "No CUDA device found."
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[args.dtype]

    print(f"Loading {args.model} ({args.dtype}) on {torch.cuda.get_device_name(0)} ...")
    tok, model = load_model(args.model, dtype)

    inputs = tok(args.prompt, return_tensors="pt").to("cuda")

    with torch.no_grad():
        if args.hf_generate:
            _ = hf_generate(torch, model, inputs, 8)
        else:
            _ = manual_generate(torch, model, inputs, 8)
    torch.cuda.synchronize()

    torch.cuda.cudart().cudaProfilerStart()

    with torch.no_grad(), nvtx_range(torch, "profiled_generation"):
        t0 = time.perf_counter()
        if args.hf_generate:
            out = hf_generate(torch, model, inputs, args.max_new_tokens)
        else:
            out = manual_generate(torch, model, inputs, args.max_new_tokens)
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0

    torch.cuda.cudart().cudaProfilerStop()

    n_new = out.shape[1] - inputs["input_ids"].shape[1]
    print(f"\nGenerated {n_new} tokens in {dt*1e3:.1f} ms "
          f"({n_new/dt:.1f} tok/s)")
    print(f"Peak GPU mem: {torch.cuda.max_memory_allocated()/1e9:.2f} GB")
    print("\n--- Output ---")
    print(tok.decode(out[0], skip_special_tokens=True))


if __name__ == "__main__":
    main()
