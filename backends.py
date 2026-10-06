"""Backend construction, shared by measure.py / check_real_model.py / train_smoke.py.

`mock` and `tiny` run anywhere and validate plumbing only. `qwen` is the real model and is only
ever constructed on the cluster; MODEL_PATH and HF_CACHE_DIR come from the paths file.
"""
import os

import paths

CHOICES = ("mock", "tiny", "qwen")


def set_hf_cache(cfg_paths):
    """Point the HuggingFace caches at HF_CACHE_DIR before transformers is imported."""
    cache = str(paths.require_dir(cfg_paths, "HF_CACHE_DIR"))
    os.environ["HF_HOME"] = cache
    os.environ.setdefault("HF_HUB_CACHE", os.path.join(cache, "hub"))
    return cache


def build(name, cfg_paths=None, seed=0, boost=3.0, device=None, batch_size=64, dtype=None):
    """-> (lm, description dict). `qwen` requires a filled-in paths file.

    dtype applies to the `qwen` backend only and defaults to load_qwen's default (float32).
    Pass "bfloat16" or torch.bfloat16 to trade precision in log pi(seq) for throughput.
    """
    if name == "mock":
        from policy import MockLM
        return MockLM(seed=seed, boost=boost), {
            "backend": "mock", "seed": seed, "boost": boost,
            "note": "deterministic pseudo-random logits; validates measurement code only"}

    if name == "tiny":
        # randomly initialised 2-layer GPT-2, built offline, same construction as test_protocol.py
        import torch
        import transformers as tr
        from hf_lm import HFLM
        from protocol import SYMS
        torch.manual_seed(seed)
        model = tr.GPT2LMHeadModel(tr.GPT2Config(vocab_size=64, n_positions=64, n_embd=32,
                                                 n_layer=2, n_head=2)).eval()
        prompt = lambda prob: [1] + [30 + n for n in prob.numbers] + [2]
        lm = HFLM(model, prompt, list(range(10, 10 + len(SYMS))), pad_id=0,
                  device="cpu", batch_size=batch_size)
        return lm, {"backend": "tiny", "seed": seed,
                    "note": "randomly initialised tiny GPT-2; validates plumbing only, "
                            "no property of a real LLM"}

    if name == "qwen":
        if cfg_paths is None:
            raise RuntimeError("the qwen backend needs a paths file")
        cache = set_hf_cache(cfg_paths)
        model_path = paths.require(cfg_paths, "MODEL_PATH")
        import torch
        from hf_lm import load_qwen
        dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        kw = {} if dtype is None else {"dtype": getattr(torch, dtype) if isinstance(dtype, str)
                                      else dtype}
        lm = load_qwen(name=model_path, device=dev, cache_dir=cache, **kw)
        lm.bs = batch_size
        return lm, {"backend": "qwen", "model_path": model_path, "device": dev,
                    "cache_dir": cache,
                    "dtype": str(next(lm.model.parameters()).dtype)}

    raise ValueError(f"unknown backend {name!r}, expected one of {CHOICES}")
