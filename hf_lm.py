"""HuggingFace causal-LM backend for Policy A.

One symbol = one token id. The model context is prompt_ids followed by the symbol token ids,
one id per symbol, regardless of how the tokenizer would merge the raw string. All conclusions
therefore concern this restricted token-level policy only.

`seq_logp_torch` is the differentiable log-prob to be used by GRPO (log pi, importance ratio,
reference-policy log-prob). It uses the same mask function as the numpy scorer.
"""
import numpy as np
import torch
from protocol import SYMS, SID, V, allowed_mask, render_prompt


class HFLM:
    def __init__(self, model, prompt_ids_fn, sym_token_ids, pad_id=0, device="cpu", batch_size=64):
        assert len(sym_token_ids) == V and len(set(sym_token_ids)) == V
        self.model, self.prompt_ids_fn = model, prompt_ids_fn
        self.sym = torch.tensor(sym_token_ids, device=device)
        self.pad_id, self.device, self.bs = pad_id, device, batch_size

    def _forward(self, prob, seqs):
        """-> symbol logits [B, Lmax+1, V] (row t follows prompt+seq[:t]); right padding."""
        p = list(self.prompt_ids_fn(prob))
        L = max(len(s) for s in seqs)
        ids = torch.full((len(seqs), len(p) + L), self.pad_id, dtype=torch.long, device=self.device)
        att = torch.zeros_like(ids)
        for i, s in enumerate(seqs):
            row = p + [int(self.sym[SID[c]]) for c in s]
            ids[i, :len(row)] = torch.tensor(row, device=self.device)
            att[i, :len(row)] = 1
        logits = self.model(input_ids=ids, attention_mask=att).logits
        return logits[:, len(p) - 1:, :].index_select(-1, self.sym)

    @torch.no_grad()
    def pos_logits(self, prob, seqs):
        out = []
        for i in range(0, len(seqs), self.bs):
            chunk = seqs[i:i + self.bs]
            lg = self._forward(prob, chunk).double().cpu().numpy()
            out += [lg[j, :len(s) + 1] for j, s in enumerate(chunk)]
        return out

    def _positions(self, prob, seqs, B, T):
        """Grammar mask, target symbol and used-position arrays for a [B, T] logit block.

        The single place allowed_mask enters the torch path: seq_logp_torch and kl_to_ref_torch
        both go through here, so they cannot drift apart.
        """
        mask = np.zeros((B, T, V), dtype=bool)
        tgt = np.zeros((B, T), dtype=np.int64)
        use = np.zeros((B, T), dtype=bool)
        for i, s in enumerate(seqs):
            for t, c in enumerate(s):
                mask[i, t] = allowed_mask(s[:t], prob.cfg)
                tgt[i, t], use[i, t] = SID[c], True
        mask[~use] = True  # unused positions: any finite value, zeroed by `use` downstream
        return mask, tgt, use

    def pos_logp_torch(self, prob, seqs):
        """Differentiable Policy A log-probs at every position: (lp [B, T, V], m, use).

        lp[i, t] is log pi(. | prompt + seqs[i][:t]) renormalised over the allowed symbols, so
        lp[i, t] is -inf exactly where m[i, t] is False. `use[i, t]` says the position is a real
        symbol of seqs[i] rather than right padding.
        """
        lg = self._forward(prob, seqs).double()
        B, T, _ = lg.shape
        mask, _, use = self._positions(prob, seqs, B, T)
        m = torch.from_numpy(mask).to(lg.device)
        lp = torch.log_softmax(lg.masked_fill(~m, float("-inf")), dim=-1)
        return lp, m, torch.from_numpy(use).to(lg.device)

    def seq_logp_torch(self, prob, seqs):
        """Differentiable log pi(seq) under Policy A, shape [B]."""
        lg = self._forward(prob, seqs).double()
        B, T, _ = lg.shape
        mask, tgt, use = self._positions(prob, seqs, B, T)
        m = torch.from_numpy(mask).to(lg.device)
        lp = torch.log_softmax(lg.masked_fill(~m, float("-inf")), dim=-1)
        tok = lp.gather(-1, torch.from_numpy(tgt).to(lg.device)[..., None])[..., 0]
        return (tok * torch.from_numpy(use).to(lg.device)).sum(-1)

    def kl_to_ref_torch(self, prob, seqs, ref_context):
        """Exact per-position KL(pi_theta || pi_ref), summed over positions, shape [B].

        At every position of every sequence both policies are evaluated under the same
        allowed_mask and the same masked log-softmax, and the KL is the exact sum over the allowed
        symbols,

            sum_v pi_theta(v | h) * (log pi_theta(v | h) - log pi_ref(v | h)),

        summed over the positions of the sequence. The mean over sequences is taken by the caller,
        exactly as for seq_logp_torch.

        This is the gradient of the stated objective. Estimators that treat the sampled symbols as
        fixed are not: `log pi_theta(s) - log pi_ref(s)` has zero expected gradient, and the k3
        form `exp(r) - r - 1` with `r = log pi_ref - log pi_theta` has expected gradient
        `d KL(pi_ref || pi_theta)`, the wrong direction. For pi_theta = (0.8, 0.2) and
        pi_ref = (0.5, 0.5) the gradient on the first logit is 0.221807; those two give 0.3 and 0.
        test_kl.py pins this.

        `ref_context` is a zero-argument callable returning a context manager under which
        self.model evaluates as the reference policy -- peft's `disable_adapter` for LoRA.
        """
        lp, m, use = self.pos_logp_torch(prob, seqs)
        with torch.no_grad(), ref_context():
            lp_ref = self.pos_logp_torch(prob, seqs)[0].detach()
        # Disallowed symbols hold -inf in both tensors. Fill before subtracting so that inf - inf
        # is never evaluated: pi_theta is exactly 0 there, so the term is 0 either way, but a nan
        # would otherwise reach the backward pass.
        p = torch.exp(lp).masked_fill(~m, 0.0)
        d = lp.masked_fill(~m, 0.0) - lp_ref.masked_fill(~m, 0.0)
        return ((p * d).sum(-1) * use.to(lp.dtype)).sum(-1)


def load_qwen(name="Qwen/Qwen2.5-1.5B-Instruct", device="cuda", dtype=torch.float32,
              cache_dir=None):
    """NOT RUN in the authoring environment (no GPU, no model hub access). Verify on first use.

    dtype defaults to float32. Policy A multiplies up to 27 masked conditionals, so rounding the
    symbol logits to bfloat16's 8 mantissa bits shows up directly in log pi(seq) and therefore in
    e; check_real_model.py measures that difference on the real model. bfloat16 remains available
    by passing dtype=torch.bfloat16, which is the right choice for throughput once a run only
    needs p_success to one significant figure.

    cache_dir is passed to both from_pretrained calls so the weight and tokenizer cache location
    is explicit rather than dependent on HF_HOME having been set before transformers was imported.
    """
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tok = AutoTokenizer.from_pretrained(name, cache_dir=cache_dir)
    model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=dtype,
                                                cache_dir=cache_dir).to(device).eval()
    sym_ids = []
    for s in SYMS:
        ids = tok.encode(s, add_special_tokens=False)
        assert len(ids) == 1, f"symbol {s!r} is not a single token: {ids}"
        sym_ids.append(ids[0])

    def prompt_ids(prob):
        msgs = [{"role": "user", "content": render_prompt(prob)}]
        return tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True)

    return HFLM(model, prompt_ids, sym_ids, pad_id=tok.pad_token_id or 0, device=device)
