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

    def seq_logp_torch(self, prob, seqs):
        """Differentiable log pi(seq) under Policy A, shape [B]."""
        lg = self._forward(prob, seqs).double()
        B, T, _ = lg.shape
        mask = np.zeros((B, T, V), dtype=bool)
        tgt = np.zeros((B, T), dtype=np.int64)
        use = np.zeros((B, T), dtype=bool)
        for i, s in enumerate(seqs):
            for t, c in enumerate(s):
                mask[i, t] = allowed_mask(s[:t], prob.cfg)
                tgt[i, t], use[i, t] = SID[c], True
        mask[~use] = True  # unused positions: any finite value, zeroed below
        m = torch.from_numpy(mask).to(lg.device)
        lp = torch.log_softmax(lg.masked_fill(~m, float("-inf")), dim=-1)
        tok = lp.gather(-1, torch.from_numpy(tgt).to(lg.device)[..., None])[..., 0]
        return (tok * torch.from_numpy(use).to(lg.device)).sum(-1)


def load_qwen(name="Qwen/Qwen2.5-1.5B-Instruct", device="cuda", dtype=torch.bfloat16):
    """NOT RUN in the authoring environment (no GPU, no model hub access). Verify on first use."""
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=dtype).to(device).eval()
    sym_ids = []
    for s in SYMS:
        ids = tok.encode(s, add_special_tokens=False)
        assert len(ids) == 1, f"symbol {s!r} is not a single token: {ids}"
        sym_ids.append(ids[0])

    def prompt_ids(prob):
        msgs = [{"role": "user", "content": render_prompt(prob)}]
        return tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True)

    return HFLM(model, prompt_ids, sym_ids, pad_id=tok.pad_token_id or 0, device=device)
