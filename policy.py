"""Sampler, exact teacher-forced scorer, tabular copy, measurement. All use Policy A.

LM interface (any backend):
    lm.pos_logits(prob, seqs) -> list of float arrays [len(seq)+1, V];
        row t = raw logits over SYMS for the symbol after prompt + seq[:t].
"""
import zlib
from collections import defaultdict
import numpy as np
from protocol import (SYMS, SID, V, NL, allowed_mask, masked_log_softmax, is_terminal,
                      classify, success_trajectories, valid_lines, apply_line,
                      canonical_event, split_prefix)


def next_logp(lm, prob, prefixes):
    """Masked log-probs [B, V] for the next symbol after each prefix."""
    rows = np.stack([r[-1] for r in lm.pos_logits(prob, prefixes)])
    masks = np.stack([allowed_mask(p, prob.cfg) for p in prefixes])
    return masked_log_softmax(rows, masks)


def seq_logp(lm, prob, seqs):
    """Exact log-prob of each (possibly partial) symbol string under Policy A."""
    out = []
    for seq, rows in zip(seqs, lm.pos_logits(prob, seqs)):
        lp = 0.0
        for t, s in enumerate(seq):
            lp += masked_log_softmax(rows[t], allowed_mask(seq[:t], prob.cfg))[SID[s]]
        out.append(lp)
    return np.array(out)


def sample(lm, prob, n, rng, prefix=""):
    """n samples continuing `prefix`. Never discards: hitting max_len unfinished = 'overlength'."""
    cfg = prob.cfg
    seqs = [prefix] * n
    live = [i for i in range(n) if not is_terminal(prefix, cfg)]
    while live:
        live = [i for i in live if len(seqs[i]) < cfg.max_len]
        if not live:
            break
        uniq = sorted({seqs[i] for i in live})
        lp = dict(zip(uniq, next_logp(lm, prob, uniq)))
        nxt = []
        for i in live:
            p = np.exp(lp[seqs[i]])
            seqs[i] += SYMS[rng.choice(V, p=p / p.sum())]
            if not is_terminal(seqs[i], cfg):
                nxt.append(i)
        live = nxt
    return seqs


# ---------------------------------------------------------------- backends
class MockLM:
    """Deterministic pseudo-random logits per (problem, history); `boost` favours success prefixes."""

    def __init__(self, seed=0, boost=3.0, uniform=False):
        self.seed, self.boost, self.uniform = seed, boost, uniform
        self._good, self._cache = {}, {}

    def _row(self, prob, prefix):
        k = (prob.key, prefix)
        if k not in self._cache:
            if self.uniform:
                row = np.zeros(V)
            else:
                h = zlib.crc32(f"{self.seed}|{prob.key}|{prefix}".encode())
                row = np.random.default_rng(h).normal(size=V)
                if prob.key not in self._good:
                    g = set()
                    for tr in success_trajectories(prob):
                        g.update(tr[:i] for i in range(1, len(tr) + 1))
                    self._good[prob.key] = g
                for s in SYMS:
                    if prefix + s in self._good[prob.key]:
                        row[SID[s]] += self.boost
            self._cache[k] = row
        return self._cache[k]

    def pos_logits(self, prob, seqs):
        return [np.stack([self._row(prob, s[:t]) for t in range(len(s) + 1)]) for s in seqs]


class TabularLM:
    """Independent logits per full history, initialised from a frozen base LM on first access.

    Parameters live in self.theta[(problem key, history string)]; nothing is shared between
    histories and histories are never merged by remaining-number state.
    """

    def __init__(self, base):
        self.base, self.theta = base, {}

    def _row(self, prob, prefix):
        k = (prob.key, prefix)
        if k not in self.theta:
            self.theta[k] = np.array(self.base.pos_logits(prob, [prefix])[0][-1], dtype=np.float64)
        return self.theta[k]

    def pos_logits(self, prob, seqs):
        return [np.stack([self._row(prob, s[:t]) for t in range(len(s) + 1)]) for s in seqs]


# ---------------------------------------------------------------- exact measurement
def measure(lm, prob):
    """Exact frozen-policy quantities for one problem.

    entry         = one specific first-line string (fixed prefix, ends with newline)
    p_entry       = probability the restricted policy writes that first line
    p_success     = probability mass of successful strings starting with that entry
    e             = p_success / p_entry  (fixed-entry completion: sum over ALL correct suffixes)
    share         = p_success / total success probability
    events        = the same, summed over strings denoting the same canonical first action
    """
    cfg = prob.cfg
    trajs = success_trajectories(prob)
    firsts = sorted(valid_lines(tuple(sorted(prob.numbers)), cfg))
    p_first = dict(zip(firsts, np.exp(seq_logp(lm, prob, [f + NL for f in firsts]))))
    p_traj = np.exp(seq_logp(lm, prob, trajs)) if trajs else np.array([])
    succ = defaultdict(float)
    for tr, p in zip(trajs, p_traj):
        succ[tr.split(NL)[0]] += float(p)
    total = float(sum(succ.values()))
    entries, events = {}, defaultdict(lambda: {"p_entry": 0.0, "p_success": 0.0, "strings": []})
    for f in firsts:
        pe, ps = float(p_first[f]), succ.get(f, 0.0)
        entries[f] = {"p_entry": pe, "p_success": ps, "e": ps / pe if pe > 0 else None,
                      "share": ps / total if total > 0 else None, "solvable": f in succ}
        ev = events[canonical_event(f)]
        ev["p_entry"] += pe
        ev["p_success"] += ps
        ev["strings"].append(f)
    for ev in events.values():
        ev["e"] = ev["p_success"] / ev["p_entry"] if ev["p_entry"] > 0 else None
        ev["share"] = ev["p_success"] / total if total > 0 else None
    p_valid_first = float(sum(p_first.values()))
    return {"problem": prob.key, "p_success_total": total,
            "p_first_line_semantically_wrong": 1.0 - p_valid_first,
            "p_fail_total": 1.0 - total, "p_overlength": 0.0,  # zero by construction of the grammar
            "n_success_strings": len(trajs), "entries": entries, "events": dict(events)}


def mass_conservation(lm, prob):
    """Independent symbol-level walk over every allowed generation.

    Returns (success, semantic_fail, overlength). Failure mass is accumulated explicitly at the
    first symbol that leaves the set of semantically valid lines; nothing is renormalised away.
    """
    cfg = prob.cfg
    acc = {"success": 0.0, "semantic_fail": 0.0, "overlength": 0.0}

    def rec(state, prefix, logp):
        if len(prefix) >= cfg.max_len and not is_terminal(prefix, cfg):
            acc["overlength"] += np.exp(logp)
            return
        done, line = split_prefix(prefix)
        vl = valid_lines(state, cfg)
        lp = next_logp(lm, prob, [prefix])[0]
        for i in np.flatnonzero(np.isfinite(lp)):
            s, q = SYMS[i], logp + lp[i]
            if s == NL:
                if line not in vl:
                    acc["semantic_fail"] += np.exp(q)
                    continue
                ns = apply_line(state, line, cfg)
                if done + 1 >= cfg.n_lines:
                    acc["success" if ns == (cfg.target,) else "semantic_fail"] += np.exp(q)
                else:
                    rec(ns, prefix + s, q)
            elif any(v.startswith(line + s) and len(v) > len(line) for v in vl):
                rec(state, prefix + s, q)
            else:
                acc["semantic_fail"] += np.exp(q)

    rec(tuple(sorted(prob.numbers)), "", 0.0)
    return acc["success"], acc["semantic_fail"], acc["overlength"]
