"""Breadth-first, batched version of policy.mass_conservation.

policy.mass_conservation asks the LM for one prefix at a time, which is fine for the mock but
very slow for a real model. This walks exactly the same tree, but level by level, so every LM
call covers many prefixes at once. The tree shape is a function of the grammar and the
semantics only -- never of the probabilities -- so the set of nodes visited and the outcome each
leaf is charged to are identical to the recursive version. Only the order in which float64
masses are summed differs, which `test_batched.py` pins to 1e-12.

Nothing is renormalised and nothing is discarded; failure mass is charged at the first symbol
that leaves the set of semantically valid lines, exactly as in the recursive version.
"""
import numpy as np

from policy import next_logp
from protocol import SYMS, NL, apply_line, is_terminal, split_prefix, valid_lines


def mass_conservation_batched(lm, prob, batch_size=256, progress=None):
    """-> (success, semantic_fail, overlength). Same result as policy.mass_conservation."""
    cfg = prob.cfg
    acc = {"success": 0.0, "semantic_fail": 0.0, "overlength": 0.0}
    vl_cache = {}
    n_lm_calls = n_nodes = 0

    frontier = [(tuple(sorted(prob.numbers)), "", 0.0)]
    while frontier:
        live = []
        for node in frontier:
            state, prefix, logp = node
            if len(prefix) >= cfg.max_len and not is_terminal(prefix, cfg):
                acc["overlength"] += np.exp(logp)   # unreachable under Config(), still accounted
            else:
                live.append(node)
        if not live:
            break
        n_nodes += len(live)
        nxt = []
        for i in range(0, len(live), batch_size):
            chunk = live[i:i + batch_size]
            lps = next_logp(lm, prob, [c[1] for c in chunk])
            n_lm_calls += 1
            for (state, prefix, logp), lp in zip(chunk, lps):
                done, line = split_prefix(prefix)
                if state not in vl_cache:
                    vl_cache[state] = valid_lines(state, cfg)
                vl = vl_cache[state]
                for j in np.flatnonzero(np.isfinite(lp)):
                    s, q = SYMS[j], logp + lp[j]
                    if s == NL:
                        if line not in vl:
                            acc["semantic_fail"] += np.exp(q)
                            continue
                        ns = apply_line(state, line, cfg)
                        if done + 1 >= cfg.n_lines:
                            acc["success" if ns == (cfg.target,) else "semantic_fail"] += np.exp(q)
                        else:
                            nxt.append((ns, prefix + s, q))
                    elif any(v.startswith(line + s) and len(v) > len(line) for v in vl):
                        nxt.append((state, prefix + s, q))
                    else:
                        acc["semantic_fail"] += np.exp(q)
        if progress:
            progress(len(nxt), n_nodes)
        frontier = nxt
    return acc["success"], acc["semantic_fail"], acc["overlength"], {"nodes": n_nodes,
                                                                    "lm_calls": n_lm_calls}
