"""Unit tests a-g plus the termination / truncation mass-conservation test.

Backends here are a deterministic mock LM and a tiny randomly initialised GPT-2.
They validate the measurement code, not any property of a real LLM.
"""
import itertools
import numpy as np
import pytest
from scipy import stats
from protocol import (Config, Problem, SYMS, NL, allowed_symbols, allowed_mask, is_terminal,
                      classify, success_trajectories, valid_lines, canonical_event)
from policy import MockLM, TabularLM, seq_logp, next_logp, sample, measure, mass_conservation

P4 = Problem((2, 3, 4, 6))
MOCK = MockLM(seed=1)


def all_complete(cfg):
    """Every grammar-complete string, by exhaustive expansion."""
    out, stack = [], [""]
    while stack:
        p = stack.pop()
        if is_terminal(p, cfg):
            out.append(p)
            continue
        assert len(p) < cfg.max_len
        _, line = p.rsplit(NL, 1) if NL in p else ("", p)
        stack += [p + s for s in allowed_symbols(line, cfg)]
    return out


# a. probabilities of all complete sequences sum to 1 (exhaustive, small grammar)
def test_a_full_enumeration_sums_to_one():
    prob = Problem((2, 3), Config(n_numbers=2, target=6, digits="0236"))
    seqs = all_complete(prob.cfg)
    assert len(seqs) == len(set(seqs)) > 1000
    total = np.exp(seq_logp(MockLM(seed=3), prob, seqs)).sum()
    assert abs(total - 1) < 1e-4
    assert abs(total - 1) < 1e-9  # float64 mock should be far tighter


# b. sampler entry frequencies match exact entry probabilities
def test_b_sampler_matches_entry_probs():
    m = measure(MOCK, P4)
    names = list(m["entries"])
    p = np.array([m["entries"][k]["p_entry"] for k in names] + [m["p_first_line_semantically_wrong"]])
    seqs = sample(MOCK, P4, 20000, np.random.default_rng(0))
    idx = {k: i for i, k in enumerate(names)}
    cnt = np.zeros(len(p))
    for s in seqs:
        cnt[idx.get(s.split(NL)[0], len(names))] += 1
    keep = p * 20000 >= 5
    obs = np.append(cnt[keep], cnt[~keep].sum())
    exp = np.append(p[keep], p[~keep].sum()) * 20000
    if exp[-1] == 0:
        obs, exp = obs[:-1], exp[:-1]
    assert stats.chisquare(obs, exp).pvalue > 1e-3
    succ = np.mean([classify(s, P4) == "success" for s in seqs])
    se = np.sqrt(m["p_success_total"] * (1 - m["p_success_total"]) / 20000)
    assert abs(succ - m["p_success_total"]) < 4 * se


# c. forced-entry conditional sampling matches exact e, including a low-probability entry
def test_c_forced_entry_completion():
    m = measure(MOCK, P4)
    solv = sorted((k for k, v in m["entries"].items() if v["solvable"]),
                  key=lambda k: m["entries"][k]["p_entry"])
    for entry in (solv[0], solv[-1]):
        e = m["entries"][entry]["e"]
        seqs = sample(MOCK, P4, 2000, np.random.default_rng(1), prefix=entry + NL)
        hat = np.mean([classify(s, P4) == "success" for s in seqs])
        assert abs(hat - e) < 4 * np.sqrt(max(e * (1 - e), 1e-4) / 2000)


def tiny_hf():
    torch = pytest.importorskip("torch")
    tr = pytest.importorskip("transformers")
    from hf_lm import HFLM
    torch.manual_seed(0)
    model = tr.GPT2LMHeadModel(tr.GPT2Config(vocab_size=64, n_positions=64, n_embd=32,
                                             n_layer=2, n_head=2)).eval()
    prompt = lambda prob: [1] + [30 + n for n in prob.numbers] + [2]
    return HFLM(model, prompt, list(range(10, 26)), pad_id=0)


# d. differentiable (training) log-prob equals the teacher-forced scorer
def test_d_train_logp_equals_scorer():
    lm = tiny_hf()
    seqs = success_trajectories(P4)[:6] + ["6/2=3\n3+3=6\n", "9", "12*34=56\n7-7=0\n0+0=0\n"]
    a = seq_logp(lm, P4, seqs)
    b = lm.seq_logp_torch(P4, seqs)
    assert b.requires_grad
    assert np.abs(a - b.detach().numpy()).max() < 1e-5


# e. single-sequence and padded-batch computation agree
def test_e_batch_padding_invariance():
    lm = tiny_hf()
    seqs = ["6/2=3\n", "6/2=3\n3+3=6\n6*4=24\n", "4", "12*34=56\n7-7=0\n"]
    batched = seq_logp(lm, P4, seqs)
    single = np.array([seq_logp(lm, P4, [s])[0] for s in seqs])
    assert np.abs(batched - single).max() < 1e-5
    bt = lm.seq_logp_torch(P4, seqs).detach().numpy()
    st = np.array([lm.seq_logp_torch(P4, [s]).item() for s in seqs])
    assert np.abs(bt - st).max() < 1e-5


# f. tabular copy reproduces the LM on every full history
@pytest.mark.parametrize("backend", ["mock", "hf"])
def test_f_tabular_copy(backend):
    lm = MOCK if backend == "mock" else tiny_hf()
    tab = TabularLM(lm)
    hist = set()
    for tr in success_trajectories(P4):
        hist.update(tr[:i] for i in range(len(tr)))
    for s in sample(lm, P4, 300, np.random.default_rng(2)):
        hist.update(s[:i] for i in range(len(s)))
    hist = sorted(hist)
    assert np.abs(np.exp(next_logp(lm, P4, hist)) - np.exp(next_logp(tab, P4, hist))).max() < 1e-6
    ma, mb = measure(lm, P4), measure(tab, P4)
    assert abs(ma["p_success_total"] - mb["p_success_total"]) < 1e-9
    for k in ma["entries"]:
        assert abs(ma["entries"][k]["p_entry"] - mb["entries"][k]["p_entry"]) < 1e-9
    # parameters are keyed by (problem key, full history string), and the set of stored histories
    # is prefix closed: every proper prefix of a stored history is itself stored (the empty
    # history included). Nothing is shared between histories and nothing is merged by state.
    stored = set()
    for k in tab.theta:
        assert isinstance(k, tuple) and len(k) == 2, f"theta key is not a 2-tuple: {k!r}"
        pk, h = k
        assert pk == P4.key, f"theta key holds the wrong problem: {pk!r}"
        assert isinstance(h, str) and set(h) <= set(SYMS), f"history is not a symbol string: {h!r}"
        stored.add(h)
    assert "" in stored
    for h in stored:
        for i in range(len(h)):
            assert h[:i] in stored, f"history {h!r} stored but its prefix {h[:i]!r} is not"


# g. event-level sums: hand-computed, no omission and no double counting
def test_g_hand_computed_events():
    uni = MockLM(uniform=True)
    # "2*3=6\n": 1/10 * 1/14 * 1/10 * 1/11 * 1/10 * 1/11 under the uniform masked policy
    p_str = 1 / (10 * 14 * 10 * 11 * 10 * 11)
    prob = Problem((2, 3), Config(n_numbers=2, target=6))
    assert success_trajectories(prob) == ["2*3=6\n", "3*2=6\n"]
    m = measure(uni, prob)
    assert abs(m["p_success_total"] - 2 * p_str) < 1e-15
    assert sorted(m["events"]["2*3"]["strings"]) == ["2*3=6", "3*2=6"]
    assert abs(m["events"]["2*3"]["p_success"] - 2 * p_str) < 1e-15
    assert abs(m["entries"]["2*3=6"]["e"] - 1.0) < 1e-12
    # duplicate numbers must not create duplicate strings
    dup = Problem((2, 2), Config(n_numbers=2, target=4))
    assert success_trajectories(dup) == ["2*2=4\n", "2+2=4\n"]
    assert abs(measure(uni, dup)["p_success_total"] - 2 * p_str) < 1e-15
    assert canonical_event("6+4=10") == canonical_event("4+6=10") == "4+6"
    assert canonical_event("6-4=2") != canonical_event("4-6=2")


# termination / truncation: every allowed generation ends in exactly one of three outcomes
@pytest.mark.parametrize("nums,seed", [((2, 3, 4, 6), 1), ((1, 1, 5, 5), 2), ((3, 3, 8, 8), 3)])
def test_termination_mass_conservation(nums, seed):
    prob, lm = Problem(nums), MockLM(seed=seed)
    s, f, o = mass_conservation(lm, prob)
    assert abs(s + f + o - 1) < 1e-9
    assert o == 0.0
    assert abs(s - measure(lm, prob)["p_success_total"]) < 1e-9


def test_termination_grammar_bounds():
    cfg, rng = Config(), np.random.default_rng(0)
    assert cfg.max_len == 27
    lens = []
    for _ in range(3000):
        p = ""
        while not is_terminal(p, cfg):
            m = allowed_mask(p, cfg)
            assert m.any() and len(p) < cfg.max_len  # no dead end, no unfinished mass at the cap
            p += SYMS[rng.choice(np.flatnonzero(m))]
        assert not allowed_mask(p, cfg).any()
        lens.append(len(p))
    assert 18 <= min(lens) and max(lens) <= 27
    seqs = sample(MockLM(seed=5, boost=0.0), P4, 500, np.random.default_rng(3))
    assert {classify(s, P4) for s in seqs} <= {"success", "semantic_fail"}
    assert classify("6/2=3\n3+3=6\n", P4) == "overlength"  # unfinished output is a failure class
