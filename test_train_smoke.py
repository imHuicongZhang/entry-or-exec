"""Plumbing tests for the GRPO step.

A randomly initialised model never solves Countdown, so running train_smoke.py on the tiny
backend gives every rollout reward 0, every advantage 0, and a gradient that is exactly zero --
which proves nothing about the optimizer path. These tests monkeypatch `classify` so that a
group contains both outcomes, and then check that a gradient actually flows through
HFLM.seq_logp_torch into the LoRA parameters.

The reward used by train_smoke.py itself is untouched: it is binary protocol.classify ==
'success'. The patch lives in the test only.
"""
import numpy as np
import pytest
import torch

import backends
import train_smoke as ts
from make_instances import analyse
from protocol import Config, Problem

CFG = Config()


def fixture_batch():
    """Two problems plus their instance records, built the same way make_instances.py builds them."""
    out = []
    for nums in [(2, 3, 4, 6), (1, 1, 2, 6)]:
        rec = analyse(nums, CFG)
        assert rec is not None
        out.append((Problem(tuple(sorted(nums)), CFG), rec))
    return out


def tiny_trainer(kl_coef=0.0, lr=1e-5, weight_decay=0.0):
    # weight_decay must be passed: torch.optim.AdamW defaults it to 1e-2, which moves every
    # LoRA parameter on a step whose gradient is exactly zero. train_smoke.py passes it too.
    lm, _ = backends.build("tiny", seed=0)
    trainable, desc = ts.wrap_lora(lm, "tiny")
    opt = torch.optim.AdamW(trainable, lr=lr, weight_decay=weight_decay)
    return lm, trainable, opt, desc


def split_reward(monkeypatch, frac_success=0.5):
    """Make classify() call a deterministic half of all strings 'success'."""
    def fake(seq, prob):
        return "success" if (hash(seq) % 100) / 100.0 < frac_success else "semantic_fail"
    monkeypatch.setattr(ts, "classify", fake)


def test_lora_rank_and_trainable_params():
    _, trainable, _, desc = tiny_trainer()
    assert desc["lora_rank"] == 16
    assert desc["trainable_params"] > 0 and desc["trainable_fraction"] < 1.0
    assert all(p.requires_grad for p in trainable)


def test_degenerate_group_gives_zero_gradient():
    """All eight rollouts share a reward -> advantage 0 -> no update. Documents the tiny backend."""
    lm, trainable, opt, _ = tiny_trainer()
    before = [p.detach().clone() for p in trainable]
    row = ts.step_once(lm, fixture_batch(), np.random.default_rng(0), opt, trainable,
                       kl_coef=0.0, n_rollouts=8, max_grad_norm=1.0)
    assert row["reward_mean"] == 0.0
    assert row["grad_norm"] == 0.0
    assert all(torch.equal(a, b) for a, b in zip(before, trainable)), (
        "with weight_decay=0 a zero-advantage step must be a no-op")


def test_adamw_default_weight_decay_would_move_params():
    """Why train_smoke.py passes weight_decay explicitly: the torch default is 1e-2, not 0."""
    lm, trainable, opt, _ = tiny_trainer(lr=1e-5, weight_decay=1e-2)
    before = [p.detach().clone() for p in trainable]
    row = ts.step_once(lm, fixture_batch(), np.random.default_rng(0), opt, trainable,
                       kl_coef=0.0, n_rollouts=8, max_grad_norm=1.0)
    assert row["grad_norm"] == 0.0
    assert any(not torch.equal(a, b) for a, b in zip(before, trainable))


def test_gradient_flows_and_params_move(monkeypatch):
    split_reward(monkeypatch)
    lm, trainable, opt, _ = tiny_trainer(lr=1e-3)
    before = [p.detach().clone() for p in trainable]
    row = ts.step_once(lm, fixture_batch(), np.random.default_rng(0), opt, trainable,
                       kl_coef=0.0, n_rollouts=8, max_grad_norm=1.0)
    assert 0.0 < row["reward_mean"] < 1.0, "the patched reward must split the group"
    assert row["grad_norm"] > 0.0
    assert np.isfinite(row["loss"])
    assert any(not torch.equal(a, b) for a, b in zip(before, trainable))


def test_kl_term_is_zero_at_init_then_positive(monkeypatch):
    """LoRA initialises B at zero, so pi_theta == pi_ref exactly before the first update."""
    split_reward(monkeypatch)
    lm, trainable, opt, _ = tiny_trainer(lr=1e-2)
    r0 = ts.step_once(lm, fixture_batch(), np.random.default_rng(0), opt, trainable,
                      kl_coef=0.01, n_rollouts=8, max_grad_norm=1.0)
    assert r0["kl"] == pytest.approx(0.0, abs=1e-12)
    r1 = ts.step_once(lm, fixture_batch(), np.random.default_rng(1), opt, trainable,
                      kl_coef=0.01, n_rollouts=8, max_grad_norm=1.0)
    assert r1["kl"] > 0.0, "k3 is non-negative and must be positive once the adapter has moved"


def test_kl_coef_zero_skips_the_term(monkeypatch):
    split_reward(monkeypatch)
    lm, trainable, opt, _ = tiny_trainer()
    row = ts.step_once(lm, fixture_batch(), np.random.default_rng(0), opt, trainable,
                       kl_coef=0.0, n_rollouts=8, max_grad_norm=1.0)
    assert row["kl"] is None
    assert row["loss"] == pytest.approx(row["loss_pg"])


def test_entry_logging_is_complete_and_consistent():
    """Rollout counts must cover every rollout, and 'entered' must agree with the counts."""
    lm, trainable, opt, _ = tiny_trainer()
    batch = fixture_batch()
    row = ts.step_once(lm, batch, np.random.default_rng(0), opt, trainable,
                       kl_coef=0.0, n_rollouts=8, max_grad_norm=1.0)
    assert sum(row["outcome_counts"].values()) == row["n_rollouts"] == 16
    for (prob, rec) in batch:
        st = row["per_problem"][prob.key]
        assert sum(st["rollouts_per_entry"].values()) == 8     # nothing dropped
        assert set(st["entered_tracked_entry"]) == set(rec["solvable_entries"])
        for e, n in st["rollouts_per_tracked_entry"].items():
            assert st["entered_tracked_entry"][e] == (n > 0)
            assert n == st["rollouts_per_entry"].get(e, 0)
