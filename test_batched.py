"""The batched tree walk must agree with the frozen recursive one.

This is the only guarantee that check_real_model.py may substitute batched.mass_conservation_batched
for policy.mass_conservation when running on a real model.
"""
import numpy as np
import pytest

from batched import mass_conservation_batched
from policy import MockLM, mass_conservation, measure
from protocol import Config, Problem


@pytest.mark.parametrize("nums,seed", [((2, 3, 4, 6), 1), ((1, 1, 5, 5), 2), ((3, 3, 8, 8), 3)])
def test_batched_matches_recursive(nums, seed):
    prob, lm = Problem(nums), MockLM(seed=seed)
    ref = mass_conservation(lm, prob)
    s, f, o, stats = mass_conservation_batched(lm, prob, batch_size=64)
    assert np.abs(np.array([s, f, o]) - np.array(ref)).max() < 1e-12
    assert abs(s + f + o - 1) < 1e-9
    assert o == 0.0
    assert abs(s - measure(lm, prob)["p_success_total"]) < 1e-9
    assert stats["lm_calls"] < stats["nodes"]  # the point of batching


@pytest.mark.parametrize("bs", [1, 7, 10_000])
def test_batched_independent_of_batch_size(bs):
    prob, lm = Problem((2, 3, 4, 6)), MockLM(seed=1)
    got = mass_conservation_batched(lm, prob, batch_size=bs)[:3]
    ref = mass_conservation_batched(lm, prob, batch_size=256)[:3]
    assert np.abs(np.array(got) - np.array(ref)).max() < 1e-15


def test_batched_small_grammar_sums_to_one():
    prob = Problem((2, 3), Config(n_numbers=2, target=6, digits="0236"))
    s, f, o, _ = mass_conservation_batched(MockLM(seed=3), prob)
    assert abs(s + f + o - 1) < 1e-9
