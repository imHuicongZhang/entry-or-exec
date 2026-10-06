"""The KL term must be the gradient of the objective it claims to be.

A reviewer showed that the estimators this replaced are not. Both treated the sampled symbols as
fixed, so for pi_theta = (0.8, 0.2) against pi_ref = (0.5, 0.5):

    d KL(pi_theta || pi_ref) / d z_0   = 0.221807   <- what the objective asks for
    E[grad] of k3, exp(r) - r - 1      = 0.3        = d KL(pi_ref || pi_theta) / d z_0
    E[grad] of plain, logp - logp_ref  = 0          (a score function with no baseline)

HFLM.kl_to_ref_torch instead evaluates both policies at every position under the same mask and
sums pi_theta * (log pi_theta - log pi_ref) over the allowed symbols, which is differentiable in
theta directly. These tests pin its autograd gradient to the analytic one, on a toy LM whose
logits are the parameter so the analytic form is available in closed form.
"""
import contextlib
import types

import numpy as np
import pytest
import torch

import train_smoke as ts
from hf_lm import HFLM
from protocol import SID, SYMS, V, Config, Problem, allowed_mask

# A two-symbol first position: Config(digits="01") makes allowed_symbols("") == ["0", "1"].
TWO = Config(n_numbers=2, target=1, digits="01")
P_TWO = Problem((1, 1), TWO)


class ToyLM(torch.nn.Module):
    """Position-independent logits held directly as a parameter.

    Being position independent makes the whole policy enumerable by hand, and holding the logits
    as the parameter means d/d(logit) is exactly the quantity the analytic KL gradient describes.
    `disable_adapter` mimics peft's context manager by switching to the frozen reference logits.
    """

    def __init__(self, theta_logits, ref_logits):
        super().__init__()
        self.z = torch.nn.Parameter(torch.tensor(theta_logits, dtype=torch.float64))
        self.register_buffer("zref", torch.tensor(ref_logits, dtype=torch.float64))
        self.use_ref = False

    @contextlib.contextmanager
    def disable_adapter(self):
        self.use_ref = True
        try:
            yield
        finally:
            self.use_ref = False

    def forward(self, input_ids=None, attention_mask=None, **kw):
        z = self.zref if self.use_ref else self.z
        B, L = input_ids.shape
        return types.SimpleNamespace(logits=z.view(1, 1, -1).expand(B, L, V))


def toy(theta_logits, ref_logits):
    model = ToyLM(theta_logits, ref_logits)
    return HFLM(model, lambda prob: [0], list(range(V)), pad_id=0, device="cpu"), model


def analytic_grad(zt, zr, prob, seqs):
    """Closed form of d/dz sum_t KL_t, summed over the positions of every sequence.

    For p = masked_softmax(z) and q = masked_softmax(zr) at one position,
        d KL(p || q) / d z_j = p_j * [(log p_j - log q_j) - KL(p || q)]   for allowed j, else 0.
    The toy LM shares one z across all positions, so the position gradients add.
    """
    g = np.zeros(V)
    for s in seqs:
        for t in range(len(s)):
            m = allowed_mask(s[:t], prob.cfg)
            def dist(z):
                x = np.where(m, z, -np.inf)
                x = x - x.max()
                e = np.exp(x) * m
                return e / e.sum()
            p, q = dist(zt), dist(zr)
            lr = np.zeros(V)
            lr[m] = np.log(p[m]) - np.log(q[m])
            kl = float((p[m] * lr[m]).sum())
            g[m] += p[m] * (lr[m] - kl)
    return g


# ---------------------------------------------------------------- the reviewer's example
def test_two_symbol_example_matches_0_221807():
    """pi_theta = (0.8, 0.2), pi_ref = (0.5, 0.5): value 0.192744757, gradient 0.221807."""
    zt, zr = np.zeros(V), np.zeros(V)
    zt[SID["0"]], zt[SID["1"]] = np.log(0.8), np.log(0.2)
    zr[SID["0"]], zr[SID["1"]] = np.log(0.5), np.log(0.5)
    lm, model = toy(zt, zr)

    kl = lm.kl_to_ref_torch(P_TWO, ["0"], model.disable_adapter)
    assert kl.shape == (1,)
    # the forward KL, not the reverse one
    assert float(kl.detach()) == pytest.approx(0.192744757, abs=1e-9)

    kl.mean().backward()
    g = model.z.grad.numpy()
    assert g[SID["0"]] == pytest.approx(0.221807, abs=5e-7)
    assert g[SID["1"]] == pytest.approx(-0.221807, abs=5e-7)
    # the two estimators this replaced would have given these instead
    assert g[SID["0"]] != pytest.approx(0.3, abs=1e-3), "this is the k3 gradient, wrong direction"
    assert g[SID["0"]] != pytest.approx(0.0, abs=1e-3), "this is the plain gradient, no signal"


def test_value_is_the_forward_kl_not_the_reverse():
    zt, zr = np.zeros(V), np.zeros(V)
    zt[SID["0"]], zt[SID["1"]] = np.log(0.8), np.log(0.2)
    zr[SID["0"]], zr[SID["1"]] = np.log(0.5), np.log(0.5)
    lm, model = toy(zt, zr)
    kl = float(lm.kl_to_ref_torch(P_TWO, ["0"], model.disable_adapter).mean().detach())
    assert kl == pytest.approx(0.192744757, abs=1e-9)
    assert kl != pytest.approx(0.223143551, abs=1e-6)  # KL(ref||theta) is a different number


# ---------------------------------------------------------------- general analytic agreement
@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("seqs", [["0"], ["0+1=1\n"], ["0+1=1\n", "1", "1+1=10\n"]])
def test_autograd_equals_analytic_gradient(seed, seqs):
    """Over multiple positions, multiple sequences and random logits, to 1e-10."""
    rng = np.random.default_rng(seed)
    zt, zr = rng.normal(size=V), rng.normal(size=V)
    lm, model = toy(zt, zr)
    kl = lm.kl_to_ref_torch(P_TWO, seqs, model.disable_adapter)
    assert kl.shape == (len(seqs),)
    assert (kl.detach().numpy() >= -1e-12).all(), "a KL is non-negative"
    kl.sum().backward()                         # sum, so the analytic form adds over sequences
    got = model.z.grad.numpy()
    want = analytic_grad(zt, zr, P_TWO, seqs)
    assert np.abs(got - want).max() < 1e-10, f"max |diff| {np.abs(got - want).max():.3e}"


def test_zero_when_reference_equals_policy():
    rng = np.random.default_rng(0)
    z = rng.normal(size=V)
    lm, model = toy(z, z.copy())
    kl = lm.kl_to_ref_torch(P_TWO, ["0+1=1\n"], model.disable_adapter)
    assert float(kl.sum()) == pytest.approx(0.0, abs=1e-12)
    kl.sum().backward()
    assert np.abs(model.z.grad.numpy()).max() < 1e-12


def test_masked_symbols_contribute_nothing_and_no_nan():
    """Disallowed symbols hold -inf in both policies; the term must stay finite, gradient included."""
    rng = np.random.default_rng(3)
    zt, zr = rng.normal(size=V) * 5, rng.normal(size=V) * 5
    lm, model = toy(zt, zr)
    seqs = ["0+1=1\n", "1"]
    kl = lm.kl_to_ref_torch(P_TWO, seqs, model.disable_adapter)
    assert torch.isfinite(kl).all()
    kl.sum().backward()
    assert torch.isfinite(model.z.grad).all()
    # a disallowed symbol at every visited position gets exactly zero gradient
    visited = set()
    for s in seqs:
        for t in range(len(s)):
            visited |= {SYMS[i] for i in np.flatnonzero(allowed_mask(s[:t], TWO))}
    for sym in set(SYMS) - visited:
        assert model.z.grad[SID[sym]].item() == 0.0


# ---------------------------------------------------------------- the real LoRA path
def test_exactly_zero_at_lora_initialisation():
    """LoRA initialises B at zero, so pi_theta and pi_ref are the same distribution."""
    from test_train_smoke import fixture_batch, tiny_trainer
    lm, trainable, opt, _ = tiny_trainer()
    prob, _ = fixture_batch()[0]
    seqs = ["6/2=3\n3+3=6\n6*4=24\n", "12*34=56\n7-7=0\n0+0=0\n", "9"]
    kl = lm.kl_to_ref_torch(prob, seqs, lm.model.disable_adapter)
    assert kl.shape == (len(seqs),)
    assert float(kl.abs().max()) == 0.0, f"not exactly zero: {kl}"


def test_positive_once_the_adapter_has_moved(monkeypatch):
    from test_train_smoke import fixture_batch, split_reward, tiny_trainer
    split_reward(monkeypatch)
    lm, trainable, opt, _ = tiny_trainer(lr=1e-2)
    batch = fixture_batch()
    r0 = ts.step_once(lm, batch, np.random.default_rng(0), opt, trainable,
                      kl_coef=0.01, n_rollouts=8, max_grad_norm=1.0)
    assert r0["kl"] == pytest.approx(0.0, abs=1e-12)
    r1 = ts.step_once(lm, batch, np.random.default_rng(1), opt, trainable,
                      kl_coef=0.01, n_rollouts=8, max_grad_norm=1.0)
    assert r1["kl"] > 0.0
