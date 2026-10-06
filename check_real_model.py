"""Checks that can only be run where the real model lives. Run this FIRST on Skipjack.

Nothing here is a property of Countdown; these are the four things that have to hold before any
number produced by measure.py means anything.

  tokenizer  every one of the 16 symbols is a single token for the Qwen tokenizer
  d          the differentiable training log-prob equals the teacher-forced scorer
  e          single-sequence and padded-batch scoring agree
  f          a tabular copy of the policy reproduces it on every full history
  mass       the exhaustive tree walk puts total mass 1 on success / semantic_fail / overlength,
             the overlength mass is 0, and the success mass equals measure()'s p_success_total
  precision  the same 20 sequences scored by a float32 and a bfloat16 copy of the model, with the
             max absolute difference in sequence log-prob printed. load_qwen defaults to float32;
             this says what bfloat16 would cost if a run switches to it for throughput. It is
             reported, never asserted: the number is the finding.

The mass-conservation walk visits every grammar-and-semantics-reachable prefix, which is tens of
thousands of forward positions per problem. batched.mass_conservation_batched is used instead of
policy.mass_conservation; test_batched.py pins the two to 1e-12 on the mock, so the result is the
same walk with the LM calls batched. It is still the slow part -- budget time for it.

Tolerances for d and e are looser than in test_protocol.py on purpose. Both models run in float32
now that load_qwen defaults to it, but the tiny test model runs on CPU in a single batch while
Qwen runs on GPU and the two code paths group sequences into batches differently, so identical
mathematics can still differ in the last digits. The measured maxima are printed, not just a
pass/fail, so a surprise is visible: on the tiny model they are around 2e-6, and a real-model
figure far above that is a finding about precision rather than a pass. If a run deliberately
switches to bfloat16 for throughput, expect these to grow and read the precision check first.

    python check_real_model.py                        # qwen, first two instances
    python check_real_model.py --backend tiny         # exercise this script without the model
"""
import argparse
import json
import sys
import time

import numpy as np

import backends
import paths
from batched import mass_conservation_batched
from make_instances import load_instances
from policy import TabularLM, measure, next_logp, sample, seq_logp
from protocol import SYMS, Problem, success_trajectories

TOL_D = 1e-3
TOL_E = 1e-3
TOL_F = 1e-3
TOL_MASS = 1e-6
N_PRECISION_SEQS = 20

BAD_SEQS = ["12*34=56\n7-7=0\n0+0=0\n", "9", "6/2=3\n3+3=6\n"]


class Report:
    def __init__(self):
        self.rows = []

    def add(self, name, ok, detail):
        self.rows.append((name, bool(ok), detail))
        print(f"  {'PASS' if ok else 'FAIL'}  {name:<34} {detail}", flush=True)

    @property
    def ok(self):
        return all(r[1] for r in self.rows)


def check_tokenizer(model_path, cache_dir, rep):
    """Every symbol must be exactly one token id, and the 16 ids must be distinct."""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_path, cache_dir=cache_dir)
    ids, bad = [], []
    for s in SYMS:
        enc = tok.encode(s, add_special_tokens=False)
        if len(enc) != 1:
            bad.append((s, enc))
        else:
            ids.append(enc[0])
    rep.add("tokenizer: one symbol one token", not bad,
            f"{len(SYMS)} symbols, offenders: {bad}" if bad else f"ids {ids}")
    rep.add("tokenizer: symbol ids distinct", len(set(ids)) == len(ids),
            f"{len(set(ids))} distinct of {len(ids)}")
    return tok


def check_d(lm, prob, rep):
    """d. differentiable (training) log-prob equals the teacher-forced scorer."""
    seqs = success_trajectories(prob)[:6] + BAD_SEQS
    a = seq_logp(lm, prob, seqs)
    b = lm.seq_logp_torch(prob, seqs)
    diff = float(np.abs(a - b.detach().float().cpu().numpy()).max())
    rep.add(f"d  train logp == scorer [{prob.key}]", diff < TOL_D and bool(b.requires_grad),
            f"max |diff| {diff:.3e} (tol {TOL_D:.0e}), requires_grad {bool(b.requires_grad)}, "
            f"{len(seqs)} sequences")


def check_e(lm, prob, rep):
    """e. single-sequence and padded-batch computation agree, numpy and torch paths both."""
    seqs = success_trajectories(prob)[:3] + BAD_SEQS + ["4"]
    batched = seq_logp(lm, prob, seqs)
    single = np.array([seq_logp(lm, prob, [s])[0] for s in seqs])
    d_np = float(np.abs(batched - single).max())
    bt = lm.seq_logp_torch(prob, seqs).detach().float().cpu().numpy()
    st = np.array([lm.seq_logp_torch(prob, [s]).item() for s in seqs])
    d_pt = float(np.abs(bt - st).max())
    rep.add(f"e  batch/pad invariance [{prob.key}]", max(d_np, d_pt) < TOL_E,
            f"numpy {d_np:.3e}, torch {d_pt:.3e} (tol {TOL_E:.0e})")


def check_f(lm, prob, rep, n_sample=200, seed=2):
    """f. tabular copy reproduces the LM on every full history it was built from."""
    tab = TabularLM(lm)
    hist = set()
    for tr in success_trajectories(prob):
        hist.update(tr[:i] for i in range(len(tr)))
    for s in sample(lm, prob, n_sample, np.random.default_rng(seed)):
        hist.update(s[:i] for i in range(len(s)))
    hist = sorted(hist)
    d = float(np.abs(np.exp(next_logp(lm, prob, hist)) - np.exp(next_logp(tab, prob, hist))).max())
    ma, mb = measure(lm, prob), measure(tab, prob)
    d_tot = abs(ma["p_success_total"] - mb["p_success_total"])
    d_ent = max((abs(ma["entries"][k]["p_entry"] - mb["entries"][k]["p_entry"])
                 for k in ma["entries"]), default=0.0)
    rep.add(f"f  tabular copy [{prob.key}]", max(d, d_tot, d_ent) < TOL_F,
            f"{len(hist)} histories, max |dp| {d:.3e}, p_success {d_tot:.3e}, "
            f"p_entry {d_ent:.3e} (tol {TOL_F:.0e})")


def check_mass(lm, prob, rep, batch_size=256):
    """Termination / truncation: all mass lands in exactly one of the three outcome classes."""
    t0 = time.time()
    seen = {"n": 0}

    def progress(n_next, n_done):
        seen["n"] = n_done
        print(f"      ... {n_done} nodes walked, frontier {n_next}, {time.time() - t0:.0f}s",
              flush=True)

    s, f, o, stats = mass_conservation_batched(lm, prob, batch_size=batch_size, progress=progress)
    exact = measure(lm, prob)["p_success_total"]
    ok = abs(s + f + o - 1) < TOL_MASS and o == 0.0 and abs(s - exact) < TOL_MASS
    rep.add(f"mass conservation [{prob.key}]", ok,
            f"success {s:.6e} semantic_fail {f:.6e} overlength {o:.1e} "
            f"sum {s + f + o:.12f} vs measure {exact:.6e} (|d| {abs(s - exact):.2e}); "
            f"{stats['nodes']} nodes in {stats['lm_calls']} LM calls, {time.time() - t0:.0f}s")


def precision_seqs(prob, n=N_PRECISION_SEQS):
    """Exactly n distinct scorable strings, deterministically: successes, then deliberately
    wrong ones, then partial prefixes of successes to fill. Partials are included on purpose --
    a short string exercises fewer masked conditionals, so the error should scale with length."""
    trajs = success_trajectories(prob)
    pool = list(trajs) + BAD_SEQS + [tr[:i] for tr in trajs for i in range(1, len(tr))]
    out = list(dict.fromkeys(pool))[:n]
    assert len(out) == n, f"only {len(out)} distinct strings available for {prob.key}"
    return out


def check_precision(lm_fp32, prob, cfg_paths, rep, batch_size=64):
    """float32 vs bfloat16 on identical sequences. Measures, does not judge."""
    import torch
    seqs = precision_seqs(prob)
    a = seq_logp(lm_fp32, prob, seqs)
    lm_bf16, desc = backends.build("qwen", cfg_paths, batch_size=batch_size, dtype="bfloat16")
    try:
        b = seq_logp(lm_bf16, prob, seqs)
    finally:
        del lm_bf16
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    d = np.abs(a - b)
    ratio = np.exp(b - a)
    worst = float(max(abs(ratio.min() - 1.0), abs(ratio.max() - 1.0)))
    rep.add(f"precision fp32 vs bf16 [{prob.key}]", True,
            f"{len(seqs)} sequences, max |d log pi| {d.max():.3e} nats "
            f"(median {np.median(d):.3e}), worst pi(seq) error {100 * worst:.2f}%; "
            f"bf16 dtype {desc['dtype']}  [reported, not asserted]")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    paths.add_paths_arg(ap)
    ap.add_argument("--backend", default="qwen", choices=["qwen", "tiny"])
    ap.add_argument("--instances", default=None,
                    help="instance file (default: the frozen instances.json)")
    ap.add_argument("--keys", nargs="*", default=None,
                    help="problem keys to use (default: the first two instances)")
    ap.add_argument("--n-problems", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=64, help="LM batch size")
    ap.add_argument("--mass-batch-size", type=int, default=256,
                    help="prefixes per LM call in the exhaustive walk")
    ap.add_argument("--skip-mass", action="store_true",
                    help="skip the exhaustive walk (the slow check)")
    ap.add_argument("--skip-precision", action="store_true",
                    help="skip the float32/bfloat16 comparison (it loads a second model copy)")
    args = ap.parse_args()

    cfg_paths = paths.load_paths(args.paths)
    meta, instances = load_instances(args.instances or str(paths.REPO_ROOT / "instances.json"))
    if args.keys:
        by_key = {p.key: (p, r) for p, r in instances}
        missing = [k for k in args.keys if k not in by_key]
        if missing:
            sys.exit(f"keys not in the instance file: {missing}")
        chosen = [by_key[k] for k in args.keys]
    else:
        chosen = instances[:args.n_problems]

    print(f"instances   {args.instances or 'instances.json'} "
          f"({len(instances)} problems, target {meta['target']})")
    print(f"problems    {[p.key for p, _ in chosen]}")
    rep = Report()

    if args.backend == "qwen":
        cache = backends.set_hf_cache(cfg_paths)
        model_path = paths.require(cfg_paths, "MODEL_PATH")
        print(f"model       {model_path}\nhf cache    {cache}\n")
        print("tokenizer")
        check_tokenizer(model_path, cache, rep)
    else:
        print("\nbackend 'tiny': randomly initialised GPT-2, no tokenizer. The symbol/token "
              "check does not apply and is skipped; this run only proves the script executes.\n")

    lm, desc = backends.build(args.backend, cfg_paths, batch_size=args.batch_size)
    print(f"\nbackend     {json.dumps(desc)}\n")

    print("tests d, e, f")
    for prob, _ in chosen:
        check_d(lm, prob, rep)
        check_e(lm, prob, rep)
        check_f(lm, prob, rep)

    if args.backend == "qwen" and not args.skip_precision:
        print("\nprecision: float32 vs bfloat16 (loads a second copy of the model)")
        check_precision(lm, chosen[0][0], cfg_paths, rep, batch_size=args.batch_size)
    elif args.backend != "qwen":
        print("\nprecision check skipped: it compares two dtypes of the real model")

    if args.skip_mass:
        print("\nmass conservation skipped (--skip-mass)")
    else:
        print("\nmass conservation (exhaustive walk, slow)")
        for prob, _ in chosen:
            check_mass(lm, prob, rep, batch_size=args.mass_batch_size)

    n_ok = sum(r[1] for r in rep.rows)
    print(f"\n{n_ok}/{len(rep.rows)} checks passed")
    sys.exit(0 if rep.ok else 1)


if __name__ == "__main__":
    paths.run_main(main)
