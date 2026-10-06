"""Minimal GRPO with LoRA. A smoke test of plumbing, not an experiment.

What it does, per optimizer step, for each problem in the step's batch:
  * 8 rollouts from policy.sample  (Policy A, temperature 1, nothing discarded)
  * binary reward from protocol.classify: 1.0 for 'success', 0.0 for 'semantic_fail' and
    'overlength'
  * advantage = reward - group mean reward, the group being that problem's 8 rollouts
  * log pi from HFLM.seq_logp_torch, i.e. the same masked log-softmax as every measurement
  * optional reference-policy KL: the exact per-position KL(pi_theta || pi_ref) summed over the
    positions of each sampled sequence and averaged over sequences, the reference being the same
    weights with the LoRA adapter disabled under no_grad, both policies evaluated with the same
    mask and the same masked log-softmax (HFLM.kl_to_ref_torch)
  * one AdamW step

What it logs, per optimizer step: mean reward, the number of rollouts per first-line entry per
problem, and -- separately -- whether the update contained any rollout at all entering each
tracked entry. The second is not derivable from an average and is the thing that makes a later
reading of a route's trajectory honest: a step in which no rollout entered a route carries no
gradient signal about that route, and must not be read as evidence about it. At evaluation points
it logs the exact policy.measure output on the frozen instances.

This script deliberately does NOT sweep. --lr and --kl-coef take one value each from the two
pre-declared options. Choosing a setting because it moves some route's e is exactly the thing
that would invalidate the later experiment.

    python train_smoke.py --backend tiny --steps 3 --out-dir <dir>     # local plumbing check
    python train_smoke.py --backend qwen --lr 1e-5 --kl-coef 0.01      # on the cluster
"""
import argparse
import collections
import datetime as dt
import json
import time

import numpy as np
import torch

import backends
import measure as measure_mod
import paths
from make_instances import load_instances
from policy import measure, sample
from protocol import NL, classify

LR_CHOICES = [1e-5, 5e-5]
KL_CHOICES = [0.01, 0.0]
LORA_RANK = 16
ROLLOUTS = 8
OUTCOMES = ("success", "semantic_fail", "overlength")

# LoRA targets per architecture. Qwen2 uses the llama-style projection names; the local tiny
# GPT-2 used for the plumbing check has fused attention under one name.
TARGET_MODULES = {
    "qwen": ["q_proj", "k_proj", "v_proj", "o_proj"],
    "tiny": ["c_attn"],
}


def wrap_lora(lm, backend, rank=LORA_RANK, alpha=None, targets=None):
    from peft import LoraConfig, get_peft_model
    cfg = LoraConfig(r=rank, lora_alpha=alpha or 2 * rank, lora_dropout=0.0, bias="none",
                     task_type="CAUSAL_LM",
                     target_modules=targets or TARGET_MODULES[backend])
    lm.model = get_peft_model(lm.model, cfg)
    lm.model.eval()  # no dropout anywhere; LoRA dropout is 0 and the base model has no others
    trainable = [p for p in lm.model.parameters() if p.requires_grad]
    n_tr = sum(p.numel() for p in trainable)
    n_all = sum(p.numel() for p in lm.model.parameters())
    return trainable, {"lora_rank": rank, "lora_alpha": alpha or 2 * rank,
                       "target_modules": targets or TARGET_MODULES[backend],
                       "trainable_params": n_tr, "all_params": n_all,
                       "trainable_fraction": n_tr / n_all}


def rollout_stats(seqs, rewards, rec):
    """Entry counts over ALL first lines, and entered-or-not over the tracked (solvable) entries."""
    first = [s.split(NL)[0] for s in seqs]
    counts = collections.Counter(first)
    tracked = rec["solvable_entries"]
    return {
        "rollouts_per_entry": dict(sorted(counts.items())),
        "entered_tracked_entry": {e: bool(counts.get(e, 0)) for e in tracked},
        "rollouts_per_tracked_entry": {e: int(counts.get(e, 0)) for e in tracked},
        "n_distinct_first_lines": len(counts),
        "reward_mean": float(np.mean(rewards)),
        "rewards": [float(r) for r in rewards],
    }


def step_once(lm, batch, rng, opt, trainable, kl_coef, n_rollouts, max_grad_norm):
    """One optimizer step over `batch` = [(Problem, record), ...]. -> log row."""
    opt.zero_grad(set_to_none=True)
    pg_terms, kl_terms, per_problem = [], [], {}
    outcomes = {k: 0 for k in OUTCOMES}
    for prob, rec in batch:
        seqs = sample(lm, prob, n_rollouts, rng)
        cls = [classify(s, prob) for s in seqs]
        for c in cls:
            outcomes[c] += 1
        r = np.array([1.0 if c == "success" else 0.0 for c in cls])
        adv = r - r.mean()                              # group-mean baseline
        logp = lm.seq_logp_torch(prob, seqs)
        a = torch.as_tensor(adv, dtype=logp.dtype, device=logp.device)
        pg_terms.append(-(a * logp))
        if kl_coef:
            # exact per-position KL(pi_theta || pi_ref), shape [B]; see HFLM.kl_to_ref_torch
            kl_terms.append(lm.kl_to_ref_torch(prob, seqs, lm.model.disable_adapter))
        st = rollout_stats(seqs, r, rec)
        st["outcomes"] = dict(collections.Counter(cls))
        st["logp_mean"] = float(logp.detach().float().mean())
        per_problem[prob.key] = st

    pg = torch.cat(pg_terms).mean()
    loss = pg
    kl_val = None
    if kl_terms:
        klm = torch.cat(kl_terms).mean()
        kl_val = float(klm.detach().float())
        loss = loss + kl_coef * klm
    loss.backward()
    gnorm = float(torch.nn.utils.clip_grad_norm_(trainable, max_grad_norm))
    opt.step()

    rewards = [v["reward_mean"] for v in per_problem.values()]
    return {
        "problems": list(per_problem),
        "reward_mean": float(np.mean(rewards)),
        "reward_per_problem": {k: v["reward_mean"] for k, v in per_problem.items()},
        "n_rollouts": n_rollouts * len(batch),
        "outcome_counts": outcomes,
        "loss": float(loss.detach().float()), "loss_pg": float(pg.detach().float()),
        "kl": kl_val, "grad_norm": gnorm,
        "per_problem": per_problem,
    }


def eval_exact(lm, instances):
    """Exact measure() on the frozen instances, no sampling. The quantity the project is about."""
    rows, summary = measure_mod.run(lm, instances, with_crosscheck=False, verbose=False)
    return {"summary": summary, "problems": rows}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    paths.add_paths_arg(ap)
    ap.add_argument("--backend", default="qwen", choices=["qwen", "tiny"])
    ap.add_argument("--lr", type=float, default=LR_CHOICES[0], choices=LR_CHOICES,
                    help="one of the two pre-declared learning rates")
    ap.add_argument("--kl-coef", type=float, default=KL_CHOICES[0], choices=KL_CHOICES,
                    help="reference-policy KL coefficient; 0 disables the term. The term is the "
                         "exact per-position KL(pi_theta || pi_ref); there is no choice of "
                         "estimator")
    ap.add_argument("--lora-rank", type=int, default=LORA_RANK)
    ap.add_argument("--rollouts", type=int, default=ROLLOUTS, help="rollouts per problem per step")
    ap.add_argument("--steps", type=int, default=20, help="optimizer steps")
    ap.add_argument("--problems-per-step", type=int, default=2)
    ap.add_argument("--eval-every", type=int, default=10,
                    help="exact measure() every N steps (also at step 0 and at the end)")
    ap.add_argument("--eval-limit", type=int, default=None,
                    help="evaluate on the first N frozen instances (default: all)")
    ap.add_argument("--instances", default=None)
    ap.add_argument("--out-dir", default=None, help="default: OUTPUT_DIR from the paths file")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=64, help="LM batch size")
    ap.add_argument("--max-grad-norm", type=float, default=1.0)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    args = ap.parse_args()

    cfg_paths = paths.load_paths(args.paths)
    out_dir = args.out_dir or str(paths.require_dir(cfg_paths, "OUTPUT_DIR"))
    paths.require_dir({"OUTPUT_DIR": out_dir, "__file__": "--out-dir"}, "OUTPUT_DIR")
    meta, instances = load_instances(args.instances or str(paths.REPO_ROOT / "instances.json"))
    eval_instances = instances[:args.eval_limit] if args.eval_limit else instances

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    lm, backend_desc = backends.build(args.backend, cfg_paths, seed=args.seed,
                                      batch_size=args.batch_size)
    trainable, lora_desc = wrap_lora(lm, args.backend, rank=args.lora_rank)
    opt = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)

    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    name = "_".join(x for x in ["train_smoke", args.backend, args.tag, stamp] if x) + ".jsonl"
    log_path = f"{out_dir.rstrip('/')}/{name}"
    header = {"record": "header", "schema": "entry-or-exec/train_smoke/1",
              "created_utc": stamp, "git_commit": measure_mod.git_commit(),
              "args": vars(args) | {"paths_file": cfg_paths["__file__"]},
              "backend": backend_desc, "lora": lora_desc,
              "instances_file": args.instances or "instances.json",
              "instances_md5": measure_mod.file_md5(
                  args.instances or str(paths.REPO_ROOT / "instances.json")),
              "n_eval_instances": len(eval_instances),
              "reward": "binary, 1.0 iff protocol.classify == 'success'",
              "baseline": "group mean over the problem's rollouts",
              "kl": "exact per-position KL(pi_theta || pi_ref), summed over positions, "
                    "mean over sequences; reference = same weights, adapter disabled",
              "note": "smoke test of plumbing; not a sweep and not an experiment"}
    log = open(log_path, "w")

    def emit(row):
        log.write(json.dumps(row, default=float) + "\n")
        log.flush()

    emit(header)
    print(f"log         {log_path}")
    print(f"backend     {json.dumps(backend_desc)}")
    print(f"lora        {json.dumps(lora_desc)}")
    print(f"optimizer   AdamW lr={args.lr} wd={args.weight_decay} "
          f"kl_coef={args.kl_coef} (exact per-position KL) clip={args.max_grad_norm}")
    print(f"schedule    {args.steps} steps x {args.problems_per_step} problems "
          f"x {args.rollouts} rollouts, eval every {args.eval_every} on "
          f"{len(eval_instances)} instances\n")

    t0 = time.time()
    order = list(range(len(instances)))

    def do_eval(step):
        te = time.time()
        ev = eval_exact(lm, eval_instances)
        s = ev["summary"]
        emit({"record": "eval", "step": step, "seconds": round(time.time() - te, 1), **ev})
        print(f"  eval @ step {step:<4} mean p_success {s['mean_p_success_total']:.4e}  "
              f"mean wrong_first {s['mean_p_first_line_semantically_wrong']:.4f}  "
              f"filter {s['filter']['n_pass_entry_level']}/{s['n_problems']}  "
              f"{time.time() - te:.0f}s", flush=True)

    do_eval(0)
    for step in range(1, args.steps + 1):
        k = args.problems_per_step
        pick = [order[((step - 1) * k + i) % len(order)] for i in range(k)]
        batch = [instances[i] for i in pick]
        row = step_once(lm, batch, rng, opt, trainable, args.kl_coef, args.rollouts,
                        args.max_grad_norm)
        row = {"record": "step", "step": step, "seconds": round(time.time() - t0, 1), **row}
        emit(row)
        entered = sum(sum(v["entered_tracked_entry"].values()) for v in row["per_problem"].values())
        n_tracked = sum(len(v["entered_tracked_entry"]) for v in row["per_problem"].values())
        print(f"  step {step:<4} reward {row['reward_mean']:.3f}  loss {row['loss']:+.4f}  "
              f"kl {row['kl'] if row['kl'] is None else round(row['kl'], 5)}  "
              f"|g| {row['grad_norm']:.3f}  tracked entries entered "
              f"{entered}/{n_tracked}  {row['problems']}", flush=True)
        if args.eval_every and step % args.eval_every == 0 and step != args.steps:
            do_eval(step)
    do_eval(args.steps)

    log.close()
    print(f"\ndone in {time.time() - t0:.0f}s -> {log_path}")


if __name__ == "__main__":
    paths.run_main(main)
