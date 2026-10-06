"""Build the frozen instance set.

Enumerate every 4-multiset of 1..9, keep the problems that have at least two distinct canonical
first-step events leading to the target, sample `--n` of them with a fixed seed, and write
instances.json. Each record carries the full list of successful strings and the solvable
entries, so measure.py and train_smoke.py never have to re-derive them.

instances.json is FROZEN once committed. Re-running this script with the default arguments must
reproduce it byte for byte; `--target` writes a differently named file for small-target
calibration and those files are gitignored.

    python make_instances.py                      # 32 problems, target 24 -> instances.json
    python make_instances.py --target 10          # calibration set -> <DATA_DIR>/instances_target10.json
"""
import argparse
import json
from itertools import combinations_with_replacement

import numpy as np

import paths
from protocol import Config, Problem, NL, canonical_event, success_trajectories, valid_lines

DEFAULT_TARGET = 24
DEFAULT_N = 32
DEFAULT_SEED = 0
MIN_SOLVABLE_EVENTS = 2


def analyse(numbers, cfg):
    """Record for one multiset, or None if it has fewer than MIN_SOLVABLE_EVENTS solvable events."""
    prob = Problem(tuple(sorted(numbers)), cfg)
    trajs = success_trajectories(prob)
    if not trajs:
        return None
    firsts = sorted({t.split(NL)[0] for t in trajs})
    events = sorted({canonical_event(f) for f in firsts})
    if len(events) < MIN_SOLVABLE_EVENTS:
        return None
    return {
        "numbers": list(prob.numbers),
        "target": cfg.target,
        "key": prob.key,
        "n_valid_first_lines": len(valid_lines(prob.numbers, cfg)),
        "n_success_strings": len(trajs),
        "success_strings": trajs,
        "solvable_entries": firsts,
        "solvable_events": events,
        "entry_to_event": {f: canonical_event(f) for f in firsts},
    }


def build(target=DEFAULT_TARGET, n=DEFAULT_N, seed=DEFAULT_SEED, lo=1, hi=9, k=4):
    cfg = Config(n_numbers=k, target=target)
    pool = []
    for nums in combinations_with_replacement(range(lo, hi + 1), k):
        rec = analyse(nums, cfg)
        if rec is not None:
            pool.append(rec)
    if len(pool) < n:
        raise RuntimeError(f"only {len(pool)} eligible multisets for target {target}, need {n}")
    rng = np.random.default_rng(seed)
    idx = sorted(rng.choice(len(pool), size=n, replace=False).tolist())
    chosen = [pool[i] for i in idx]
    return {
        "schema": "entry-or-exec/instances/1",
        "frozen": True,
        "target": target,
        "n_numbers": k,
        "number_range": [lo, hi],
        "seed": seed,
        "n_instances": n,
        "min_solvable_events": MIN_SOLVABLE_EVENTS,
        "n_multisets_enumerated": sum(1 for _ in combinations_with_replacement(range(lo, hi + 1), k)),
        "n_eligible": len(pool),
        "instances": chosen,
    }


def load_instances(path):
    """Read an instance file and return (meta, [(Problem, record), ...])."""
    with open(path) as fh:
        data = json.load(fh)
    cfg = Config(n_numbers=data["n_numbers"], target=data["target"])
    out = []
    for rec in data["instances"]:
        prob = Problem(tuple(rec["numbers"]), cfg)
        assert prob.key == rec["key"], f"instance key mismatch: {prob.key} vs {rec['key']}"
        out.append((prob, rec))
    return data, out


def default_out(args, cfg_paths):
    if args.out:
        return args.out
    if args.target == DEFAULT_TARGET:
        return str(paths.REPO_ROOT / "instances.json")  # frozen, lives in the repository
    data_dir = paths.require_dir(cfg_paths, "DATA_DIR")
    return str(data_dir / f"instances_target{args.target}.json")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    paths.add_paths_arg(ap)
    ap.add_argument("--target", type=int, default=DEFAULT_TARGET,
                    help="target value (default 24; other values are calibration sets)")
    ap.add_argument("--n", type=int, default=DEFAULT_N, help="problems to sample")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED, help="sampling seed")
    ap.add_argument("--out", default=None, help="output path (overrides the default location)")
    args = ap.parse_args()

    cfg_paths = paths.load_paths(args.paths)
    out = default_out(args, cfg_paths)
    data = build(target=args.target, n=args.n, seed=args.seed)
    with open(out, "w") as fh:
        json.dump(data, fh, indent=1, sort_keys=False)
        fh.write("\n")
    print(f"target {data['target']}: {data['n_eligible']} / {data['n_multisets_enumerated']} "
          f"multisets eligible, sampled {data['n_instances']} with seed {data['seed']}")
    print(f"wrote {out}")
    ns = [r["n_success_strings"] for r in data["instances"]]
    ev = [len(r["solvable_events"]) for r in data["instances"]]
    print(f"success strings per problem: min {min(ns)} median {int(np.median(ns))} max {max(ns)}")
    print(f"solvable events per problem: min {min(ev)} median {int(np.median(ev))} max {max(ev)}")


if __name__ == "__main__":
    paths.run_main(main)
