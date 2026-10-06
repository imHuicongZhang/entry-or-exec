"""Exact measurement loop on a frozen policy, plus a sampling cross-check.

For every instance in instances.json this writes, for every entry and every canonical event,
the four quantities of PROTOCOL.md section 7 (p_entry, p_success, share, e) together with
p_first_line_semantically_wrong, and then checks the exact numbers against sampling:

  * N samples from the root, success frequency vs the exact p_success_total;
  * N forced-entry samples for two solvable entries, success frequency vs the exact e.

Agreement is reported as z = (sampled - exact) / standard error, so a run is readable without
re-deriving anything. Nothing is tuned to make z small or to make the pre-registered filter
count large; both are reported as found.

    python measure.py --backend mock --out-dir <dir>      # local, validates the code only
    python measure.py --backend qwen                      # on the cluster, OUTPUT_DIR from paths
"""
import argparse
import datetime as dt
import hashlib
import json
import platform
import subprocess
import time

import numpy as np

import backends
import paths
from make_instances import load_instances
from policy import measure, sample
from protocol import NL, classify

# Pre-registered, fixed in advance. See PROTOCOL.md section 9.
FILTER_E_LO, FILTER_E_HI, FILTER_P_ENTRY_MIN, FILTER_MIN_ENTRIES = 0.2, 0.8, 0.05, 2
OUTCOMES = ("success", "semantic_fail", "overlength")


def git_commit():
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=paths.REPO_ROOT,
                              capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return None


def file_md5(path):
    with open(path, "rb") as fh:
        return hashlib.md5(fh.read()).hexdigest()


def z_score(hat, exact, n):
    """(sampled - exact) in units of the binomial standard error of the exact probability."""
    se = float(np.sqrt(exact * (1.0 - exact) / n)) if 0.0 < exact < 1.0 else 0.0
    return {"sampled": float(hat), "exact": float(exact), "n": int(n), "se": se,
            "z": (float(hat) - float(exact)) / se if se > 0 else None,
            "abs_diff": abs(float(hat) - float(exact)),
            # with fewer than 5 expected successes (or failures) the normal approximation says
            # nothing; such a comparison is recorded but excluded from the |z| summary.
            "informative": bool(exact * n >= 5 and (1.0 - exact) * n >= 5)}


def outcome_counts(seqs, prob):
    c = {k: 0 for k in OUTCOMES}
    for s in seqs:
        c[classify(s, prob)] += 1
    return c


def two_tracked_entries(m, rec):
    """Lowest- and highest-p_entry solvable entries (mirrors test_c). Always two distinct keys."""
    solv = sorted(rec["solvable_entries"], key=lambda k: (m["entries"][k]["p_entry"], k))
    if len(solv) < 2:
        return solv
    return [solv[0], solv[-1]]


def passes_filter(level):
    """Pre-registered filter at entry or event level: >=2 routes with e in range and p_entry high."""
    good = [k for k, v in level.items()
            if v.get("e") is not None and FILTER_E_LO <= v["e"] <= FILTER_E_HI
            and v["p_entry"] >= FILTER_P_ENTRY_MIN]
    return good


def crosscheck(lm, prob, m, rec, n_samples, rng):
    """Sampling cross-check for one problem: root plus the two tracked entries."""
    root_seqs = sample(lm, prob, n_samples, rng)
    root_cnt = outcome_counts(root_seqs, prob)
    out = {"n_samples": n_samples,
           "root": {**z_score(root_cnt["success"] / n_samples, m["p_success_total"], n_samples),
                    "outcome_counts": root_cnt},
           "forced_entries": {}}
    for entry in two_tracked_entries(m, rec):
        seqs = sample(lm, prob, n_samples, rng, prefix=entry + NL)
        cnt = outcome_counts(seqs, prob)
        e = m["entries"][entry]["e"]
        out["forced_entries"][entry] = {
            **z_score(cnt["success"] / n_samples, e if e is not None else 0.0, n_samples),
            "outcome_counts": cnt, "p_entry": m["entries"][entry]["p_entry"]}
    return out


def entry_record(f, v, rec):
    return {"p_entry": v["p_entry"], "p_success": v["p_success"], "e": v["e"],
            "share": v["share"], "solvable": v["solvable"],
            "event": rec["entry_to_event"].get(f)}


def run(lm, instances, n_samples=2000, seed=0, with_crosscheck=True, verbose=True):
    rng = np.random.default_rng(seed)
    problems, zs = [], []
    n_checks_total = 0
    n_pass_entries = n_pass_events = 0
    pass_keys = []
    for i, (prob, rec) in enumerate(instances):
        t0 = time.time()
        m = measure(lm, prob)
        entries = {f: entry_record(f, v, rec) for f, v in m["entries"].items()}
        events = {k: {"p_entry": v["p_entry"], "p_success": v["p_success"], "e": v["e"],
                      "share": v["share"], "strings": sorted(v["strings"]),
                      "solvable": k in rec["solvable_events"]}
                  for k, v in m["events"].items()}
        good_e = passes_filter(entries)
        good_v = passes_filter(events)
        ok_e = len(good_e) >= FILTER_MIN_ENTRIES
        ok_v = len(good_v) >= FILTER_MIN_ENTRIES
        n_pass_entries += ok_e
        n_pass_events += ok_v
        if ok_e:
            pass_keys.append(prob.key)
        row = {
            "key": prob.key, "numbers": list(prob.numbers), "target": prob.cfg.target,
            "p_success_total": m["p_success_total"], "p_fail_total": m["p_fail_total"],
            "p_overlength": m["p_overlength"],
            "p_first_line_semantically_wrong": m["p_first_line_semantically_wrong"],
            "n_success_strings": m["n_success_strings"],
            "solvable_entries": rec["solvable_entries"],
            "solvable_events": rec["solvable_events"],
            "entries": entries, "events": events,
            "filter": {"entries_in_band": sorted(good_e), "events_in_band": sorted(good_v),
                       "passes_entry_level": bool(ok_e), "passes_event_level": bool(ok_v)},
            "seconds_exact": round(time.time() - t0, 3),
        }
        if with_crosscheck:
            row["crosscheck"] = crosscheck(lm, prob, m, rec, n_samples, rng)
            checks = [row["crosscheck"]["root"]] + list(row["crosscheck"]["forced_entries"].values())
            n_checks_total += len(checks)
            zs += [abs(c["z"]) for c in checks if c["informative"] and c["z"] is not None]
        problems.append(row)
        if verbose:
            cc = (f" root_z {row['crosscheck']['root']['z']}"
                  if with_crosscheck and row["crosscheck"]["root"]["z"] is not None else "")
            print(f"[{i + 1:>3}/{len(instances)}] {prob.key:<14} "
                  f"p_succ {m['p_success_total']:.3e}  "
                  f"wrong_first {m['p_first_line_semantically_wrong']:.4f}  "
                  f"filter {'PASS' if ok_e else 'fail'}  {row['seconds_exact']:.1f}s{cc}",
                  flush=True)
    summary = {
        "n_problems": len(instances),
        "filter": {"e_lo": FILTER_E_LO, "e_hi": FILTER_E_HI,
                   "p_entry_min": FILTER_P_ENTRY_MIN, "min_routes": FILTER_MIN_ENTRIES,
                   "n_pass_entry_level": n_pass_entries, "n_pass_event_level": n_pass_events,
                   "passing_keys_entry_level": pass_keys},
        "crosscheck": {"n_checks": n_checks_total, "n_informative": len(zs),
                       "max_abs_z": max(zs) if zs else None,
                       "mean_abs_z": float(np.mean(zs)) if zs else None,
                       "n_abs_z_gt_3": int(sum(z > 3 for z in zs))} if with_crosscheck else None,
        "mean_p_success_total": float(np.mean([p["p_success_total"] for p in problems])),
        "mean_p_first_line_semantically_wrong":
            float(np.mean([p["p_first_line_semantically_wrong"] for p in problems])),
    }
    return problems, summary


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    paths.add_paths_arg(ap)
    ap.add_argument("--backend", default="mock", choices=backends.CHOICES)
    ap.add_argument("--instances", default=None,
                    help="instance file (default: the frozen instances.json in the repository)")
    ap.add_argument("--out-dir", default=None, help="output directory (default: OUTPUT_DIR)")
    ap.add_argument("--tag", default=None, help="extra token in the output file name")
    ap.add_argument("--n-samples", type=int, default=2000,
                    help="samples per cross-check (root, and each tracked entry)")
    ap.add_argument("--seed", type=int, default=0, help="sampling seed, and mock backend seed")
    ap.add_argument("--limit", type=int, default=None, help="use only the first N instances")
    ap.add_argument("--no-crosscheck", action="store_true", help="exact quantities only")
    ap.add_argument("--batch-size", type=int, default=64, help="LM batch size (hf backends)")
    ap.add_argument("--boost", type=float, default=3.0,
                    help="mock backend only: logit bonus on success prefixes. Exists so the "
                         "cross-check machinery can be exercised at informative sample sizes; "
                         "it is a property of the mock, never of a measurement.")
    args = ap.parse_args()

    cfg_paths = paths.load_paths(args.paths)
    inst_path = args.instances or str(paths.REPO_ROOT / "instances.json")
    out_dir = args.out_dir or str(paths.require_dir(cfg_paths, "OUTPUT_DIR"))
    paths.require_dir({"OUTPUT_DIR": out_dir, "__file__": "--out-dir"}, "OUTPUT_DIR")

    meta_inst, instances = load_instances(inst_path)
    if args.limit:
        instances = instances[:args.limit]
    lm, backend_desc = backends.build(args.backend, cfg_paths, seed=args.seed,
                                      boost=args.boost, batch_size=args.batch_size)

    t0 = time.time()
    problems, summary = run(lm, instances, n_samples=args.n_samples, seed=args.seed,
                            with_crosscheck=not args.no_crosscheck)
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    name = "_".join(x for x in ["measure", args.backend, args.tag, stamp] if x) + ".json"
    out_path = f"{out_dir.rstrip('/')}/{name}"
    doc = {
        "schema": "entry-or-exec/measure/1",
        "created_utc": stamp, "git_commit": git_commit(),
        "python": platform.python_version(), "host": platform.node(),
        "instances_file": inst_path, "instances_md5": file_md5(inst_path),
        "instances_meta": {k: v for k, v in meta_inst.items() if k != "instances"},
        "backend": backend_desc,
        "args": vars(args) | {"paths_file": cfg_paths["__file__"]},
        "seconds_total": round(time.time() - t0, 1),
        "summary": summary, "problems": problems,
    }
    with open(out_path, "w") as fh:
        json.dump(doc, fh, indent=1, default=float)
        fh.write("\n")

    print()
    print(f"problems                              {summary['n_problems']}")
    print(f"mean p_success_total                  {summary['mean_p_success_total']:.4e}")
    print("mean p_first_line_semantically_wrong  "
          f"{summary['mean_p_first_line_semantically_wrong']:.4f}")
    f = summary["filter"]
    print(f"pre-registered filter (>= {f['min_routes']} routes, e in "
          f"[{f['e_lo']}, {f['e_hi']}], p_entry >= {f['p_entry_min']}):")
    print(f"  pass at entry level                 {f['n_pass_entry_level']}/{summary['n_problems']}")
    print(f"  pass at event level                 {f['n_pass_event_level']}/{summary['n_problems']}")
    if summary["crosscheck"]:
        c = summary["crosscheck"]
        print(f"cross-check: {c['n_checks']} comparisons, {c['n_informative']} informative "
              f"(>= 5 expected successes and failures)")
        if c["n_informative"]:
            print(f"  over the informative ones: max |z| {c['max_abs_z']:.2f}, "
                  f"mean |z| {c['mean_abs_z']:.2f}, {c['n_abs_z_gt_3']} with |z| > 3")
        else:
            print("  no comparison had enough expected successes to be informative at this n")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    paths.run_main(main)
