# entry-or-exec

**When RL post-training makes a correct solution route rarer, has the model only stopped
*choosing* that route, or has it also become worse at *finishing* the route once forced to enter
it?**

GRPO on reasoning tasks reliably narrows the set of solutions a model produces. Measuring that
narrowing by sampling cannot separate the two explanations, because a route that is no longer
chosen is also no longer observed. This repository measures them apart, by conditioning the
policy on a specific first line and asking, exactly, how much probability mass remains on
*every* correct continuation from there.

* **entry-only collapse** — entry probability and share among correct answers fall, while
  fixed-entry completion `e` is unchanged. The model still finishes the route as well as before;
  it has only stopped selecting it.
* **entry-and-execution collapse** — `e` falls too. The model has also become worse at the route
  it no longer takes.

Task: four-number Countdown, reach 24 in three lines of the form `a?b=c`.
Model: `Qwen/Qwen2.5-1.5B-Instruct` with LoRA.

## Scope

This is **step zero**: a measurement loop on a frozen model, then a short training smoke test to
show the GRPO plumbing runs and logs what the later comparison needs. It deliberately contains
**no** mechanism attribution, no optimizer studies, no second task, and no anti-collapse method.
`train_smoke.py` is not a sweep: it takes one learning rate and one KL coefficient per run, and
settings must not be chosen by looking at what happens to a route.

## The policy — "Policy A"

At every token position the next symbol is masked to the grammar-allowed symbols, renormalised
over that mask, and the conditional probabilities are multiplied along the string. One symbol is
one token. Temperature 1, no top-k, no top-p. The model writes each arithmetic result itself.

```
SYMS = 0 1 2 3 4 5 6 7 8 9 + - * / = \n    (V = 16)
numbers: 1-2 digits, no leading zero   lines: 3   symbols: at most 27
```

Consequences that the whole design rests on:

* **Format errors have probability zero by construction.** The policy cannot emit a space, a
  word, a negative sign, a leading zero, a three-digit number or a fourth line.
* **Semantically wrong lines stay possible and count as failure.** Wrong operands, a
  non-divisible division, and a correct operation with a wrongly written result are all
  `semantic_fail`.
* **Every intermediate must be an integer from 0 to 99, which excludes some standard Countdown
  solutions.** Two digits and no sign means a negative, a fraction or a value above 99 cannot be
  written at all, so a route through one is not a solution here. Of the 495 four-multisets of
  1..9, 7 reach 24 in ordinary Countdown but have no solution under this grammar — `(1,5,5,5)`
  needs `5*(5-1/5)`, `(3,3,8,8)` needs `8/(3-8/3)`. Consequently **"all correct suffixes" always
  means all correct suffixes under this grammar**, and the instance set is built with the same
  rule: `make_instances.py` requires two distinct solvable first-step events *under this grammar*,
  and the `success_strings` and `solvable_entries` in `instances.json` are the same restricted
  sets. Every number reported is about this action space, never about the ordinary game.
* **Overlength has probability zero**, because a line is at most 9 symbols and the cap is
  3 × 9 = 27. The class still exists and is still accounted for; unfinished generations are
  never dropped or renormalised away.

`protocol.py` is the single definition. The sampler, the teacher-forced scorer, the tabular copy,
the exhaustive tree walk and the differentiable log-prob used for GRPO all go through
`allowed_mask` and `masked_log_softmax` from that file. There is no second notion of probability
anywhere in the repository, and `protocol.py`, `policy.py`, `hf_lm.py` and `test_protocol.py` are
committed unmodified.

Full specification: **[PROTOCOL.md](PROTOCOL.md)**.

## The four measured quantities

For a problem, an **entry** is a specific first-line string such as `6/2=3`, and an **event** is
the canonical first action with operand order ignored for `+` and `*`, so `6+4=10` and `4+6=10`
are one event. Event-level numbers are **sums over the strings denoting the same action**.

| quantity | meaning |
| --- | --- |
| `p_entry` | probability the policy writes that first line |
| `p_success` | absolute probability mass of successful strings beginning with that entry |
| `share` | `p_success` divided by total success probability — share among correct answers |
| `e` | `p_success / p_entry` — fixed-entry completion, summed over **all** correct suffixes |

Plus `p_first_line_semantically_wrong`: the mass on a grammatical but semantically invalid
opening line, which separates "did not choose the route" from "cannot write a legal line at all".

All four are computed **exactly**, by teacher-forced scoring of every successful string, never by
sampling. `measure.py` additionally samples, only to confirm the exact numbers. `e` sums over
every correct suffix *in this action space* — see the integer 0..99 restriction above.

## Status

| | implemented | tested locally on mock / tiny GPT-2 | tested on real model |
| --- | --- | --- | --- |
| `protocol.py` — grammar, semantics, Policy A (frozen) | yes | yes | **no** |
| `policy.py` — sampler, scorer, tabular copy, `measure`, `mass_conservation` (frozen) | yes | yes | **no** |
| `hf_lm.py` — HF backend, `seq_logp_torch`, `load_qwen` (frozen) | yes | yes, tiny GPT-2 only | **no** |
| `make_instances.py` + frozen `instances.json` | yes | yes | **no** |
| `measure.py` — four quantities, sampling cross-check, filter count | yes | yes, mock | **no** |
| `batched.py` — batched exhaustive walk | yes | yes, pinned to the frozen walk | **no** |
| `check_real_model.py` — tokenizer, d, e, f, mass conservation | yes | yes, tiny GPT-2 (tokenizer check N/A) | **no** |
| `train_smoke.py` — GRPO + LoRA r16 | yes | yes, tiny GPT-2 | **no** |
| `scripts/run.sbatch` | yes | syntax only, never submitted | **no** |

`load_qwen` has never been called, no Qwen weights have been downloaded, and no GPU has been
used. **Nothing in the third column may be changed to "yes" without pasting the run output that
justifies it.** The two local backends are a deterministic pseudo-random mock and a randomly
initialised 2-layer GPT-2; neither has any property of a real LLM, and neither solves Countdown.

Local test suite: **41 passed** — the 12 frozen tests in `test_protocol.py`, 7 in
`test_batched.py`, 7 in `test_train_smoke.py`, 15 in `test_kl.py`.

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Python 3.12 was used locally. The repository is a flat set of scripts run from the repository
root, not an installable package; `pyproject.toml` only carries tool configuration.

## Placeholders

Every machine-specific value lives in exactly one file, **`config/paths.example.yaml`**, read by
`paths.py`, which every script uses. Nothing else in the repository contains a directory name, a
partition or a model location.

```bash
cp config/paths.example.yaml config/paths.yaml   # gitignored; then replace every <PLACEHOLDER>
```

| placeholder | what it is |
| --- | --- |
| `<PROJECT_ROOT>` | absolute path of this checkout on the cluster |
| `<DATA_DIR>` | generated, non-frozen data (calibration instance files) |
| `<OUTPUT_DIR>` | where `measure.py` and `train_smoke.py` write runs, and SLURM logs |
| `<HF_CACHE_DIR>` | exported as `HF_HOME` so weights cache on scratch, not `$HOME` |
| `<MODEL_PATH>` | `Qwen/Qwen2.5-1.5B-Instruct`, or a local snapshot directory |
| `<CONDA_ENV>` | environment holding the `requirements.txt` dependencies |
| `<SLURM_PARTITION>` | `#SBATCH --partition` |
| `<SLURM_ACCOUNT>` | `#SBATCH --account` |
| `<SLURM_GPU_TYPE>` | GPU type in `#SBATCH --gres=gpu:<type>:1` |

Scripts abort with a message naming the key and the file if one they need is still unfilled.
`scripts/run.sbatch` duplicates four of these values in its `#SBATCH` lines, because SLURM
directives cannot read the yaml; it refuses to run while they are still placeholders. That
duplication is the only one, and the file says so.

Override the location of the paths file with `--paths` or `$PILOT_PATHS`.

## How to run things

### Tests

```bash
pytest                         # all 41
pytest test_protocol.py -v     # the 12 frozen tests
pytest test_kl.py -v           # the KL term against its analytic gradient
```

### Instances (already committed — only rerun to verify reproducibility)

```bash
python make_instances.py                 # 32 problems, target 24, seed 0 -> instances.json
python make_instances.py --target 10     # calibration set -> <DATA_DIR>/instances_target10.json
```

All 495 four-multisets of 1..9 are enumerated; 356 have at least two distinct canonical
first-step events reaching 24; 32 are sampled with seed 0. `instances.json` is **frozen** —
rerunning with the default arguments reproduces it byte for byte. Each record carries every
successful string, every solvable entry, and the entry-to-event map.

### Measurement

```bash
python measure.py --backend mock --out-dir /tmp/out        # locally, validates the code
python measure.py --backend qwen                           # on the cluster
```

Writes one JSON per run with all four quantities for every entry and every event of every
problem, plus `p_first_line_semantically_wrong`, and per problem:

* a sampling cross-check — `N` samples from the root against the exact total success probability,
  and `N` forced-entry samples for the lowest- and highest-`p_entry` solvable entry against the
  exact `e` — reported as `z = (sampled − exact) / standard error`. A comparison with fewer than
  5 expected successes is marked `informative: false` and excluded from the summary, because at
  that sample size it would show agreement no matter what;
* whether the problem passes the **pre-registered filter**: at least two solvable entries with
  `e` in `[0.2, 0.8]` and `p_entry` at least `0.05`.

The filter count is a property of the frozen model. It is reported as found; nothing — prompt,
instance sample, mock parameters — may be tuned to raise it, and a low count is a result. On the
mock backend the count is **0/32**, as expected: the mock is pseudo-random noise with a bonus on
success prefixes, so its `e` values are either near zero or near one, never in the band.

### Real-model checks — run these FIRST on the cluster

```bash
python check_real_model.py --backend qwen           # tokenizer, d, e, f, mass conservation
python check_real_model.py --backend tiny           # exercise the script without the model
```

Asserts every one of the 16 symbols is a single token for the Qwen tokenizer, then runs tests d
(differentiable log-prob equals the teacher-forced scorer), e (batch/padding invariance), f
(tabular copy reproduces the policy) and mass conservation on two problems, plus the float32 /
bfloat16 precision comparison, and prints the measured numbers, not just pass/fail.

The mass-conservation walk visits every grammar-and-semantics-reachable prefix — about 10 000
nodes per problem — so it is the slow check. `batched.py` walks the same tree breadth-first so
each LM call covers many prefixes (about 10 800 nodes in 59 calls at `--mass-batch-size 256`);
`test_batched.py` pins it to the frozen recursive version at 1e-12 and checks the result does not
depend on batch size. Use `--skip-mass` to get the fast checks first.

Tolerances for d and e are 1e-3 there, against 1e-5 in `test_protocol.py`, because Qwen runs in
bfloat16 and the two code paths group sequences into batches differently. If the printed maxima
are much larger than the tiny model's ~2e-6, that is a finding about precision, not a pass.

**Read the precision numbers that check prints.** `load_qwen` now defaults to `float32`, and
`check_real_model.py` measures what `bfloat16` would cost: it scores the same 20 sequences with a
float32 and a bfloat16 copy of the model and prints the max absolute difference in sequence
log-prob. That difference is reported, never asserted — the number is the finding.

The reason the default changed: in `bfloat16` the 16 symbol logits are rounded to 8 mantissa bits
before Policy A ever sees them, and the error compounds over up to 27 masked conditionals.
Emulating a bf16 logit head on the tiny model, with the logits rescaled to stand in for a trained
model's larger magnitudes, gives:

| logit magnitude scale | max error in log pi(seq) | worst error in pi(seq) |
| --- | --- | --- |
| 1 (the tiny model's own) | 1.9e-3 nats | 0.2% |
| 5 | 1.0e-2 nats | 1.0% |
| 10 | 4.3e-2 nats | 4.4% |
| 20 | 1.1e-1 nats | 11.2% |

A trained 1.5B model's selected logits sit at the bottom of that table, not the top. A 10% error
on `p_success` is tolerable for a probability reported to one significant figure; it is not
tolerable for a claim that `e` changed by a few percent between two checkpoints, which is exactly
the claim this project exists to make. Hence `float32` by default (1.5B parameters is about 6 GB
of weights, and the measurement loop is not memory bound). `bfloat16` is still available —
`load_qwen(dtype=torch.bfloat16)`, or `backends.build(..., dtype="bfloat16")` — and is the right
choice for throughput once a run only needs `p_success` to one significant figure. The table above
is an emulation on a tiny model; the real number comes from the precision check, so read that
before switching. The check loads a second copy of the model, so `--skip-precision` turns it off.

### Training smoke test

```bash
python train_smoke.py --backend tiny --steps 4 --out-dir /tmp/out   # local plumbing
python train_smoke.py --backend qwen --lr 1e-5 --kl-coef 0.01       # on the cluster
```

Minimal GRPO: LoRA rank 16, 8 rollouts per problem from `policy.sample`, binary reward from
`protocol.classify`, group-mean baseline, log-probs from `HFLM.seq_logp_torch`, AdamW.
`--lr` takes `1e-5` or `5e-5`; `--kl-coef` takes `0.01` or `0`. The KL term is the **exact
per-position** `KL(pi_theta || pi_ref)`: at every position of every sampled sequence both policies
are evaluated under the same `allowed_mask` and the same masked log-softmax, and
`sum_v pi_theta(v|h) (log pi_theta(v|h) − log pi_ref(v|h))` is summed over positions and averaged
over sequences (`HFLM.kl_to_ref_torch`). The reference is the same weights with the LoRA adapter
disabled, under `no_grad`. There is no choice of estimator.

Logged per optimizer step: mean reward, the number of rollouts per first-line entry per problem,
and — separately — **whether the update contained any rollout entering each tracked entry**. The
second is not recoverable from an average and is what keeps a later reading honest: a step in
which no rollout entered a route carries no gradient signal about that route and must not be read
as evidence about it. At evaluation points the exact `policy.measure` output on the frozen
instances is logged in full.

Two things to know when reading a local run:

* on a randomly initialised model every reward is 0, so every advantage is 0 and the gradient is
  **exactly** zero. The run proves the loop executes and logs correctly, nothing more. The
  optimizer path is tested in `test_train_smoke.py`, which patches `classify` so a group contains
  both outcomes and then asserts the LoRA parameters actually move;
* the KL term is exactly 0 on the first step whatever the coefficient, because LoRA initialises
  `B` at zero, so `pi_theta` and `pi_ref` are the same distribution — bit-exactly zero, not merely
  small, which `test_kl.py` asserts. It becomes positive once the adapter has moved.

### SLURM

`scripts/run.sbatch` is a template, not a runnable script. Copy it, fill in the `#SBATCH` lines
and the two shell placeholders, then:

```bash
sbatch scripts/run.mine.sbatch check      # do this first and read the output
sbatch scripts/run.mine.sbatch measure
sbatch scripts/run.mine.sbatch train
```

## Known limitations

Things that are true of this repository as it stands, and must be dealt with before the real
experiment rather than discovered during it. None of them is a bug; all of them would change a
conclusion if ignored.

* **`train_smoke.py` trains and evaluates on the same 32 problems.** The evaluation at each
  checkpoint is `policy.measure` over the whole frozen instance set, which is also the set the
  rollouts come from. That is fine for a plumbing test, and useless as evidence: a change in `e`
  measured on the training problems cannot distinguish a change in the policy from fitting those
  problems. The real experiment needs a held-out set, reported separately from the training set.
* **The KL term is exact, and the sampling of the states it is evaluated at is not.** The term is
  the exact per-position `KL(pi_theta || pi_ref)` over the allowed symbols, summed over the
  positions of each sampled sequence and averaged over sequences, so it is differentiable in
  `theta` directly and no sampling noise enters the KL at a visited position. What remains
  approximate is *which* positions are visited: the prefixes come from the current policy's
  rollouts, and the gradient of that visitation distribution is not taken. That is the standard
  choice and is almost certainly fine at these coefficients, but it means the term is a KL over
  the states the policy actually reaches, not over the whole tree — worth stating plainly rather
  than describing the term as simply "the KL". Still watch the logged `kl` in the smoke run: it is
  exactly 0 on the first step (LoRA initialises `B` at zero) and should grow smoothly.

  This replaced two estimators that a reviewer showed were not gradients of the stated objective
  at all, because both treated the sampled symbols as fixed. For `pi_theta = (0.8, 0.2)` against
  `pi_ref = (0.5, 0.5)` the gradient on the first logit is `0.221807`; `plain` gave `0` (a score
  function with no baseline, zero in expectation) and k3's `exp(r) − r − 1` gave `0.3`, which is
  the gradient of `KL(pi_ref || pi_theta)` — the wrong direction. `test_kl.py` pins the exact term
  to the analytic gradient, that example included, and there is no longer a choice of estimator.
* **`--max-grad-norm 1.0` and `--weight-decay 0.0` are implementation defaults, not choices.**
  They were picked while writing the script because something had to be passed — `weight_decay`
  explicitly because torch's `AdamW` defaults it to `1e-2`, which moves every LoRA parameter even
  on a step whose advantage is zero. Neither has been justified or varied. Both must be fixed
  deliberately, and recorded, before the real experiment; they are logged in every run header so a
  later run can be checked against the one it is compared with.

## Layout

```
protocol.py            FROZEN  grammar, semantics, Policy A, outcome classes
policy.py              FROZEN  sampler, exact scorer, tabular copy, measure, mass_conservation
hf_lm.py               FROZEN  HF backend, differentiable Policy A log-prob, load_qwen
test_protocol.py       FROZEN  the 12 tests a-g + termination / mass conservation
paths.py                       the only reader of config/paths.example.yaml
backends.py                    mock / tiny GPT-2 / qwen construction
batched.py                     breadth-first mass_conservation for the real model
make_instances.py              builds the frozen instance set
instances.json         FROZEN  32 problems, target 24, seed 0
measure.py                     the measurement loop + sampling cross-check + filter count
check_real_model.py            tokenizer + d + e + f + mass conservation on the real backend
train_smoke.py                 minimal GRPO with LoRA
test_batched.py                batched walk == frozen recursive walk
test_train_smoke.py            the GRPO step actually produces a gradient
test_kl.py                     the KL term == the analytic gradient of KL(pi_theta || pi_ref)
config/paths.example.yaml      every machine-specific value, nowhere else
scripts/run.sbatch             SLURM template
PROTOCOL.md                    the full specification
```

Model weights, caches and run outputs are gitignored and never committed.
