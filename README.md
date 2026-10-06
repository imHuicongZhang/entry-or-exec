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
sampling. `measure.py` additionally samples, only to confirm the exact numbers.

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

Local test suite: **26 passed** — the 12 frozen tests in `test_protocol.py`, 7 in
`test_batched.py`, 7 in `test_train_smoke.py`.

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
pytest                         # all 26
pytest test_protocol.py -v     # the 12 frozen tests
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
(tabular copy reproduces the policy) and mass conservation on two problems, and prints the
measured numbers, not just pass/fail.

The mass-conservation walk visits every grammar-and-semantics-reachable prefix — about 10 000
nodes per problem — so it is the slow check. `batched.py` walks the same tree breadth-first so
each LM call covers many prefixes (about 10 800 nodes in 59 calls at `--mass-batch-size 256`);
`test_batched.py` pins it to the frozen recursive version at 1e-12 and checks the result does not
depend on batch size. Use `--skip-mass` to get the fast checks first.

Tolerances for d and e are 1e-3 there, against 1e-5 in `test_protocol.py`, because Qwen runs in
bfloat16 and the two code paths group sequences into batches differently. If the printed maxima
are much larger than the tiny model's ~2e-6, that is a finding about precision, not a pass.

### Training smoke test

```bash
python train_smoke.py --backend tiny --steps 4 --out-dir /tmp/out   # local plumbing
python train_smoke.py --backend qwen --lr 1e-5 --kl-coef 0.01       # on the cluster
```

Minimal GRPO: LoRA rank 16, 8 rollouts per problem from `policy.sample`, binary reward from
`protocol.classify`, group-mean baseline, log-probs from `HFLM.seq_logp_torch`, AdamW.
`--lr` takes `1e-5` or `5e-5`; `--kl-coef` takes `0.01` or `0`, where the reference policy is the
same weights with the LoRA adapter disabled, scored with the same Policy A log-probs on the same
sampled sequences (k3 estimator, `exp(r) − r − 1`).

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
  `B` at zero, so `pi_theta` and `pi_ref` are the same distribution. It becomes positive once the
  adapter has moved.

### SLURM

`scripts/run.sbatch` is a template, not a runnable script. Copy it, fill in the `#SBATCH` lines
and the two shell placeholders, then:

```bash
sbatch scripts/run.mine.sbatch check      # do this first and read the output
sbatch scripts/run.mine.sbatch measure
sbatch scripts/run.mine.sbatch train
```

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
config/paths.example.yaml      every machine-specific value, nowhere else
scripts/run.sbatch             SLURM template
PROTOCOL.md                    the full specification
```

Model weights, caches and run outputs are gitignored and never committed.
