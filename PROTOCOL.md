# Action protocol, Policy A, and the four measured quantities

This document describes what `protocol.py` and `policy.py` already implement. It is a
description, not a proposal: the definitions in those two files are frozen, and nothing in this
repository may introduce a second notion of "the probability of a generation".

## 1. Task

Four-number Countdown. A problem is a multiset of four numbers and a target (24 by default).
The model must consume the four numbers in three lines, each line combining two of the
currently remaining numbers and replacing them by the result, so that the single remaining
number is the target. Every intermediate must be an integer from 0 to 99, which excludes some
solutions of the ordinary game; section 4 says exactly what that rules out and where it applies.

The prompt is `protocol.render_prompt`; it states the format, the integer range, and shows one
worked example with a different target (10) so the example cannot leak a solution.

## 2. Grammar

One **symbol** is one token. The symbol alphabet is

```
SYMS = 0 1 2 3 4 5 6 7 8 9 + - * / = \n      (V = 16)
```

A line has the shape `a?b=c\n` where `?` is one of `+ - * /` and `a`, `b`, `c` are written by
the model as decimal numbers of one or two digits with no leading zero (`0` itself is allowed,
`07` is not). With `Config()`:

| quantity | value |
| --- | --- |
| numbers per problem | 4 |
| lines per episode (`n_lines`) | 3 |
| digits per number (`max_digits`) | 1..2, no leading zero |
| largest writable number (`max_value`) | 99 |
| symbols per line (`max_line_len`) | 6..9 |
| symbols per episode (`max_len`) | 18..27 |

`allowed_symbols(line, cfg)` returns the grammar-allowed continuations of the current
unfinished line, and `allowed_mask(prefix, cfg)` lifts that to a boolean mask over `SYMS`.
Worked examples (`\n` shown literally):

| prefix of current line | allowed next symbols | count |
| --- | --- | --- |
| `` (empty) | `0123456789` | 10 |
| `6` | `0123456789+-*/` | 14 |
| `0` | `+-*/` | 4 |
| `12` | `+-*/` | 4 |
| `6/` | `0123456789` | 10 |
| `6/2` | `0123456789=` | 11 |
| `6/0` | `=` | 1 |
| `6/2=` | `0123456789` | 10 |
| `6/2=3` | `0123456789\n` | 11 |
| `6/2=12` | `\n` | 1 |

Two structural consequences, both relied on elsewhere:

* The mask is never empty except at a terminal prefix (three completed lines), so generation
  can never dead-end. `test_termination_grammar_bounds` checks this over 3000 random walks.
* Because a line is at most `3 * max_digits + 3 = 9` symbols and `max_len = n_lines * 9 = 27`,
  a three-line episode always completes strictly inside the cap. **Overlength therefore has
  probability zero by construction**, not by truncation, and `measure` reports
  `p_overlength: 0.0` as an identity rather than an estimate.

**Format errors have probability zero.** The policy cannot emit a space, a word, a negative
sign, a leading zero, a three-digit number, or a fourth line. Everything that remains possible
is a *semantic* question.

## 3. Policy A

At every position the raw next-token logits of the model are restricted to
`allowed_mask(prefix, cfg)`, renormalised over that mask, and the resulting conditional
probabilities are multiplied along the string:

```python
masked_log_softmax(logits, mask)               # protocol.py -- THE definition
log pi(seq) = sum_t masked_log_softmax(logits_t, allowed_mask(seq[:t]))[seq[t]]
```

Temperature is 1. There is no top-k and no top-p. Logits for the 16 symbols are read out of
the model's full vocabulary by `index_select` on a fixed list of 16 single-token ids
(`hf_lm.HFLM`); `load_qwen` asserts that each symbol really is one token for the tokenizer in
use, and `check_real_model.py` re-asserts it.

All of the following go through that one mask and that one log-softmax:

| consumer | entry point |
| --- | --- |
| sampler | `policy.sample` -> `policy.next_logp` |
| exact teacher-forced scorer | `policy.seq_logp` |
| tabular copy of the policy | `policy.TabularLM` |
| exhaustive tree walk | `policy.mass_conservation` |
| differentiable log-prob for GRPO | `hf_lm.HFLM.seq_logp_torch` |

Policy A is a statement about this restricted token-level policy. It is not a claim about what
the unrestricted model would have written, and no conclusion in this project may be phrased as
one.

## 4. Semantics

A line `a?b=c` is **semantically valid** in a state (the sorted multiset of remaining numbers)
when all of the following hold (`protocol.apply_line`, `protocol.compute`):

* `a` and `b` are both present in the state, as two separate elements (so `3+3=6` needs two
  threes);
* the operation is permitted: division requires `b != 0` and `a % b == 0`;
* the true result lies in `0 .. max_value`, that is **every intermediate must be an integer from
  0 to 99** — see the restriction below;
* the number `c` written by the model equals that true result. **The model writes the result
  itself**; a correct choice of operands with a wrong arithmetic result is a failure, not a
  repaired step.

Applying a valid line removes `a` and `b` and inserts the result.

### The integer 0..99 restriction, and what it excludes

Every intermediate value must be an integer from 0 to 99. Negative intermediates, fractional
intermediates, and values above 99 are all semantically invalid, so a line producing one is a
`semantic_fail` and no continuation of it can succeed.

**This excludes some standard Countdown solutions.** A route through a negative number
(`3-5=-2`), through a fraction (`5/2=2.5`), or through a large product (`12*9=108`, then
`108-84=24`) is a legitimate solution of the usual game and is *not* a solution here. Of the 495
four-multisets of 1..9, **7 reach 24 in ordinary Countdown but have no solution at all under this
grammar**: `(1,3,4,6)`, `(1,4,5,6)`, `(1,5,5,5)`, `(1,6,6,8)`, `(3,3,7,7)`, `(3,3,8,8)`,
`(4,4,7,7)` — for instance `(1,5,5,5)` needs `5*(5-1/5)` and `(3,3,8,8)` needs `8/(3-8/3)`, both
of which pass through a fraction. A further 41 have exactly one solvable first-step event here and
are excluded by the two-event eligibility rule rather than by the arithmetic. The
restriction comes from the grammar: two digits and no sign means there is no way to write such a
value in the first place, which is what makes format errors impossible (section 2). It is a
property of the protocol, not of the model, and it applies uniformly to every quantity:

* `success_trajectories(prob)` enumerates the successful strings **under this grammar**, so
  wherever this project says "all correct suffixes" — in particular the fixed-entry completion
  `e` of section 7 — it means all correct suffixes reachable under this grammar. There is no
  hidden correct continuation that `e` fails to count, but there are solutions of ordinary
  Countdown that are simply not in the action space;
* **instance eligibility uses the same rule.** `make_instances.py` keeps a 4-multiset only if it
  has at least two distinct canonical first-step events leading to the target *under this
  grammar*, and the `success_strings` and `solvable_entries` stored in `instances.json` are the
  same restricted sets. A multiset solvable only through a negative or a value above 99 counts as
  unsolvable here and is not in the instance set;
* `classify` applies it too, so a model that writes a mathematically correct line with an
  out-of-range intermediate is scored `semantic_fail`.

Reported numbers are therefore about this action space throughout. A statement about "the"
solution set of a Countdown problem would be a different claim and this repository does not make
it.

## 5. Outcome classes

`protocol.classify(seq, prob)` returns exactly one of three labels for every generated string:

| class | condition |
| --- | --- |
| `success` | three lines, every line semantically valid, final remaining number equals the target |
| `semantic_fail` | three lines, but some line is semantically invalid, or the final number is not the target |
| `overlength` | fewer than three completed lines (the string is unfinished) |

`overlength` has probability zero under the grammar of section 2, but the class exists and is
accounted for, because an unfinished generation must be a *failure* and must never be dropped.
`policy.sample` never resamples or discards, and `policy.mass_conservation` accumulates failure
mass explicitly at the first symbol that leaves the set of semantically valid lines. The
termination test asserts `success + semantic_fail + overlength == 1` to 1e-9 on the exhaustive
walk.

## 6. Entries and canonical events

An **entry** is a specific first-line string, including its result, without the newline — for
example `6/2=3`. Fixing an entry means conditioning the policy on the prefix `6/2=3\n`.

Two distinct strings can denote the same action. `protocol.canonical_event` maps a line to
`a?b` with the operands sorted when the operation is commutative:

```
canonical_event("6+4=10") == canonical_event("4+6=10") == "4+6"
canonical_event("6-4=2")  != canonical_event("4-6=2")
```

An **event** is a canonical first action, and the event-level quantities are *sums over the
entry strings that denote it* — never averages, and never a representative string. For
`(2,3,4,6)` there are 33 semantically valid first lines, collapsing to 21 distinct canonical
events, of which 13 can still reach 24.

Why both levels exist: the entry level is the quantity the policy actually parameterises (it is
a prefix, so conditioning on it is exact), while the event level is the quantity a claim about
"a solution route" refers to. Reporting only one of the two would either double-count a route
written two ways or hide that the model shifted which spelling it prefers.

## 7. The four measured quantities

For one problem, let `S` be the set of all successful strings (`protocol.success_trajectories`,
deduplicated at string level) and `f` an entry. `policy.measure` returns, for every entry and
every event:

| name | key | definition |
| --- | --- | --- |
| entry probability | `p_entry` | `pi(f + "\n")`, the probability the policy writes that first line |
| absolute success probability | `p_success` | sum of `pi(s)` over all `s` in `S` beginning with `f` |
| share among correct answers | `share` | `p_success / sum_{s in S} pi(s)` |
| fixed-entry completion | `e` | `p_success / p_entry` |

`e` is the quantity the research question turns on: **the probability of reaching the target
given that the route has been entered, summed over every correct suffix** — not the probability
of one particular continuation, and not a renormalised success rate. "Every correct suffix" means
every correct suffix *under this grammar*: intermediates must be integers from 0 to 99, so
suffixes that would pass through a negative, a fraction or a value above 99 are not in the action
space at all (section 4). It is computed exactly, by
teacher-forced scoring of every successful string, never by sampling; the sampling check in
`measure.py` exists only to confirm the exact numbers.

`measure` additionally returns:

* `p_success_total` — total success probability from the root;
* `p_fail_total = 1 - p_success_total`;
* `p_first_line_semantically_wrong = 1 - sum_f p_entry(f)` over the semantically valid first
  lines — the mass the policy puts on a *grammatical but semantically invalid* opening line.
  This is the measurement that separates "did not choose the route" from "cannot write a legal
  line at all";
* `p_overlength = 0.0`, as an identity;
* `n_success_strings = |S|`.

## 8. The research question in these terms

GRPO reduces `share` (and usually `p_entry`) for some correct route. The two hypotheses are:

* **entry-only** — `p_entry` and `share` fall while `e` is unchanged: the model still finishes
  the route as well as before once forced into it, it merely stops choosing it;
* **entry-and-execution** — `e` falls as well: the model has also become worse at completing
  the route it no longer takes.

Distinguishing them requires `e` at fixed entry, which is why the entry prefix is conditioned
on exactly rather than selected for by rejection sampling. This repository implements the
measurement of these quantities on a frozen model plus a training smoke test. It does not
attribute mechanism, study optimizers, add a second task, or test anti-collapse methods.

## 9. Pre-registered instance filter

A problem is *usable* for the later comparison when it has at least two solvable entries whose
fixed-entry completion `e` lies in `[0.2, 0.8]` and whose `p_entry` is at least `0.05` — wide
enough that a change in either direction is visible and not clipped by a floor or a ceiling.
`measure.py` reports how many problems pass. This threshold is fixed in advance and nothing
(prompt, instance sample, mock parameters) may be tuned to raise the count; the count is a
property of the frozen model, and reporting a low one is a result.
