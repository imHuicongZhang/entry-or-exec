"""Action protocol and restricted policy definition (Policy A). FROZEN: see PROTOCOL.md.

Everything that touches probabilities (sampler, teacher-forced scorer, tabular copy,
torch log-prob used for GRPO) must go through `allowed_mask` + masked log-softmax
defined here. Temperature is 1. No top-k / top-p.
"""
from dataclasses import dataclass
from itertools import permutations
import numpy as np

SYMS = list("0123456789+-*/=") + ["\n"]
SID = {s: i for i, s in enumerate(SYMS)}
V = len(SYMS)
OPS = "+-*/"
NL = "\n"


@dataclass(frozen=True)
class Config:
    n_numbers: int = 4          # episode has n_numbers-1 action lines
    target: int = 24
    max_digits: int = 2         # every written number has 1..max_digits digits, no leading zero
    digits: str = "0123456789"  # digit alphabet (reduced only in unit tests)

    @property
    def n_lines(self):
        return self.n_numbers - 1

    @property
    def max_line_len(self):
        return 3 * self.max_digits + 3  # a op b = r \n

    @property
    def max_len(self):
        return self.n_lines * self.max_line_len

    @property
    def max_value(self):
        return 10 ** self.max_digits - 1


@dataclass(frozen=True)
class Problem:
    numbers: tuple
    cfg: Config = Config()

    @property
    def key(self):
        return ",".join(map(str, self.numbers)) + f"->{self.cfg.target}"


# ---------------------------------------------------------------- grammar
def _num_next(num, cfg, terminators):
    """Allowed symbols while writing a number whose digits so far are `num`."""
    if len(num) == 0:
        return list(cfg.digits)
    if num == "0" or len(num) >= cfg.max_digits:
        return list(terminators)
    return list(cfg.digits) + list(terminators)


def allowed_symbols(line, cfg):
    """Grammar-allowed next symbols given the current (unfinished) line."""
    op_pos = next((i for i, c in enumerate(line) if c in OPS), -1)
    if op_pos < 0:
        return _num_next(line, cfg, OPS)
    if "=" not in line:
        return _num_next(line[op_pos + 1:], cfg, "=")
    return _num_next(line.split("=")[1], cfg, NL)


def split_prefix(prefix):
    """-> (number of finished lines, current unfinished line)."""
    parts = prefix.split(NL)
    return len(parts) - 1, parts[-1]


def is_terminal(prefix, cfg):
    return split_prefix(prefix)[0] >= cfg.n_lines


def allowed_mask(prefix, cfg):
    """Boolean mask over SYMS for the next symbol. All-False iff terminal."""
    m = np.zeros(V, dtype=bool)
    done, line = split_prefix(prefix)
    if done >= cfg.n_lines:
        return m
    for s in allowed_symbols(line, cfg):
        m[SID[s]] = True
    return m


def masked_log_softmax(logits, mask):
    """THE policy definition (numpy). logits: [..., V]; mask: [..., V] bool."""
    x = np.where(mask, np.asarray(logits, dtype=np.float64), -np.inf)
    mx = x.max(axis=-1, keepdims=True)
    return x - mx - np.log(np.exp(x - mx).sum(axis=-1, keepdims=True))


# ---------------------------------------------------------------- semantics
def parse_line(line):
    op_pos = next(i for i, c in enumerate(line) if c in OPS)
    a, rest = line[:op_pos], line[op_pos + 1:]
    b, r = rest.split("=")
    return int(a), line[op_pos], int(b), int(r)


def compute(a, op, b, cfg):
    """Exact integer arithmetic. None if the operation is not permitted."""
    if op == "+":
        res = a + b
    elif op == "-":
        res = a - b
    elif op == "*":
        res = a * b
    else:
        if b == 0 or a % b:
            return None
        res = a // b
    return res if 0 <= res <= cfg.max_value else None


def apply_line(state, line, cfg):
    """state: sorted tuple of remaining numbers. -> new state, or None if semantically wrong."""
    a, op, b, r = parse_line(line)
    rest = list(state)
    for x in (a, b):
        if x not in rest:
            return None
        rest.remove(x)
    res = compute(a, op, b, cfg)
    if res is None or res != r:
        return None
    return tuple(sorted(rest + [res]))


def valid_lines(state, cfg):
    """Set of semantically valid line strings (without newline) from `state`."""
    out = set()
    for i, j in permutations(range(len(state)), 2):
        a, b = state[i], state[j]
        for op in OPS:
            res = compute(a, op, b, cfg)
            if res is not None:
                out.add(f"{a}{op}{b}={res}")
    return out


def canonical_event(line):
    """Action event of a line: operand order is ignored for commutative ops."""
    a, op, b, _ = parse_line(line)
    if op in "+*" and a > b:
        a, b = b, a
    return f"{a}{op}{b}"


def classify(seq, prob):
    """Outcome of a full generated string: 'success' | 'semantic_fail' | 'overlength'."""
    cfg = prob.cfg
    if not is_terminal(seq, cfg):
        return "overlength"
    state = tuple(sorted(prob.numbers))
    for line in seq.split(NL)[:cfg.n_lines]:
        state = apply_line(state, line, cfg)
        if state is None:
            return "semantic_fail"
    return "success" if state == (cfg.target,) else "semantic_fail"


def success_trajectories(prob):
    """All distinct successful strings (each ends with newline). Deduplicated at string level."""
    cfg, out = prob.cfg, set()

    def rec(state, pre):
        if len(state) == 1:
            if state[0] == cfg.target:
                out.add(pre)
            return
        for ln in valid_lines(state, cfg):
            rec(apply_line(state, ln, cfg), pre + ln + NL)

    rec(tuple(sorted(prob.numbers)), "")
    return sorted(out)


PROMPT_TEMPLATE = (
    "Use each of the numbers {nums} exactly once with + - * / to reach {target}.\n"
    "Write exactly {k} lines. Each line combines two of the remaining numbers as "
    "a?b=c with no spaces, where ? is one of + - * / and c is the exact result. "
    "The result replaces the two numbers. All numbers must be integers from 0 to {mx}.\n"
    "Example for numbers 1 2 3 4 and target 10:\n1+2=3\n3+3=6\n6+4=10\n"
    "Now solve for numbers {nums} and target {target}. Output only the {k} lines."
)


def render_prompt(prob):
    c = prob.cfg
    return PROMPT_TEMPLATE.format(nums=" ".join(map(str, prob.numbers)), target=c.target,
                                  k=c.n_lines, mx=c.max_value)
