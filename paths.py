"""Single reader for config/paths.example.yaml (or the filled-in config/paths.yaml).

No other module in this repository may contain a machine-specific directory name, partition,
account or model location. Scripts ask for exactly the keys they need; an unfilled
<PLACEHOLDER> for a needed key is a hard error with a message naming the key and the file.
"""
import os
import re
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent
EXAMPLE = REPO_ROOT / "config" / "paths.example.yaml"
LOCAL = REPO_ROOT / "config" / "paths.yaml"

KEYS = ["PROJECT_ROOT", "DATA_DIR", "OUTPUT_DIR", "HF_CACHE_DIR", "MODEL_PATH",
        "CONDA_ENV", "SLURM_PARTITION", "SLURM_ACCOUNT", "SLURM_GPU_TYPE"]

_PLACEHOLDER = re.compile(r"<[A-Z_]+>")


class UnfilledPlaceholder(RuntimeError):
    pass


def config_file(explicit=None):
    """Which paths file to read: --paths, then $PILOT_PATHS, then paths.yaml, then the example."""
    for cand in (explicit, os.environ.get("PILOT_PATHS")):
        if cand:
            p = Path(cand).expanduser()
            if not p.is_file():
                raise FileNotFoundError(f"paths file not found: {p}")
            return p
    return LOCAL if LOCAL.is_file() else EXAMPLE


def load_paths(explicit=None):
    """Raw dict of every key, placeholders included. Use `require` to consume values."""
    path = config_file(explicit)
    with open(path) as fh:
        cfg = yaml.safe_load(fh) or {}
    missing = [k for k in KEYS if k not in cfg]
    if missing:
        raise RuntimeError(f"{path} is missing keys: {', '.join(missing)}")
    cfg["__file__"] = str(path)
    return cfg


def require(cfg, *keys):
    """Values for `keys`, erroring if any is still an unfilled <PLACEHOLDER>."""
    bad = [k for k in keys if _PLACEHOLDER.search(str(cfg.get(k, "")))]
    if bad:
        raise UnfilledPlaceholder(
            "unfilled placeholder(s) " + ", ".join(bad) + " in " + cfg["__file__"] + "\n"
            "  cp config/paths.example.yaml config/paths.yaml and replace every <PLACEHOLDER>,\n"
            "  or point --paths / $PILOT_PATHS at a filled-in file."
        )
    out = [str(cfg[k]) for k in keys]
    return out[0] if len(out) == 1 else out


def require_dir(cfg, key, create=True):
    """Directory value for `key`, created if missing."""
    d = Path(require(cfg, key)).expanduser()
    if create:
        d.mkdir(parents=True, exist_ok=True)
    return d


def add_paths_arg(ap):
    ap.add_argument("--paths", default=None,
                    help="paths yaml to read (default: config/paths.yaml, else the example)")
    return ap
