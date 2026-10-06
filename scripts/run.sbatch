#!/bin/bash
#SBATCH --job-name=entry-or-exec
#SBATCH --partition=<SLURM_PARTITION>
#SBATCH --account=<SLURM_ACCOUNT>
#SBATCH --gres=gpu:<SLURM_GPU_TYPE>:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=<OUTPUT_DIR>/slurm-%x-%j.out
#SBATCH --error=<OUTPUT_DIR>/slurm-%x-%j.out
#
# Template for Skipjack. Every <PLACEHOLDER> above and below must be replaced, and the same
# values must be written into config/paths.yaml -- the #SBATCH lines cannot read the yaml,
# so these four are the one place a value is duplicated. Keep them in sync.
#
#   cp config/paths.example.yaml config/paths.yaml   # fill it in
#   cp scripts/run.sbatch scripts/run.mine.sbatch    # fill in the #SBATCH lines
#   sbatch scripts/run.mine.sbatch check             # tokenizer + tests d/e/f + mass conservation
#   sbatch scripts/run.mine.sbatch measure           # the measurement loop on the frozen model
#   sbatch scripts/run.mine.sbatch train             # the GRPO smoke test
#
# Run `check` first and read its output. A failure there makes every number from `measure`
# meaningless, so do not queue the others until it passes.

set -euo pipefail

STAGE="${1:-check}"
shift || true

PROJECT_ROOT="<PROJECT_ROOT>"
CONDA_ENV="<CONDA_ENV>"

# Fail loudly rather than creating a directory literally named "<PROJECT_ROOT>".
for v in PROJECT_ROOT CONDA_ENV; do
  case "${!v}" in
    "<"*">") echo "scripts/run.sbatch: $v is still the placeholder ${!v}." >&2
             echo "Copy this file, fill in every #SBATCH line and every <PLACEHOLDER>," >&2
             echo "and fill config/paths.yaml with the same values." >&2
             exit 2 ;;
  esac
done

cd "$PROJECT_ROOT"

# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

# HF_HOME is set from HF_CACHE_DIR by backends.set_hf_cache; exported here as well so that any
# tokenizer download started before that call also lands on scratch rather than in $HOME.
export HF_HOME="$(python - <<'PY'
import paths
print(paths.require(paths.load_paths(), "HF_CACHE_DIR"))
PY
)"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

echo "host      $(hostname)"
echo "stage     $STAGE"
echo "commit    $(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
echo

case "$STAGE" in
  check)
    # Slowest part is the exhaustive mass-conservation walk; raise --mass-batch-size if there is
    # GPU memory to spare, or pass --skip-mass to get the fast checks first.
    python check_real_model.py --backend qwen --n-problems 2 --batch-size 64 \
        --mass-batch-size 256 "$@"
    ;;

  measure)
    python measure.py --backend qwen --n-samples 2000 --batch-size 64 --tag frozen "$@"
    ;;

  train)
    # One setting per submission. --lr takes 1e-5 or 5e-5, --kl-coef takes 0.01 or 0.
    # This is a plumbing smoke test: do not submit an array over the four combinations and do
    # not choose between them by looking at what happens to a route.
    python train_smoke.py --backend qwen --lr 1e-5 --kl-coef 0.01 \
        --steps 20 --problems-per-step 2 --rollouts 8 --eval-every 10 \
        --batch-size 64 --tag lr1e-5_kl0.01 "$@"
    ;;

  *)
    echo "unknown stage '$STAGE' (expected: check | measure | train)" >&2
    exit 2
    ;;
esac

echo
echo "stage $STAGE finished"
