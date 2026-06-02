#!/bin/bash
#SBATCH --partition=small
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:15:00
#SBATCH --output=smoke-%j.out
#SBATCH --job-name=spec-smoke

# Cheap validation: gpt2 pair on the H100 via the NGC PyTorch container.
# No HF token required. Confirms container + deps + GPU + harness all work
# before spending GPU-hours on the real Llama run.

PROJ="$HOME/cs153"
cd "$PROJ"

srun --container-image='nvcr.io#nvidia/pytorch:24.12-py3' \
     --container-mounts="$HOME:$HOME" \
     --container-workdir="$PROJ" \
     bash -lc "
       set -e
       export HOME='$HOME'
       export PYTHONUSERBASE=\"\$HOME/.local\"
       export PYTHONPATH=\"$PROJ:\$PYTHONPATH\"
       echo '=== installing pinned deps (compatible with container torch) ==='
       pip install -q --user 'transformers==4.46.3' 'accelerate==1.1.1' matplotlib tqdm
       echo '=== running smoke_test.py ==='
       USE_CLUSTER=1 DRAFT_MODEL=distilgpt2 TARGET_MODEL=gpt2 MPLBACKEND=Agg python cluster/smoke_test.py
     "
