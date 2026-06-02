#!/bin/bash
#SBATCH --partition=small
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:40:00
#SBATCH --output=sweep-%j.out
#SBATCH --job-name=spec-sweep

# Draft-size sweep: fixed Qwen2.5-7B target, drafts {0.5B, 1.5B, 3B}.
# Qwen2.5 is ungated and shares one tokenizer across sizes (required for
# speculative decoding). No HF token strictly needed, but we export it anyway.

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
       export MPLBACKEND=Agg
       [ -f \"\$HOME/.hf_token\" ] && export HF_TOKEN=\$(cat \"\$HOME/.hf_token\")
       pip install -q --user 'transformers==4.46.3' 'accelerate==1.1.1' matplotlib tqdm
       echo '=== starting draft-size sweep ==='
       python draft_sweep.py
       echo '=== outputs ==='
       ls -la draft_sweep.json draft_sweep.png 2>/dev/null
     "
