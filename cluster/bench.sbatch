#!/bin/bash
#SBATCH --partition=small
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:30:00
#SBATCH --output=bench-%j.out
#SBATCH --job-name=spec-bench

# Real run: full-precision Llama 3.2 1B/3B on the H100 (no quantization).
# Runs the complete pipeline in main.py: K-sweep benchmark, domain analysis,
# plots, latency breakdown, and the token-trace HTML.
# Reads the HF token from $HOME/.hf_token (not in the repo).

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
       echo '=== starting full benchmark (USE_CLUSTER=1, full precision) ==='
       USE_CLUSTER=1 python main.py
       echo '=== outputs ==='
       ls -la results.json latency_breakdown.json *.png token_trace.html 2>/dev/null
     "
