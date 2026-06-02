#!/bin/bash
#SBATCH --partition=small
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:30:00
#SBATCH --output=kv-%j.out
#SBATCH --job-name=spec-kv

# KV-cache correctness + speedup on the real Llama 3.2 1B/3B pair (full precision).
# kv_compare.py first verifies cached output is token-identical to non-cached
# (losslessness), then benchmarks cached vs non-cached for baseline + speculative.

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
       echo '=== starting KV-cache comparison ==='
       USE_CLUSTER=1 python kv_compare.py
       echo '=== outputs ==='
       ls -la kv_results.json 2>/dev/null
     "
