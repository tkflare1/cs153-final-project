#!/bin/bash
#SBATCH --partition=small
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=01:10:00
#SBATCH --output=adaptive-long-%j.out
#SBATCH --job-name=spec-adapt

# Long-context K sweep (max_new_tokens 50 & 200) plus the adaptive-K policy,
# on the full-precision Llama 3.2 1B/3B pair.

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
       echo '=== running adaptive + long-context experiments ==='
       USE_CLUSTER=1 python adaptive_long_experiments.py \
         --long-context --adaptive-k \
         --token-counts 50 200 \
         --adaptive-max-new 200 \
         --k-init 2 --k-min 1 --k-max 8 --hi 0.80 --lo 0.60
       echo '=== outputs ==='
       ls -la long_context_results.json adaptive_k_results.json \
              long_context_speedup.png adaptive_k_usage.png 2>/dev/null
     "
