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

cd /home/tawekith/cs153

srun --container-image='nvcr.io#nvidia/pytorch:24.12-py3' \
     --container-mounts=/home/tawekith:/home/tawekith \
     --container-workdir=/home/tawekith/cs153 \
     bash -lc '
       set -e
       export HOME=/home/tawekith
       export PYTHONUSERBASE=/home/tawekith/.local
       export PYTHONPATH=/home/tawekith/cs153:$PYTHONPATH
       export MPLBACKEND=Agg
       [ -f /home/tawekith/.hf_token ] && export HF_TOKEN=$(cat /home/tawekith/.hf_token)
       pip install -q --user "transformers==4.46.3" "accelerate==1.1.1" matplotlib tqdm
       echo "=== starting KV-cache comparison ==="
       USE_CLUSTER=1 python kv_compare.py
       echo "=== outputs ==="
       ls -la kv_results.json 2>/dev/null
     '
