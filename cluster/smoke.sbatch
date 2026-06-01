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

cd /home/tawekith/cs153

srun --container-image='nvcr.io#nvidia/pytorch:24.12-py3' \
     --container-mounts=/home/tawekith:/home/tawekith \
     --container-workdir=/home/tawekith/cs153 \
     bash -lc '
       set -e
       export HOME=/home/tawekith
       export PYTHONUSERBASE=/home/tawekith/.local
       export PYTHONPATH=/home/tawekith/cs153:$PYTHONPATH
       echo "=== installing pinned deps (compatible with container torch) ==="
       pip install -q --user "transformers==4.46.3" "accelerate==1.1.1" matplotlib tqdm
       echo "=== running smoke_test.py ==="
       USE_CLUSTER=1 DRAFT_MODEL=distilgpt2 TARGET_MODEL=gpt2 MPLBACKEND=Agg python cluster/smoke_test.py
     '
