This file gives Claude Code (or any AI coding assistant) the cluster environment context for this project. The setup below is filled in and verified working.

---

## Environment

- **Cluster**: shared 32× H100 SLURM cluster, 4 nodes × 8 GPUs (H100 80GB HBM3).
- **My username on the cluster**: `tawekith` (login email `tawekith@stanford.edu`, Omniva org **AMP PBC**).
- **My home directory**: `/home/tawekith` (on a shared Weka filesystem mounted on the login pod and all workers).
- **Kubernetes cluster name (Omniva)**: `amp-internal` (id `1d76578f-ab0e-4811-a7a4-8938e634c76c`).
- **Access**: via Omniva-issued kubeconfig. Connection model is `kubectl exec` into my LoginSet pod; there is no SSH and no port-forward to external services.
- **My login pod selector**: `kubectl get pod -n slurm -l stanford/user=tawekith`
- **Pod name pattern**: `slurm-login-tawekith-<hash>` (current: `slurm-login-tawekith-695944f6f7-5qvl8`).
- **Project on the pod**: `/home/tawekith/cs153` (speculative-decoding code). Cluster sbatch scripts live in `cluster/`.

## Getting connected (one-time per machine, then per-session login)

Tooling required on the laptop (all installed): `kubectl`, the Omniva `om` CLI (`/opt/homebrew/bin/om`), and the Teleport client `tsh` (`brew install teleport`). `om` shells out to `tsh` to reach the cluster, so `tsh` MUST be present or `om create kubeconfig` fails with "tsh not found".

Per-session steps that work:

```bash
om login                                          # browser SSO as tawekith@stanford.edu; session ~24h
om create kubeconfig --k8s-cluster amp-internal   # writes/merges ~/.kube/config
kubectl auth whoami                               # expect: Username  tawekith
```

The client-application credentials in `.env` (OMNIVA_ID / OMNIVA_SECRET_KEY) are for **non-interactive / machine** login only (`om login --client-credentials-file=creds.json`) and additionally require an org bot (`om create bots`, admin-only). For interactive personal use, just `om login`. Do not commit `.env` (it is gitignored).

Quick helpers (paste in a shell after `om login`):

```bash
POD=$(kubectl get pod -n slurm -l stanford/user=tawekith -o jsonpath='{.items[0].metadata.name}')
shell() { kubectl exec -it -n slurm "$POD" -c login -- runuser -l tawekith; }
run()   { kubectl exec      -n slurm "$POD" -c login -- runuser -l tawekith -c "$*"; }
# IMPORTANT: always submit Slurm jobs via `runuser -l tawekith` so they are attributed to me,
# not to root (root has no Slurm account/budget). Use kubectl cp into a tarball + chown for code.
```

## How to do anything compute-heavy

NEVER run training, evaluation, container pulls, or other GPU-bound work inside the login pod. The login pod is a small CPU shell with no GPU. All compute goes through `sbatch` to a SLURM partition. Single-node only.

### Submitting a job

```bash
sbatch myjob.sbatch
squeue -u $USER        # see my queue
sacct -u $USER -S today
scancel <jobid>
```

### sbatch template

```bash
#!/bin/bash
#SBATCH --partition=small        # see "Partitions" below
#SBATCH --gres=gpu:1             # request N GPUs (1–8)
#SBATCH --time=00:30:00          # max walltime; required
#SBATCH --output=%x-%j.out
#SBATCH --job-name=my-run

# ... my commands here ...
```

### With a container (pyxis + enroot)

```bash
srun --gres=gpu:1 --cpus-per-task=16 \
  --container-image='nvcr.io#nvidia/pytorch:24.12-py3' \
  python train.py
```

**Two rules that will bite you if you skip them:**

- **Always pass `--cpus-per-task=16` on container jobs.** Each container job imports the image (builds a ~20 GB squashfs). That build is single-threaded by default and uses only the CPUs SLURM gave the job — with the default 2 CPUs it takes ~30 minutes; with 16 CPUs it is much faster. NOTE: in practice on `amp-internal` the import runs again on each job (~6–7 min observed), so don't assume a free cached squashfs across jobs.
- **Image reference syntax rule.** For Docker Hub images use the **bare name** (`alpine:latest`, `python:3.12-slim`). For any other registry use the `<registry>#<path>` URI form (`nvcr.io#nvidia/pytorch:24.12-py3`). The specific combination `docker.io#library/<name>` breaks enroot's manifest pipeline (JSON parse error). NGC images are still preferred for ML work — they ship CUDA + NCCL + cuDNN matched to host drivers and avoid Docker Hub rate limits.

Common working images:
- `nvcr.io#nvidia/pytorch:24.12-py3` — PyTorch 2.x + CUDA + NCCL + most ML libs
- `nvcr.io#nvidia/cuda:12.6.0-base-ubuntu22.04` — minimal CUDA base
- `nvcr.io#nvidia/cuda:12.6.0-devel-ubuntu22.04` — has nvcc + headers for compiling

### Verified container workflow for this project (learned the hard way)

These are confirmed on `amp-internal` with `nvcr.io#nvidia/pytorch:24.12-py3`:

- **The image re-imports on every job (~6–7 min each).** In practice the squashfs is NOT reused across jobs here, so budget ~7 min of import per container job (download layers are cached, the squashfs rebuild is not). Prefer one larger job over many small ones.
- **Pin `transformers==4.46.3` and `accelerate==1.1.1`.** The image ships `torch 2.6.0a0+...nv24.12`, and the *latest* `transformers` breaks on it with `ImportError: cannot import name 'TransformGetItemToIndex' from 'torch._dynamo._trace_wrapped_higher_order_op'`. 4.46.3 still supports Llama 3.2. `accelerate` is required for `device_map="auto"`.
- **Install with `pip install --user` and a mounted home so installs persist.** Mount `--container-mounts=/home/tawekith:/home/tawekith`, then inside the container `export HOME=/home/tawekith` and `export PYTHONUSERBASE=/home/tawekith/.local`. Packages land in `/home/tawekith/.local` and survive across jobs (so the second job's `pip install` is fast). The container runs as root, so this is just for persistence.
- **Set `PYTHONPATH=/home/tawekith/cs153`** when running scripts that live in a subdir (e.g. `cluster/smoke_test.py`), otherwise `import config` fails.
- **Headless plots:** `export MPLBACKEND=Agg`.
- **HF token:** gated Llama 3.2 needs it. Store at `/home/tawekith/.hf_token` (chmod 600, NOT in the repo); sbatch does `export HF_TOKEN=$(cat /home/tawekith/.hf_token)`.
- **Run modes** (see `config.py`): local CPU default (gpt2 pair); `USE_COLAB=1` → gated Llama + 4-bit; `USE_CLUSTER=1` → full-precision Llama on GPU (H100 has 80 GB, no quant needed). On the H100, run the full pipeline with `USE_CLUSTER=1 python main.py`.
- **Ready-made scripts:** `cluster/smoke.sbatch` (cheap gpt2 GPU validation, no token) and `cluster/bench.sbatch` (full Llama run). Both already encode the rules above.

### Sanity-check the cluster

If something feels broken, run this from the login pod to verify pyxis + container imports work end-to-end:

```bash
srun --partition=small --gres=gpu:1 --cpus-per-task=16 \
  --container-image='nvcr.io#nvidia/pytorch:24.12-py3' \
  python -c "import torch; print(torch.cuda.device_count())"
```

Should print `1`. If it hangs at `pyxis: importing` for more than ~5 min on the first run, ping the admin — the worker's enroot scratch directory may have lost its permissions.

## Partitions

| Partition | Max walltime | Use for |
|---|---|---|
| `small`  | 24h | single-GPU jobs, quick experiments |
| `medium` | 5d  | multi-GPU runs, longer training |
| `big`    | 5d  | large reserved slots — restricted access |

Default partition is `small`. If a longer run is needed, use `medium`.

## GPU-hour budget

I have a per-user GPU-hour cap enforced by SLURM QoS. When the cap is hit, new jobs queue indefinitely until reset.

```bash
sshare -u $USER                                    # remaining budget
sacctmgr show qos qos-$USER format=GrpTRESMins     # absolute cap (in minutes)
```

When suggesting workloads:
- Prefer fewer, larger jobs over many small ones (fewer prolog/epilog cycles).
- Check feasibility against the remaining budget before suggesting a sweep.
- Default to `--gres=gpu:1` unless multi-GPU is required; multi-GPU multiplies hour consumption.

## Storage

- `/home/tawekith` — my primary workspace. Persistent across pods + nodes.
- `/home/_shared/models` — read-only shared HuggingFace cache. Use it if a model is already there before downloading.
- `/home/_shared/datasets` — read-only shared dataset cache.
- Soft cap is ~1 TB per user. Avoid hoarding checkpoints; delete or move old runs.

## Login-pod context

- Container runs as `root` by default. To act as me (so file ownership in `/home/tawekith` is correct), run interactive shells via `runuser`:
  ```bash
  kubectl exec -it -n slurm <my-pod> -c login -- runuser -u tawekith -- bash -l
  ```
- I have these pre-installed on the login pod: `python3`, `pip`, `uv`, `git`, `git-lfs`, `gh`, `rsync`, `aws` (CLI v2), `huggingface_hub[cli]`, `wandb`, `transformers`, `tokenizers`, `datasets`, `vim`, `tmux`, `htop`, `jq`.
- For heavy Python deps (torch, vllm, deepspeed, etc.), put them in the container image used by `srun --container-image=`, not on the login pod.

## Copying files

```bash
# from my laptop into my pod's home dir
kubectl cp local-file.py slurm/<my-pod>:/home/tawekith/local-file.py -c login

# from pod back to laptop
kubectl cp slurm/<my-pod>:/home/tawekith/results.tar.gz . -c login
```

## Constraints to remember when suggesting code

- **No SSH-into-the-cluster** patterns. No `ssh slurm-controller`, no rsync-over-ssh. All file movement uses `kubectl cp` or pulls/pushes via the network (HTTPS, S3, HF Hub).
- **No port-forwarding to external services.** I can't expose a Jupyter or TensorBoard port to the public internet. For interactive notebooks, run Jupyter inside an `srun --pty` session and port-forward via `kubectl port-forward <my-pod>` to my laptop only.
- **Single-node only.** The Stanford partitions (`small`, `medium`, `big`) all have `MaxNodes=1`. Don't suggest multi-node DDP. Use 1–8 GPUs on one node with `torchrun --standalone --nproc-per-node=N`.
- **No `sudo` on the login pod** (well, I'm root, but anything I install isn't persistent across pod restarts — bake into container images or use `pip install --user`).
- **All my actions on the cluster are audit-logged in Teleport.** Don't suggest probing other namespaces or other students' pods.

## Things that will likely come up

- **NCCL / multi-GPU**: the workers have RoCE v2 over 8× ConnectX-7 NDR (400 GbE). NCCL env is pre-injected by Kyverno policy on GPU jobs. Don't override `NCCL_IB_*` or `NCCL_SOCKET_IFNAME` unless I know what I'm doing.
- **Container caching**: enroot caches imported containers under `/run/enroot/${UID}` (per-job ephemeral). The first pull of a large image takes minutes; subsequent jobs reuse the cache for the duration of the worker pod's life.
- **Out-of-memory / OOM**: SLURM allocates the full node's memory by default when you take a GPU. If a job crashes with OOM, it's usually CUDA OOM (model too big for GPU memory), not host OOM.

## Project gotchas (speculative decoding runs)

- **Validate logic-heavy code locally on CPU before spending GPU-hours.** The `venv` at the repo root has CPU torch + transformers. The KV-cache rollback had a real bug (caches desynced in the all-accepted case); `kv_compare.py` with the default gpt2/distilgpt2 caught it for free in ~40s instead of via repeated ~7-min container jobs. Each cluster job pays the image-import tax regardless, so a fast local loop is strictly cheaper for correctness work.
- **Shared tokenizer ≠ shared logit width.** The Qwen2.5 family shares one tokenizer, but the 0.5B/1.5B/3B checkpoints pad the lm_head to 151936 while 7B/14B/72B pad to 152064. Speculative decoding's residual `target_probs - draft_probs` then shape-mismatches. Fix: slice both to `min(vocab)` before the residual (a no-op when widths match, e.g. the Llama 1B/3B pair at 128256). Llama 3.2 only ships 1B/3B, which is why the draft-size sweep uses Qwen2.5.
- **Container outputs land as `root:nogroup`** in the mounted home (the enroot job runs as root). They're world-readable so `kubectl cp` back works fine, but `chown tawekith` them if a later non-root step needs to overwrite.
- **HF model weights persist** in `~/.cache/huggingface` on the Weka home across jobs (only the container squashfs re-imports each job), so re-running a benchmark does not re-download models.
- Total GPU spend for the whole study (core bench + sweep + KV + smokes) was ~0.8 GPU-hours.
