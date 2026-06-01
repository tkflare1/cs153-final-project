# CS 153 GPU cluster — getting started

You've been provisioned access to a shared 32-H100 cluster for your CS 153 project. This doc walks you through the setup and the rules. Read both sections.

## One-time setup

### 1. Install the Omniva CLI

Follow the [Omniva prerequisites](https://docs.omniva.com/user/getting-started/access/#prerequisites). The last step there is downloading the `om` binary — once you have it in your Downloads folder, continue below to install it:

Make the downloaded binary executable:

```bash
chmod +x ~/Downloads/om
```

macOS blocks direct moves from Downloads to system paths, so copy to a temp location first:

```bash
cp ~/Downloads/om /private/tmp/om
chmod +x /private/tmp/om
```

Move it into the global executable path:

```bash
sudo mv /private/tmp/om /usr/local/bin/om
```

Verify the install:

```bash
om --version
```

### 2. Log in and get a kubeconfig

Log in (opens a browser):

```bash
om login
```

Generate your kubeconfig:

```bash
om create kubeconfig --k8s-cluster amp-internal
```

Verify it worked — should print `Username: <your-name>`:

```bash
kubectl auth whoami
```

If your username doesn't appear, stop and ping [@anthony](#contact) before doing anything else.

### 3. Find your login pod and shell in

Set your username:

```bash
USER_NAME=<your-name>   # e.g. jshunk
```

Find your pod:

```bash
POD=$(kubectl get pod -n slurm -l stanford/user=${USER_NAME} -o jsonpath='{.items[0].metadata.name}')
echo "Your pod: $POD"
```

Shell into it (`runuser` is important for correct file ownership):

```bash
kubectl exec -it -n slurm $POD -c login -- runuser -l ${USER_NAME}
```

Quality-of-life alias for your `~/.bashrc` or `~/.zshrc`:

```bash
cs153() {
  POD=$(kubectl get pod -n slurm -l stanford/user=<your-name> -o jsonpath='{.items[0].metadata.name}')
  kubectl exec -it -n slurm "$POD" -c login -- runuser -l <your-name>
}
```

### 4. Verify your environment

Inside the pod, you should see:

```bash
$ whoami
<your-name>
$ pwd
/home/<your-name>
$ sinfo
PARTITION AVAIL  TIMELIMIT  NODES  STATE NODELIST
big          up 5-00:00:00      4   idle slinky-[0-3]
medium       up 5-00:00:00      4   idle slinky-[0-3]
small*       up 1-00:00:00      4   idle slinky-[0-3]
```

## Submitting a job

### Hello world

Create a job file:

```bash
cat > hello.sbatch <<'EOF'
#!/bin/bash
#SBATCH --partition=small
#SBATCH --gres=gpu:1
#SBATCH --time=00:05:00
#SBATCH --output=hello-%j.out
hostname
nvidia-smi
EOF
```

Submit it:

```bash
sbatch hello.sbatch
```

Check that it's queued (short jobs may finish before you run this — that's fine):

```bash
squeue -u $USER
```

If the queue is empty, the job already completed. Check the output file:

```bash
cat hello-<jobid>.out
```

If you see `nvidia-smi` output, the job ran successfully.

### With a container (pyxis + enroot)

```bash
srun --gres=gpu:1 --cpus-per-task=16 \
  --container-image='nvcr.io#nvidia/pytorch:24.12-py3' \
  python -c "import torch; print(torch.cuda.device_count())"
```

**⚠ Always pass `--cpus-per-task=16` on container jobs.** The first time an image is used on a worker, enroot has to build a squashfs of it (the PyTorch image is ~20 GB). That build is single-threaded by default and uses only the CPUs SLURM gave you — with the default `--cpus-per-task=2`, it takes ~30 minutes. With 16 CPUs, ~3 minutes. After the first import the squashfs is cached on that worker, so subsequent jobs reuse it for free.

**⚠ Image reference syntax matters.** For Docker Hub images, use the **bare name**: `alpine:latest`, `python:3.12-slim`, `busybox`. For any other registry, use the **explicit `<registry>#<path>` form**: `nvcr.io#nvidia/pytorch:24.12-py3`. **Do not** use `docker.io#library/<name>` — that combination breaks enroot's manifest fetcher with a JSON parse error.

For ML workloads we still recommend NVIDIA NGC images (`nvcr.io#...`) over Docker Hub: they ship CUDA + NCCL + cuDNN matched to the host drivers, and you avoid Docker Hub's per-IP rate limits. If you need a specific docker.io image mirrored to NGC, ask [@anthony](#contact).

### Useful commands

| Command | What it does |
|---|---|
| `sinfo` | partition + node state |
| `squeue` / `squeue -u $USER` | current queue |
| `sbatch <file>` | submit a job |
| `scancel <jobid>` | kill your job |
| `sacct -u $USER -S today` | your job history today |
| `sshare -A stanford-<tier> -u $USER` | your remaining GPU-hour budget |
| `scontrol show job <jobid>` | details on a specific job |

## Partitions and your hour budget

| Partition | Max walltime | Max nodes | Who can use |
|---|---|---|---|
| `small` ★ | 24h | 1 (≤8 GPU) | everyone — default |
| `medium` | 5d | 1 (≤8 GPU) | everyone |
| `big` | 5d | 1 (≤8 GPU) | everyone |

★ default partition — jobs go here if you don't specify `--partition`.

Your GPU-hour budget is enforced by Slurm. When you hit the cap, new jobs queue indefinitely until your budget resets or [@anthony](#contact) bumps it. Check usage with `sshare`.

## Storage

- **`/home/<you>`** is yours. Shared across all login pods and worker nodes — your code, datasets, and checkpoints live here.
- **`/home/_shared/models`** and **`/home/_shared/datasets`** are read-only shared caches. Ask [@anthony](#contact) before pre-staging large things there.
- **Soft cap: ~1 TB per student**. There's no hard quota but [@anthony](#contact) monitors usage. If you're going over, message him so we can plan.
- Big checkpoint archives: don't accumulate them on `/home`. Delete old runs or move to object storage (ask [@anthony](#contact) for the bucket).

## Rules — read these

You technically have cluster-admin permissions in Kubernetes because of how Omniva grants access. **Don't use them.** Specifically:

1. **Only touch your own pod.** Don't `kubectl exec` into other students' login pods. Don't kill or modify infrastructure pods (`slurm-controller-*`, `slurm-accounting-*`, `mariadb-*`, `slurm-worker-*`).
2. **Don't modify Slurm itself.** Leave partitions, accounts, QoS, and ConfigMaps alone. Don't run `scontrol update partition=...` or `sacctmgr modify ...`. If you need a hour-cap bump or partition change, ask.
3. **Submit jobs only as yourself.** Don't impersonate other users by changing UID or editing /etc/passwd. Slurm tracks per-user hours — using another account is a class hour-cap violation.
4. **Don't touch the ValidatingAdmissionPolicy or RBAC.** These are the lightweight guardrails that keep the cluster sharable.
5. **Don't run jobs in the login pod.** The login pod has small CPU/RAM limits and no GPU. Use `sbatch` for everything compute-heavy.
6. **Don't `helm upgrade` or `kubectl delete` infrastructure resources.** If something looks broken, ping [@anthony](#contact) first.
7. **All actions are logged in Teleport.** Every `kubectl` command + exec session is recorded. Behave accordingly.

If you break any of these — usually by accident — ping [@anthony](#contact) and we'll fix it together. The cluster can be redeployed from scripts in ~20 minutes if it really goes sideways, so nothing is unrecoverable; but every redeploy costs everyone else's time.

## Using Claude Code (or another AI coding assistant)

**Claude Code is pre-installed on your login pod.** First run will prompt you to log in with your own Anthropic account:

```bash
claude          # interactive — first run does the OAuth flow
                # copy the URL, complete in a browser, paste the token back
```

After logging in, your credentials will be stored in `~/.config/claude/`, protected by your home dir's 0700 perms. **Cycle your OAuth session at `console.anthropic.com` when the class ends.**

Either way (CLI on the pod or your laptop), drop a copy of `CLAUDE.md.template` ([@anthony](#contact) will share it) into your project repo root, renamed to `CLAUDE.md`, with `MY_USERNAME` replaced with your username. It gives the AI the cluster-specific context — partitions, `nvcr.io`-only image policy, the `runuser` pattern, your hour budget, etc. — so it stops suggesting things that don't work here (e.g., `docker.io` images, multi-node DDP, SSH-based file copy).

Without this file, expect the AI to confidently propose patterns from generic SLURM/k8s docs that will fail on our setup.

## Getting help

- Job stuck in PENDING? `squeue --start -j <jobid>` shows estimated start
- Out of hours? Ping [@anthony](#contact)
- Container pull failing? Double-check it's `nvcr.io` not `docker.io`
- Pod won't start? Don't touch it — `kubectl describe pod -n slurm <pod>` then send the output to [@anthony](#contact)
- Anything else weird? Ask before improvising. We're sharing infrastructure.

## Contact

Ping **`@anthony`** in the class **Slack workspace** for anything cluster-related — questions, hour-cap bumps, broken pods, container debugging, etc. The Slack invite should have been emailed to your Stanford address when you were provisioned; if you can't find it, reply to that email or message [anthony@amppbc.com](mailto:anthony@amppbc.com) and we'll resend.

## Quick reference (paste in your terminal)

```bash
export YOU=<your-username>                         # set once
POD() { kubectl get pod -n slurm -l stanford/user=$YOU -o jsonpath='{.items[0].metadata.name}'; }
shell() { kubectl exec -it -n slurm $(POD) -c login -- runuser -l $YOU; }
run()   { kubectl exec      -n slurm $(POD) -c login -- runuser -l $YOU -c "$*"; }
copy()  { kubectl cp "$1" "slurm/$(POD):/home/$YOU/$(basename $1)" -c login; }
# usage:
#   copy ~/code/train.py
#   run "cd ~ && sbatch train.sbatch"
#   shell
```
