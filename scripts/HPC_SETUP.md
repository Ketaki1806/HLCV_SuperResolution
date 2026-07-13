# HPC setup guide (conduit)

**Recommended:** use the all-in-one cluster script after copying the project:

```bash
ssh hlcv_team019@conduit.hpc.uni-saarland.de
cd ~/super_resolution
bash scripts/hpc_setup_cluster.sh
```

This installs the conda env, pip dependencies, and sets `~/.bashrc` paths. **COCO is assumed to already exist** on the cluster.

Options:

```bash
bash scripts/hpc_setup_cluster.sh --skip-env          # only set paths
```

---

Run these steps **in order** if you prefer manual setup. Commands marked **LOCAL** run on your Windows machine.
Commands marked **CLUSTER** run after `ssh hlcv_team019@conduit.hpc.uni-saarland.de`.

Replace paths if your project lives somewhere else locally.

---

## Step 1 — Transfer `environment.yml` to the cluster (LOCAL)

From PowerShell on your laptop (project root):

```powershell
cd "D:\Uni\SS2026\HLCV\Super resolution"
scp environment.yml hlcv_team019@conduit.hpc.uni-saarland.de:~/
```

---

## Step 2 — Create the conda environment (CLUSTER)

SSH in, then:

```bash
ssh hlcv_team019@conduit.hpc.uni-saarland.de

# Load conda (miniconda3 is already installed on your account)
source ~/miniconda3/etc/profile.d/conda.sh

# Create env from the transferred file (~5–15 min)
conda env create -f ~/environment.yml

# Activate it
conda activate super_resolution

# Quick sanity check
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
```

If `pytorch-cuda=12.1` fails, edit `~/environment.yml` and try `pytorch-cuda=11.8`, then:

```bash
conda env create -f ~/environment.yml
```

---

## Step 3 — Transfer the full project to the cluster (LOCAL)

Option A — **scp** (simple, no git remote needed):

```powershell
cd "D:\Uni\SS2026\HLCV\Super resolution"
scp -r . hlcv_team019@conduit.hpc.uni-saarland.de:~/super_resolution
```

Option B — **git** (if the repo is pushed to GitHub/GitLab):

```bash
# on CLUSTER only
cd ~
git clone <YOUR_REPO_URL> super_resolution
```

---

## Step 4 — Set project paths on the cluster (CLUSTER)

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate super_resolution

cd ~/super_resolution

# Persist env vars for future sessions
cat >> ~/.bashrc << 'EOF'
export DATA_ROOT=/scratch/teaching/hlcv/hlcv_team019
export COCO_ROOT=$DATA_ROOT/coco
export PROJECT_ROOT=$HOME/super_resolution
EOF
source ~/.bashrc

# Add project to Python path for this session
export PYTHONPATH=$PROJECT_ROOT:$PYTHONPATH
```

---

## Step 5 — Verify COCO exists on the cluster (CLUSTER)

COCO is provided centrally for the course under `/scratch/teaching/hlcv/hlcv_team019` (team-specific path).

```bash
ls -lah $COCO_ROOT
ls $COCO_ROOT/val2017 | wc -l
ls -lah $COCO_ROOT/annotations | head
```

---

## Step 6 — Verify dataset loading (CLUSTER)

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate super_resolution
cd ~/super_resolution
export PYTHONPATH=$PWD:$PYTHONPATH

python - << 'EOF'
from src.data import load_coco_dataset

dataset = load_coco_dataset()
sample = dataset[0]
print(f"images: {len(dataset)}, sample shape: {tuple(sample.shape)}")
EOF
```

Expected output: `images: 5000, sample shape: (3, 64, 64)` (for val2017 with patch_size 64).

Optional pytest (skips if COCO missing):

```bash
pytest tests/test_datasets.py -v
```

---

## Step 7 — Request a GPU interactive session (CLUSTER)

Use Code Server via https://ood.hpc.uni-saarland.de/ for interactive work,
or submit a batch job. Minimal GPU smoke test on a compute node:

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate super_resolution
cd ~/super_resolution
export PYTHONPATH=$PWD:$PYTHONPATH

python -c "import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no GPU')"
```

---

## Quick reference (every new SSH session)

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate super_resolution
cd ~/super_resolution
export PYTHONPATH=$PWD:$PYTHONPATH
# COCO_ROOT and PROJECT_ROOT are loaded from ~/.bashrc after Step 4
```

---

## Troubleshooting

| Problem | Fix |
|--------|-----|
| `conda: command not found` | `source ~/miniconda3/etc/profile.d/conda.sh` |
| `ModuleNotFoundError: src` | `export PYTHONPATH=$HOME/super_resolution:$PYTHONPATH` |
| COCO not found | `echo $COCO_ROOT` should be `$HOME/data/coco`; re-run Step 5 |
| Conda channel errors | Retry later, or use `pytorch-cuda=11.8` in `environment.yml` |
| SSL error downloading COCO | Run download on cluster (not locally); cluster network should work |
