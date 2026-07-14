# HPC setup guide (conduit)

Follow **`gpu_instructions/GPU-instructions.md`** for Condor job submission. This doc covers conda, paths, and running project scripts.

---

## Cluster architecture (important)

| What | Where | What you do |
|------|--------|-------------|
| **Login / submit node** | `conduit.hpc.uni-saarland.de` | SSH here, install conda, edit code, `condor_submit` |
| **Execute nodes (GPU workers)** | Not directly accessible | Jobs run here after Condor schedules them |
| **Team data** | `/scratch/teaching/hlcv/hlcv_team019/` | COCO, preprocessed images, large outputs |
| **Your code** | `~/super_resolution` | Git clone of the project |

There is **no separate “SSH submit node”** to log into. You SSH to **conduit** (the login node), and that **is** where you submit Condor jobs:

```bash
ssh hlcv_team019@conduit.hpc.uni-saarland.de
cd ~/gpu_instructions   # or your project
condor_submit pytorch_docker.sub
condor_q
```

You do **not** SSH to GPU machines. Condor copies your job to a worker and runs it there.

---

## Why conda / pip failed (what went wrong)

1. **SSH dropped** during `conda env create` (“Verifying transaction”) → left a **broken** `~/miniconda3/envs/super_resolution/` folder (exists but no `bin/python`).

2. **`conda create` failed** with “prefix already exists” → next commands ran without a working env.

3. **`pip` hit system Python** → “externally-managed-environment” because `conda activate` did not succeed (you were not in the conda env).

4. **Condor jobs are the wrong place to install packages.** Install conda env on the **login node** first. Condor only **runs** your script using that env’s Python path (see `gpu_instructions/execute.sh`).

### Fix broken env (login node)

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda deactivate 2>/dev/null || true
conda deactivate 2>/dev/null || true
rm -rf ~/miniconda3/envs/super_resolution
```

---

## Option A — Minimal env (downsampling only, ~2 min)

Use this to run downsampling + PSNR/SSIM on COCO without PyTorch:

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda create -n super_resolution python=3.10 pip -y
conda activate super_resolution
which python   # MUST be .../miniconda3/envs/super_resolution/bin/python
pip install opencv-python-headless pycocotools tqdm numpy
python -c "import cv2; from pycocotools.coco import COCO; print('ok')"
```

Run downsampling on the **login node** (no GPU needed):

```bash
export COCO_ROOT=/scratch/teaching/hlcv/hlcv_team019/data/coco
cd ~/super_resolution
export PYTHONPATH=$PWD:$PYTHONPATH

python scripts/preprocessing/downsample_coco_val2017.py \
  --coco-root "$COCO_ROOT" \
  --output-dir /scratch/teaching/hlcv/hlcv_team019/coco_preprocessed/val2017_lr_x2_100 \
  --scale 2 --blur --blur-kernel 5 --blur-sigma 2 \
  --max-images 100 --metrics
```

Results are saved to `<output-dir>/downsample_quality.json` (per-image PSNR/SSIM + mean).

View summary on the cluster:

```bash
python -c "
import json
p='/scratch/teaching/hlcv/hlcv_team019/coco_preprocessed/val2017_lr_x2_100/downsample_quality.json'
d=json.load(open(p))
print('Mean:', d['metrics_mean'])
print('Images:', d['count_metrics'])
"
```

Re-evaluate without re-downsampling (HR vs bicubic-upsampled LR):

```bash
python scripts/eval_downsample_quality.py \
  --hr-dir /scratch/teaching/hlcv/hlcv_team019/data/coco/images/val2017 \
  --lr-dir /scratch/teaching/hlcv/hlcv_team019/coco_preprocessed/val2017_lr_x2_100 \
  --output ~/super_resolution/results/downsample_quality.json
```

Visualize metrics as charts (needs `matplotlib`):

```bash
pip install matplotlib
python scripts/visualize_downsample_quality.py \
  --input ~/super_resolution/results/downsample_quality.json \
  --output-dir ~/super_resolution/results/figures/downsample_quality
```

Charts saved:
- `downsample_quality_overview.png` — histograms, scatter, mean bars
- `downsample_quality_psnr_per_image.png` — sorted per-image PSNR
- `downsample_quality_ssim_per_image.png` — sorted per-image SSIM
- `downsample_quality_extremes.png` — best/worst 5 by PSNR

---

---

## Option B — Full env (PyTorch + YOLO, 15–30+ min)

Use **`tmux`** so SSH disconnect does not kill the install:

```bash
tmux new -s conda
source ~/miniconda3/etc/profile.d/conda.sh
cd ~/super_resolution
conda env create -f environment.yml
# detach: Ctrl+B then D
# reattach later: tmux attach -t conda
```

If `pytorch-cuda=12.1` fails, edit `environment.yml` to `pytorch-cuda=11.8` and retry.

Or use the course env (often more reliable on this cluster):

```bash
cd ~/gpu_instructions
conda env create -f environment.yml   # creates env name: hlcv
conda activate hlcv
```

---

## Paths (add to `~/.bashrc`)

```bash
export DATA_ROOT=/scratch/teaching/hlcv/hlcv_team019
export COCO_ROOT=$DATA_ROOT/data/coco
export PROJECT_ROOT=$HOME/super_resolution
export PYTHONPATH=$PROJECT_ROOT:$PYTHONPATH
```

Verify COCO:

```bash
ls $COCO_ROOT/images | head
ls $COCO_ROOT/annotations/instances_val2017.json
```

---

## YOLO downstream eval (SR strategies on downsampled val2017)

Requires **full PyTorch env** (`environment.yml` or course `hlcv` env) with `ultralytics`.

Evaluates **baseline**, **ESPCN 2×**, **FSRCNN 2×**, and **AnyUp** (feature-level in YOLO neck) on LR COCO val → YOLO detections → mAP@0.5.

```bash
export COCO_ROOT=/scratch/teaching/hlcv/hlcv_team019/data/coco
cd ~/super_resolution
export PYTHONPATH=$PWD:$PYTHONPATH

# Quick test (100 LR images you already downsampled)
python scripts/eval_yolo_sr_coco.py \
  --lr-dir /scratch/teaching/hlcv/hlcv_team019/coco_preprocessed/val2017_lr_x2_100 \
  --ann-file $COCO_ROOT/annotations/instances_val2017.json \
  --max-images 100 \
  --output-dir ~/super_resolution/runs

# Full val2017 (after downsample without --max-images)
python scripts/eval_yolo_sr_coco.py \
  --lr-dir /scratch/teaching/hlcv/hlcv_team019/coco_preprocessed/val2017_lr_x2 \
  --ann-file $COCO_ROOT/annotations/instances_val2017.json \
  --output-dir ~/super_resolution/runs
```

Outputs per strategy:
- `runs/yolov8n_{baseline,espcn2x,fsrcnn2x,anyup}/predictions.json`
- `runs/yolov8n_{strategy}/size_results.json` (mAP@0.5 all/small/medium/large)
- `runs/yolov8n_sr_comparison.json` (summary table)

Run on a **GPU node** (login node for small `--max-images 10` smoke test only).

---

1. Edit `~/gpu_instructions/execute.sh` — set your team paths and conda Python:
   ```bash
   PYTHON_SCRIPT_PATH="/home/hlcv_team019/super_resolution"
   CONDA_PYTHON_BINARY_PATH="/home/hlcv_team019/miniconda3/envs/super_resolution/bin/python"
   ```

2. Edit `~/gpu_instructions/pytorch_docker.sub` — set `arguments = scripts/load_yolo.py` (or your script).

3. Submit from login node:
   ```bash
   mkdir -p ~/condor_logs
   cd ~/gpu_instructions
   condor_submit pytorch_docker.sub
   condor_q
   cat ~/condor_logs/*.err
   ```

---

## Every new SSH session

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate super_resolution   # or hlcv
cd ~/super_resolution
export PYTHONPATH=$PWD:$PYTHONPATH
```

---

## Troubleshooting

| Problem | Cause | Fix |
|--------|--------|-----|
| `prefix already exists` | Broken partial env | `conda deactivate` then `rm -rf ~/miniconda3/envs/super_resolution` |
| `externally-managed-environment` | Using system `pip`, not conda | `conda activate super_resolution` then check `which pip` |
| `python: command not found` | Env broken / not activated | Recreate env (Option A) |
| Condor: `python: No such file` | Wrong path in `execute.sh` | Use absolute path to env’s `bin/python` |
| SSH disconnect during conda | Long install killed | Use `tmux` or minimal env first |
| `libGL.so.1: cannot open shared object file` | `opencv-python` needs GUI libs on headless nodes | `pip uninstall opencv-python -y && pip install opencv-python-headless` |
