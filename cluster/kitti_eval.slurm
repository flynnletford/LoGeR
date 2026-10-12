#!/bin/bash
# KITTI long-eval inference: one array task per sequence (00-10), each running
# LoGeR (win 32) then LoGeR_star (win 64), as in eval/run_kitti.sh.
# Trajectories go to results/viser_pi3_kitti/<ckpt>/<seq>.txt (results -> scratch).
# Submit from the repo root: sbatch cluster/kitti_eval.slurm
#SBATCH --job-name=loger_kitti
#SBATCH --output=cluster/logs/kitti_%A_%a.out
#SBATCH --error=cluster/logs/kitti_%A_%a.err
#SBATCH --array=0-10
#SBATCH --time=4:00:00
#SBATCH --account=ls_polle
#SBATCH --gpus=rtx_4090:1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=16G
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=fletford@ethz.ch

source /cluster/project/rsl/fletford/miniconda3/etc/profile.d/conda.sh
conda activate loger

set -eo pipefail

cd /cluster/home/fletford/MastersThesis/LoGeR

# torch.compile caches off the home quota.
CACHE=/cluster/scratch/fletford/LoGeR/cache
export TRITON_CACHE_DIR=$CACHE/triton
export TORCHINDUCTOR_CACHE_DIR=$CACHE/inductor
export PYTHONUNBUFFERED=1

# Lets LoGeR_star at win 64 fit in 24 GB by reducing allocator fragmentation.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

SEQ=$(printf %02d "$SLURM_ARRAY_TASK_ID")
echo "Job $SLURM_ARRAY_JOB_ID task $SLURM_ARRAY_TASK_ID on $(hostname): KITTI seq $SEQ"
nvidia-smi

# demo_viser.py also dumps the full predictions (points/conf/poses, several GB)
# to ./results_pi3/<seq>_image_2_0_10000_1.pt relative to the cwd, with a name
# that does not include the model. Run from a per-model scratch dir so these
# dumps neither fill the home quota nor overwrite each other.
REPO_ROOT=$(pwd)
for spec in LoGeR:32 LoGeR_star:64; do
    MODEL=${spec%:*}
    WIN=${spec#*:}
    PRED=/cluster/scratch/fletford/LoGeR/predictions/$MODEL/$SEQ
    mkdir -p "$PRED"
    (cd "$PRED" && bash "$REPO_ROOT/eval/demo_run_longeval.sh" --model "$MODEL" --mode kitti --seq "$SEQ" --win "$WIN")
done
