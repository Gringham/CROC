#!/bin/bash
#SBATCH --job-name=process_%a               # use array index for job name
#SBATCH --output=logs/croc_hum_%a.out     # %A = master job ID, %a = array index
#SBATCH --error=logs/croc_hum_%a.err
#SBATCH --cpus-per-task=5
#SBATCH --mem=70G
#SBATCH --partition=gpu-vram-48gb
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --array=0-8%8                    

# make sure the logs directory exists
mkdir -p logs

echo "[$(date)] Starting job with array index: $SLURM_ARRAY_TASK_ID"

CATEGORIES=( body_parts negation counting shapes size_relation spacial_relation action things_parts )
CATEGORY=${CATEGORIES[$SLURM_ARRAY_TASK_ID]}

# sanity check
if [ -z "$CATEGORY" ]; then
  echo "Error: invalid array index $SLURM_ARRAY_TASK_ID"
  exit 1
fi
echo "[$(date)] Running category: $CATEGORY"


source <ENVIRONMENT>


HF_HOME="<CACHE_DIR>" python croc_hum/apply_core_metrics.py --category "$CATEGORY" --base_dir "<PATH to extracted CROC_hum>" --subset PickScore
