#!/bin/bash
##################################
#        SLURM DIRECTIVES        #
##################################
# For example, if you run 2 models with 30 tasks each, use 0-59;
# if you run only one model (30 tasks), change accordingly.
#SBATCH --job-name=FluxArrayCombinedModels
#SBATCH --array=0-11%12
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=60G
#SBATCH --time=15:00:00

# Export necessary environment variables (adjust paths as needed)

#############################
#   ENVIRONMENT SETUP       #
#############################
#Add your own env

echo "=== SLURM JOB INFO =========================="
echo "Job ID:       $SLURM_JOB_ID"
echo "Array Job ID: $SLURM_ARRAY_JOB_ID"
echo "Task ID:      $SLURM_ARRAY_TASK_ID"
echo "Running on:   $(hostname)"
echo "Workdir:      $(pwd)"
echo "============================================="


if [ "$SLURM_ARRAY_TASK_ID" -eq 0 ]; then
  category="body_parts"
elif [ "$SLURM_ARRAY_TASK_ID" -eq 1 ]; then
  category="things_parts"
elif [ "$SLURM_ARRAY_TASK_ID" -eq 2 ]; then
  category="physics"
elif [ "$SLURM_ARRAY_TASK_ID" -eq 3 ]; then
  category="negation"
elif [ "$SLURM_ARRAY_TASK_ID" -eq 4 ]; then
  category="shapes"
elif [ "$SLURM_ARRAY_TASK_ID" -eq 5 ]; then
  category="counting"
elif [ "$SLURM_ARRAY_TASK_ID" -eq 6 ]; then
  category="spacial_relation"
elif [ "$SLURM_ARRAY_TASK_ID" -eq 7 ]; then
  category="size_relation"
elif [ "$SLURM_ARRAY_TASK_ID" -eq 8 ]; then
  category="text_display"
elif [ "$SLURM_ARRAY_TASK_ID" -eq 9 ]; then
  category="action"
elif [ "$SLURM_ARRAY_TASK_ID" -eq 10 ]; then
  category="bias"
elif [ "$SLURM_ARRAY_TASK_ID" -eq 11 ]; then
  category="interaction"
fi


#############################
#       MAIN EXECUTION      #
#############################
# Path to the Python script that drives Flux generation.
CMD_PY="croc_hum/img_gen/img_gen_core_prompts.py"

srun --gres=gpu:1 python "$CMD_PY" --type m1 --category "$category"
