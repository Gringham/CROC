#!/bin/bash
#SBATCH --job-name=process_%a
#SBATCH --output=logs/gen_images_%a.out
#SBATCH --error=logs/gen_images_%a.err
#SBATCH --cpus-per-task=5
#SBATCH --mem=70G
#SBATCH --partition=gpu-vram-48gb
#SBATCH --gres=gpu:1
#SBATCH --time=16:00:00
#SBATCH --array=0-8%9

mkdir -p logs

source <ENVIRONMENT>

python -c "import torch; print(torch.cuda.is_available())"

MODEL="black-forest-labs/FLUX.1-schnell"
MODEL_SHORT="flux"

#MODEL="stabilityai/stable-diffusion-3.5-large-turbo"
#MODEL_SHORT="sd"

echo "Job started at: $(date)"

# Run (script reads $SLURM_ARRAY_TASK_ID as --part automatically)
CUDA_VISIBLE_DEVICES=0 HF_HOME=<CACHE_DIR> HF_TOKEN="<HF_TOKEN>" python croc_syn/img_gen/img_gen.py \
--split_index $SLURM_ARRAY_TASK_ID \
--num_splits 9 \
--input_path "outputs/croc_syn/prompt_gen/3_extracted_prompts_${MODEL_SHORT}_ds.jsonl" \
--model_id "$MODEL"

CUDA_VISIBLE_DEVICES=0 HF_HOME=<CACHE_DIR> HF_TOKEN="<HF_TOKEN>" python croc_syn/img_gen/img_gen.py \
--split_index $SLURM_ARRAY_TASK_ID \
--num_splits 9 \
--input_path "outputs/croc_syn/prompt_gen/3_extracted_prompts_${MODEL_SHORT}_qwen.jsonl" \
--model_id "$MODEL"


