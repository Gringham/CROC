#!/bin/bash
#SBATCH --job-name=process_%a
#SBATCH --output=logs/compute_scores%a.out
#SBATCH --error=logs/compute_scores%a.err
#SBATCH --cpus-per-task=5
#SBATCH --mem=70G
#SBATCH --partition=gpu-vram-48gb
#SBATCH --gres=gpu:1
#SBATCH --time=16:00:00
#SBATCH --array=0-9%10

mkdir -p logs

source <ENVIRONMENT>

python -c "import torch; print(torch.cuda.is_available())"

IMG_MODEL_SHORT="flux"
#IMG_MODEL_SHORT="sd"

TEXT_MODEL_SHORT="qwen"
#TEXT_MODEL_SHORT="ds"

echo "Job started at: $(date)"

CUDA_VISIBLE_DEVICES=0  HF_HOME=<CACHE_DIR> HF_TOKEN="<HF_TOKEN>" python metrics/apply_bench.py \
--prompt_file "outputs/croc_syn/prompt_gen/3_extracted_prompts_${IMG_MODEL_SHORT}_${TEXT_MODEL_SHORT}.jsonl" \
--img_dir "outputs/croc_syn/img_gen/${IMG_MODEL_SHORT}/${TEXT_MODEL_SHORT}" \
--chunk_part ${SLURM_ARRAY_TASK_ID} \
--prompt_num 1 \
--img_num 5 \
--chunk_num 10 \
--output_dir "outputs/croc_syn/metrics/scores/${IMG_MODEL_SHORT}_${TEXT_MODEL_SHORT}"