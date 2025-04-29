#!/bin/bash
#SBATCH --job-name=ner_training
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem-per-cpu=8000
#SBATCH --time=02:00:00
#SBATCH --output=ner_train_%j.out
#SBATCH --error=ner_train_%j.err

source /software/rome/r24.04/Miniconda3/24.7.1-0/etc/profile.d/conda.sh
conda activate universal-ner

echo "Starting NER training job on $(hostname) at $(date)"

python train_ner.py \
  --group A \
  --num_samples 1000 \
  --comparisons_file "energy_ner_llm/energy_ner_llm/src/mod_distil_by_cot/comparisons.json" \
  --max_len 128 \
  --train_batch_size 4 \
  --valid_batch_size 2 \
  --epochs 20 \
  --learning_rate 1e-05 \
  --max_grad_norm 10 \
  --lr_decay 0.95 \
  --num_runs 1

echo "NER training job completed at $(date)"
