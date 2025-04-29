#!/bin/bash
#SBATCH --job-name=annotation_job
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem-per-cpu=8000
#SBATCH --time=02:00:00
#SBATCH --output=annotation_%j.out
#SBATCH --error=annotation_%j.err

source /software/rome/r24.04/Miniconda3/24.7.1-0/etc/profile.d/conda.sh
conda activate universal-ner

echo "Starting annotation job on $(hostname) at $(date)"

python distillation_aka_annotation.py

echo "Annotation job completed at $(date)"
