#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --time=100:00:00
#SBATCH --mem=50GB
#SBATCH --job-name=vae
#SBATCH --mail-type=END
#SBATCH --mail-user=aw2797@nyu.edu
#SBATCH --output=results.out

module purge

python train.py --data ../../ssl_data_96/unsupervised --batch-size 8
