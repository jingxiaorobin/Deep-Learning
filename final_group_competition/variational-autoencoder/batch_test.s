#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --time=100:00:00
#SBATCH --mem=50GB
#SBATCH --job-name=vae_test
#SBATCH --mail-type=END
#SBATCH --mail-user=aw2797@nyu.edu
#SBATCH --output=results_test.out

module purge

python train_test.py --data ../../unsupervised --checkpoint models_test/model_149.pth --epochs 150
