#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --time=100:00:00
#SBATCH --mem=50GB
#SBATCH --job-name=vae_classifier
#SBATCH --mail-type=END
#SBATCH --mail-user=aw2797@nyu.edu
#SBATCH --output=results.out

module purge

python train_classifier.py --data ../../ssl_data_96/supervised --model models/model_3.pth
