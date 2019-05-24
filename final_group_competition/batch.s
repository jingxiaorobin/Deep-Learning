#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --time=100:00:00
#SBATCH --mem=50GB
#SBATCH --job-name=halite
#SBATCH --mail-type=END
#SBATCH --mail-user=aw2797@nyu.edu
#SBATCH --output=results_eval_transformed.out

module purge

python eval.py --data_dir ../ssl_data_96
