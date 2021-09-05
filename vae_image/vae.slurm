#!/bin/sh
#SBATCH --job-name vaeimg
#SBATCH --partition=GPU
#SBATCH --gres=gpu:4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --mem=48GB
#SBATCH --time=100:00:00



module load cuda/10.2
module load cudnn/7.6.5-cuda10.2

python3 vae_trial_38.py -path /users/apokkunu/trial/img/