#!/bin/sh
#SBATCH --job-name nvae
#SBATCH --partition=GPU
#SBATCH --gres=gpu:2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --mem=48GB
#SBATCH --time=200:00:00



module load cuda/10.2
module load cudnn/7.6.5-cuda10.2

python3 train_multigpu.py -path /users/apokkunu/trial/img/nvae/