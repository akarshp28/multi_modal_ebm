#!/bin/sh
#SBATCH --job-name vaeimg
#SBATCH --partition=GPU
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --mem=48GB
#SBATCH --time=24:00:00



module load cuda/10.2
module load cudnn/7.6.5-cuda10.2

python3 train.py --epoch=50 --dataset='mnist' --mode='train'