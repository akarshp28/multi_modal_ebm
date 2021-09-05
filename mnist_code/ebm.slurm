#!/bin/sh
#SBATCH --job-name mnist
#SBATCH --partition=GPU
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --time=150:00:00
#SBATCH --mem-per-cpu=8GB


module load cuda/10.2
module load cudnn/7.6.5-cuda10.2

python3 ebm_mnist_v18.py -path /users/apokkunu/trial/mnist/