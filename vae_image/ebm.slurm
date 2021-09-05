#!/bin/sh
#SBATCH --job-name ebm
#SBATCH --partition=GPU
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --time=500:00:00
#SBATCH --mem-per-cpu=16GB


module load cuda/10.2
module load cudnn/7.6.5-cuda10.2

python3 ebm_text2image_gridsearch.py -path /users/apokkunu/trial/img/